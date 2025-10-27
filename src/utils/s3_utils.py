# utils.py
"""
Lightweight S3 utilities: pass s3://bucket/key paths and go.

Requirements:
  - boto3  (pip install boto3)
Optional:
  - pandas, pyarrow or fastparquet (for DataFrame helpers)

Credentials:
  - Respects AWS standard resolution: env vars, shared credentials, or IAM role.

Example:
  from utils import (
      s3_exists, s3_read_text, s3_write_text, s3_list, s3_copy, s3_move,
      s3_upload_file, s3_download_file, s3_delete_prefix, s3_presigned_url
  )

  s3_write_text("s3://my-bucket/path/hello.txt", "hi there")
  print(s3_read_text("s3://my-bucket/path/hello.txt"))
  for obj in s3_list("s3://my-bucket/path/", recursive=True, suffix=".txt"):
      print(obj.key, obj.size)
"""

from __future__ import annotations

import sys
import json
import os
import re
import io
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generator, Iterable, List, Optional, Tuple
from functools import wraps

import pandas as pd
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

from src.exception import TSException
from src.logger import logging

# at top of utils.py you already have: import os

def _truthy(s: str | None) -> bool:
    return str(s).lower() in {"1", "true", "yes", "on"} if s is not None else False

def _env_region() -> Optional[str]:
    # boto3 already reads envs, but we also use it to set defaults when args are None
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

def _env_endpoint() -> Optional[str]:
    # support common names for S3-compatible services
    return (
        os.getenv("AWS_S3_ENDPOINT_URL")
        or os.getenv("S3_ENDPOINT_URL")
        or os.getenv("MINIO_ENDPOINT")
        or os.getenv("LOCALSTACK_S3_URL")
    )

def _env_addressing_style() -> str:
    # allow forcing path-style (MinIO/LocalStack) via env
    if _truthy(os.getenv("AWS_S3_FORCE_PATH_STYLE")):
        return "path"
    return os.getenv("AWS_S3_ADDRESSING_STYLE", "auto")


# ---------- Core: client / path parsing ----------

def _client(region_name: Optional[str] = None,
            endpoint_url: Optional[str] = None,
            retries: int = 10,
            connect_timeout: int = 5,
            read_timeout: int = 60):
    """
    Create a tuned S3 client with sensible retries/timeouts.
    Env overrides (if args are None):
      - AWS_REGION / AWS_DEFAULT_REGION
      - AWS_S3_ENDPOINT_URL / S3_ENDPOINT_URL / MINIO_ENDPOINT / LOCALSTACK_S3_URL
      - AWS_S3_FORCE_PATH_STYLE=1 (forces path style)
      - AWS_PROFILE (use a shared-credentials profile)
    """
    try:
        region_name  = region_name  or _env_region()
        endpoint_url = endpoint_url or _env_endpoint()

        logging.info("Creating a s3 client config (region=%s endpoint=%s)", region_name, endpoint_url)
        cfg = Config(
            retries={"max_attempts": retries, "mode": "adaptive"},
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            s3={"addressing_style": _env_addressing_style()},
            signature_version="s3v4",  # works well across AWS/MinIO/LocalStack
        )

        profile = os.getenv("AWS_PROFILE")
        if profile:
            logging.info("Using AWS profile: %s", profile)
            session = boto3.Session(profile_name=profile, region_name=region_name)
            client = session.client("s3", endpoint_url=endpoint_url, config=cfg)
        else:
            client = boto3.client("s3", region_name=region_name, endpoint_url=endpoint_url, config=cfg)

        logging.info("S3 Client connection successful")
        return client
    except Exception as e:
        logging.exception("Failed to create S3 client")
        raise TSException(e, sys)


_S3_URI_RE = re.compile(r"^s3://(?P<bucket>[^/]+)/?(?P<key>.*)$")

@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str  # can be '' for bucket root

def _parse_s3_uri(s3_uri: str) -> S3ObjectRef:
    m = _S3_URI_RE.match(s3_uri)
    if not m:
        raise ValueError(f"Not a valid S3 URI: {s3_uri!r}")
    return S3ObjectRef(bucket=m.group("bucket"), key=m.group("key"))


# --- new: centralized error handling decorator ---
def _handle_boto_errors(op_name: str):
    """
    Decorator to log and wrap boto/botocore errors into TSException.
    Use for functions performing S3 network/IO actions where a unified
    error conversion is desired.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            logging.info("Starting S3 operation: %s", op_name)
            logging.debug("Args: %s, Kwargs: %s", args, kwargs)
            try:
                result = fn(*args, **kwargs)
                logging.info("Completed S3 operation: %s", op_name)
                return result
            except (ClientError, BotoCoreError) as e:
                logging.exception("S3 %s failed: %s", op_name, e)
                raise TSException(e, sys)
            except Exception as e:
                # catch-all to ensure unexpected errors are logged consistently
                logging.exception("Unexpected error in S3 %s: %s", op_name, e)
                raise TSException(e, sys)
        return wrapper
    return decorator

# ---------- Object metadata / head helpers ----------

def s3_exists(s3_uri: str) -> bool:
    ref = _parse_s3_uri(s3_uri)
    try:
        _client().head_object(Bucket=ref.bucket, Key=ref.key)
        logging.debug("s3_exists: found %s", s3_uri)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        logging.debug("s3_exists: head_object error code=%s for %s", code, s3_uri)
        if code in ("404", "NotFound", "NoSuchKey"):
            return False
        logging.exception("s3_exists encountered unexpected ClientError for %s", s3_uri)
        raise TSException(e, sys)

def s3_path_exists(s3_uri: str) -> bool:
    """
    Check if a file OR folder exists in S3.

    A 'folder' in S3 is just a prefix that contains at least one object.
    This function works for both files (exact key) and folders (prefix).
    """
    ref = _parse_s3_uri(s3_uri)
    client = _client()

    # First, check if it's an exact file
    try:
        client.head_object(Bucket=ref.bucket, Key=ref.key)
        logging.debug("s3_path_exists: exact file found %s", s3_uri)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code not in ("404", "NotFound", "NoSuchKey"):
            logging.exception("s3_path_exists: unexpected error for %s", s3_uri)
            raise TSException(e, sys)

    # If not a file, check if it's a folder (prefix)
    prefix = ref.key.rstrip("/") + "/" if ref.key else ""
    try:
        resp = client.list_objects_v2(Bucket=ref.bucket, Prefix=prefix, MaxKeys=1)
        exists = "Contents" in resp
        logging.debug("s3_path_exists: prefix=%s exists=%s", prefix, exists)
        return exists
    except (ClientError, BotoCoreError) as e:
        logging.exception("s3_path_exists list_objects_v2 failed for %s", s3_uri)
        raise TSException(e, sys)

def s3_head(s3_uri: str) -> dict:
    ref = _parse_s3_uri(s3_uri)
    try:
        return _client().head_object(Bucket=ref.bucket, Key=ref.key)
    except (ClientError, BotoCoreError) as e:
        logging.exception("s3_head failed for %s", s3_uri)
        raise TSException(e, sys)

def s3_size(s3_uri: str) -> int:
    return int(s3_head(s3_uri)["ContentLength"])

def s3_etag(s3_uri: str) -> str:
    return s3_head(s3_uri)["ETag"].strip('"')

def s3_last_modified(s3_uri: str) -> datetime:
    dt = s3_head(s3_uri)["LastModified"]
    # ensure TZ-aware UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------- Read / Write (bytes, text, json) ----------

@_handle_boto_errors("s3_read_bytes")
def s3_read_bytes(s3_uri: str, byte_range: Optional[Tuple[int, int]] = None) -> bytes:
    """
    Read object bytes. Optionally pass byte_range=(start, end_inclusive).
    """
    ref = _parse_s3_uri(s3_uri)
    kwargs = {}
    if byte_range:
        kwargs["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    logging.debug("s3_read_bytes: getting %s range=%s", s3_uri, byte_range)
    resp = _client().get_object(Bucket=ref.bucket, Key=ref.key, **kwargs)
    data = resp["Body"].read()
    logging.debug("s3_read_bytes: read %d bytes from %s", len(data), s3_uri)
    return data

@_handle_boto_errors("s3_read_text")
def s3_read_text(s3_uri: str, encoding: str = "utf-8") -> str:
    return s3_read_bytes(s3_uri).decode(encoding)

@_handle_boto_errors("s3_read_json")
def s3_read_json(s3_uri: str, encoding: str = "utf-8"):
    return json.loads(s3_read_text(s3_uri, encoding=encoding))

@_handle_boto_errors("s3_write_bytes")
def s3_write_bytes(s3_uri: str, data: bytes, content_type: Optional[str] = None):
    ref = _parse_s3_uri(s3_uri)
    kwargs = dict(Bucket=ref.bucket, Key=ref.key, Body=data)
    if content_type:
        kwargs["ContentType"] = content_type
    logging.debug("s3_write_bytes: putting %s bytes to %s", len(data), s3_uri)
    _client().put_object(**kwargs)
    logging.info("s3_write_bytes: upload complete %s", s3_uri)

def s3_write_text(s3_uri: str, text: str, encoding: str = "utf-8", content_type: str = "text/plain; charset=utf-8"):
    s3_write_bytes(s3_uri, text.encode(encoding), content_type=content_type)

@_handle_boto_errors("s3_write_json")
def s3_write_json(s3_uri: str, obj, encoding: str = "utf-8", indent: Optional[int] = 2):
    s = json.dumps(obj, ensure_ascii=False, indent=indent)
    s3_write_text(s3_uri, s, encoding=encoding, content_type="application/json")


# ---------- Upload / Download large files (multipart) ----------

@_handle_boto_errors("s3_upload_file")
def s3_upload_file(local_path: str, s3_uri: str, extra_args: Optional[dict] = None):
    """
    Uploads a local file using boto3's managed multipart uploader.
    """
    ref = _parse_s3_uri(s3_uri)
    logging.info("s3_upload_file: %s -> %s", local_path, s3_uri)
    _client().upload_file(local_path, ref.bucket, ref.key, ExtraArgs=extra_args or {})
    logging.info("s3_upload_file completed: %s", s3_uri)

@_handle_boto_errors("s3_download_file")
def s3_download_file(s3_uri: str, local_path: str):
    ref = _parse_s3_uri(s3_uri)
    logging.info("s3_download_file: %s -> %s", s3_uri, local_path)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    _client().download_file(ref.bucket, ref.key, local_path)
    logging.info("s3_download_file completed: %s", local_path)


# ---------- List & paginate ----------

@dataclass
class S3ListedObject:
    bucket: str
    key: str
    size: int
    etag: str
    last_modified: datetime

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

def s3_list(s3_uri_or_prefix: str,
            recursive: bool = True,
            suffix: Optional[str] = None,
            max_keys: int = 1000) -> Generator[S3ListedObject, None, None]:
    """
    Iterate objects under a prefix. Accepts either 's3://bucket/prefix/' or 's3://bucket'.
    """
    ref = _parse_s3_uri(s3_uri_or_prefix)
    bucket, prefix = ref.bucket, ref.key
    logging.debug("s3_list: bucket=%s prefix=%s recursive=%s suffix=%s", bucket, prefix, recursive, suffix)
    paginator = _client().get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    if not recursive:
        kwargs["Delimiter"] = "/"
    for page in paginator.paginate(**kwargs, PaginationConfig={"PageSize": max_keys}):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if suffix and not key.endswith(suffix):
                continue
            yield S3ListedObject(
                bucket=bucket,
                key=key,
                size=int(obj.get("Size", 0)),
                etag=obj.get("ETag", "").strip('"'),
                last_modified=obj.get("LastModified"),
            )


# ---------- Copy / Move / Delete ----------

@_handle_boto_errors("s3_copy")
def s3_copy(src_s3_uri: str, dst_s3_uri: str, metadata: Optional[dict] = None, metadata_directive: str = "COPY"):
    src = _parse_s3_uri(src_s3_uri)
    dst = _parse_s3_uri(dst_s3_uri)
    logging.info("s3_copy: %s -> %s", src_s3_uri, dst_s3_uri)
    copy_source = {"Bucket": src.bucket, "Key": src.key}
    kwargs = dict(Bucket=dst.bucket, Key=dst.key, CopySource=copy_source, MetadataDirective=metadata_directive)
    if metadata and metadata_directive == "REPLACE":
        kwargs["Metadata"] = metadata
    _client().copy_object(**kwargs)
    logging.info("s3_copy completed: %s", dst_s3_uri)

@_handle_boto_errors("s3_move")
def s3_move(src_s3_uri: str, dst_s3_uri: str):
    s3_copy(src_s3_uri, dst_s3_uri)
    s3_delete_object(src_s3_uri)
    logging.info("s3_move completed: %s -> %s", src_s3_uri, dst_s3_uri)

@_handle_boto_errors("s3_delete_object")
def s3_delete_object(s3_uri: str):
    ref = _parse_s3_uri(s3_uri)
    logging.info("s3_delete_object: %s", s3_uri)
    _client().delete_object(Bucket=ref.bucket, Key=ref.key)
    logging.info("s3_delete_object completed: %s", s3_uri)

@_handle_boto_errors("s3_delete_prefix")
def s3_delete_prefix(s3_prefix_uri: str, batch_size: int = 1000):
    """
    Delete everything under a prefix efficiently (batch delete).
    """
    ref = _parse_s3_uri(s3_prefix_uri)
    client = _client()
    to_delete: List[dict] = []
    logging.info("s3_delete_prefix: %s", s3_prefix_uri)
    for obj in s3_list(s3_prefix_uri, recursive=True):
        to_delete.append({"Key": obj.key})
        if len(to_delete) >= batch_size:
            client.delete_objects(Bucket=ref.bucket, Delete={"Objects": to_delete})
            logging.debug("s3_delete_prefix: deleted batch of %d objects", len(to_delete))
            to_delete.clear()
    if to_delete:
        client.delete_objects(Bucket=ref.bucket, Delete={"Objects": to_delete})
        logging.debug("s3_delete_prefix: deleted final batch of %d objects", len(to_delete))
    logging.info("s3_delete_prefix completed: %s", s3_prefix_uri)


# ---------- Presigned URLs ----------

@_handle_boto_errors("s3_presigned_url")
def s3_presigned_url(s3_uri: str, expires_in: int = 3600, method: str = "get") -> str:
    """
    method: 'get' or 'put'
    """
    ref = _parse_s3_uri(s3_uri)
    method = method.lower()
    if method not in ("get", "put"):
        raise ValueError("method must be 'get' or 'put'")
    op = "get_object" if method == "get" else "put_object"
    logging.debug("s3_presigned_url: %s method=%s expires_in=%d", s3_uri, method, expires_in)
    return _client().generate_presigned_url(
        ClientMethod=op,
        Params={"Bucket": ref.bucket, "Key": ref.key},
        ExpiresIn=expires_in,
    )


# ---------- Simple “sync” helpers (minimal, no fancy diffing) ----------

@_handle_boto_errors("s3_sync_dir_to_prefix")
def s3_sync_dir_to_prefix(local_dir: str, dst_prefix_uri: str):
    """
    Upload all files under local_dir to s3://bucket/prefix/ keeping relative paths.
    """
    if not os.path.isdir(local_dir):
        raise ValueError(f"Not a directory: {local_dir}")
    dst = _parse_s3_uri(dst_prefix_uri)
    base = os.path.abspath(local_dir)
    logging.info("s3_sync_dir_to_prefix: %s -> %s", local_dir, dst_prefix_uri)
    for root, _, files in os.walk(base):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, base).replace(os.sep, "/")
            s3_uri = f"s3://{dst.bucket}/{dst.key.rstrip('/')}/{rel}" if dst.key else f"s3://{dst.bucket}/{rel}"
            s3_upload_file(local_path, s3_uri)

@_handle_boto_errors("s3_sync_prefix_to_dir")
def s3_sync_prefix_to_dir(src_prefix_uri: str, local_dir: str, suffix: Optional[str] = None):
    """
    Download all objects under prefix into local_dir.
    """
    os.makedirs(local_dir, exist_ok=True)
    src = _parse_s3_uri(src_prefix_uri)
    base = local_dir
    logging.info("s3_sync_prefix_to_dir: %s -> %s", src_prefix_uri, local_dir)
    for obj in s3_list(src_prefix_uri, recursive=True, suffix=suffix):
        rel = obj.key[len(src.key):].lstrip("/") if src.key else obj.key
        local_path = os.path.join(base, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_download_file(f"s3://{obj.bucket}/{obj.key}", local_path)



@_handle_boto_errors("s3_read_csv_df")
def s3_read_csv_df(s3_uri: str, **read_csv_kwargs):
    """
    Read CSV on S3 into a pandas DataFrame without s3fs.
    Note: downloads object into memory.
    """
    logging.info("s3_read_csv_df: %s", s3_uri)
    data = s3_read_bytes(s3_uri)
    return pd.read_csv(io.BytesIO(data), **read_csv_kwargs)

@_handle_boto_errors("s3_write_csv_df")
def s3_write_csv_df(s3_uri: str, df, index: bool = False, **to_csv_kwargs):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    logging.info("s3_write_csv_df: %s", s3_uri)
    buf = io.StringIO()
    df.to_csv(buf, index=index, **to_csv_kwargs)
    s3_write_text(s3_uri, buf.getvalue(), content_type="text/csv; charset=utf-8")

@_handle_boto_errors("s3_read_parquet_df")
def s3_read_parquet_df(s3_uri: str, **read_parquet_kwargs):
    """
    Read Parquet into DataFrame (requires pyarrow or fastparquet).
    """
    logging.info("s3_read_parquet_df: %s", s3_uri)
    data = s3_read_bytes(s3_uri)
    return pd.read_parquet(io.BytesIO(data), **read_parquet_kwargs)

@_handle_boto_errors("s3_write_parquet_df")
def s3_write_parquet_df(s3_uri: str, df, index: bool = False, **to_parquet_kwargs):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    logging.info("s3_write_parquet_df: %s", s3_uri)
    buf = io.BytesIO()
    df.to_parquet(buf, index=index, **to_parquet_kwargs)
    s3_write_bytes(s3_uri, buf.getvalue(), content_type="application/octet-stream")


# ---------- Utilities ----------

def s3_md5_of_bytes(data: bytes) -> str:
    """MD5 of bytes (useful for small content integrity)."""
    return hashlib.md5(data).hexdigest()

def guess_content_type_from_ext(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".txt": "text/plain; charset=utf-8",
        ".json": "application/json",
        ".csv": "text/csv; charset=utf-8",
        ".parquet": "application/octet-stream",
        ".html": "text/html; charset=utf-8",
        ".xml": "application/xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".pdf": "application/pdf",
    }.get(ext)

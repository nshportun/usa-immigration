"""S3 helpers: upload, download, existence check, list."""

import io
import json
import os
from pathlib import Path
from typing import Any

import orjson
import structlog

from scripts.aws_config import S3_BUCKET, s3_client, s3_key

log = structlog.get_logger()


def upload_json(data: Any, relative_path: str) -> str:
    """Serialize data as JSON and upload to S3. Returns the full S3 key."""
    key = s3_key(relative_path)
    body = orjson.dumps(data, option=orjson.OPT_INDENT_2)
    s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
    log.info("s3_upload", key=key, bytes=len(body))
    return key


def upload_jsonl(records: list[dict], relative_path: str) -> str:
    """Serialize list of dicts as JSONL and upload to S3."""
    key = s3_key(relative_path)
    lines = b"\n".join(orjson.dumps(r) for r in records)
    s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=lines, ContentType="application/x-ndjson")
    log.info("s3_upload_jsonl", key=key, records=len(records), bytes=len(lines))
    return key


def upload_text(text: str, relative_path: str, content_type: str = "text/plain") -> str:
    key = s3_key(relative_path)
    body = text.encode("utf-8")
    s3_client().put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType=content_type)
    log.info("s3_upload_text", key=key, bytes=len(body))
    return key


def upload_file(local_path: Path, relative_path: str) -> str:
    key = s3_key(relative_path)
    s3_client().upload_file(str(local_path), S3_BUCKET, key)
    log.info("s3_upload_file", key=key, local=str(local_path))
    return key


def download_json(relative_path: str) -> Any:
    key = s3_key(relative_path)
    resp = s3_client().get_object(Bucket=S3_BUCKET, Key=key)
    return orjson.loads(resp["Body"].read())


def download_jsonl(relative_path: str) -> list[dict]:
    key = s3_key(relative_path)
    resp = s3_client().get_object(Bucket=S3_BUCKET, Key=key)
    return [orjson.loads(line) for line in resp["Body"].read().splitlines() if line.strip()]


def exists(relative_path: str) -> bool:
    key = s3_key(relative_path)
    try:
        s3_client().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3_client().exceptions.ClientError:
        return False
    except Exception:
        return False


def list_keys(prefix: str) -> list[str]:
    """List all S3 keys under prefix (relative, without S3_PREFIX)."""
    full_prefix = s3_key(prefix)
    paginator = s3_client().get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=full_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

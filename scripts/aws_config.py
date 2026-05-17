"""
AWS client factory.

Reads standard AWS credential env vars:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

Also used:
  S3_BUCKET, S3_PREFIX   — data storage
  BEDROCK_REGION         — Bedrock model region (may differ from S3 region)
  BEDROCK_MODEL_ID       — Claude model ID for QA generation
  BUDGET_LIMIT_USD       — monthly spend cap (crawl / data stage)
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION       = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET        = os.getenv("S3_BUCKET", "usa-immigration-2026")
S3_PREFIX        = os.getenv("S3_PREFIX", "v1")
BEDROCK_REGION   = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
BUDGET_LIMIT_USD = float(os.getenv("BUDGET_LIMIT_USD", "130.0"))


def make_boto_session(region: str | None = None) -> boto3.Session:
    """Return a boto3 Session using credentials from environment."""
    return boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region or AWS_REGION,
    )


def s3_client():
    """S3 client (us-east-1 default)."""
    return make_boto_session().client("s3")


def ce_client():
    """Cost Explorer — must be us-east-1."""
    return make_boto_session("us-east-1").client("ce")


def bedrock_client():
    """Bedrock runtime client."""
    return make_boto_session(BEDROCK_REGION).client("bedrock-runtime")


def s3_key(relative_path: str) -> str:
    return f"{S3_PREFIX}/{relative_path}"

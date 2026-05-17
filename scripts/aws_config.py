"""
AWS client factory.

ACTIVE_ACCOUNT=1  → Account 1 credentials (crawl + S3 + budget checks)
ACTIVE_ACCOUNT=2  → Account 2 credentials (Bedrock QA compute)

Bedrock uses the ABSK API key passed as a Bearer token, not IAM credentials.
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

ACTIVE_ACCOUNT = os.getenv("ACTIVE_ACCOUNT", "1")

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "usa-immigration-2026")
S3_PREFIX = os.getenv("S3_PREFIX", "v1")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-6",
)
BUDGET_LIMIT_USD = float(os.getenv("BUDGET_LIMIT_USD", "130.0"))
BEDROCK_BUDGET_USD = float(os.getenv("BEDROCK_BUDGET_USD", "200.0"))


def _account_creds(account: str) -> dict:
    return {
        "aws_access_key_id": os.getenv(f"ACCOUNT{account}_AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv(f"ACCOUNT{account}_AWS_SECRET_ACCESS_KEY"),
        "region_name": AWS_REGION,
    }


def _active_creds() -> dict:
    return _account_creds(ACTIVE_ACCOUNT)


def s3_client():
    """S3 always uses Account 1 (storage account)."""
    return boto3.client("s3", **_account_creds("1"))


def ce_client():
    """Cost Explorer always uses Account 1, must be us-east-1."""
    creds = _account_creds("1")
    creds["region_name"] = "us-east-1"
    return boto3.client("ce", **creds)


def bedrock_client():
    """Bedrock runtime client — uses Account 1 IAM credentials."""
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("ACCOUNT1_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("ACCOUNT1_AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("BEDROCK_REGION", "us-east-1"),
    )


def s3_key(relative_path: str) -> str:
    return f"{S3_PREFIX}/{relative_path}"

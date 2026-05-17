"""
One-time AWS setup:
  - Create S3 bucket usa-immigration-2026
  - Enable versioning
  - Create AWS Budget alert at $120 (warn) and $130 (hard stop via SNS)
  - Print confirmation

Run once: python -m scripts.setup_aws
"""

import json
import sys

import boto3
import structlog
from dotenv import load_dotenv

from scripts.aws_config import AWS_REGION, BUDGET_LIMIT_USD, S3_BUCKET, s3_client

load_dotenv()
log = structlog.get_logger()

WARN_THRESHOLD = BUDGET_LIMIT_USD * 0.92  # ~$120 on a $130 budget


def create_bucket():
    s3 = s3_client()
    try:
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )
        log.info("bucket_created", bucket=S3_BUCKET)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        log.info("bucket_exists", bucket=S3_BUCKET)

    s3.put_bucket_versioning(
        Bucket=S3_BUCKET,
        VersioningConfiguration={"Status": "Enabled"},
    )
    log.info("versioning_enabled", bucket=S3_BUCKET)

    # Block all public access
    s3.put_public_access_block(
        Bucket=S3_BUCKET,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    log.info("public_access_blocked", bucket=S3_BUCKET)


def create_budget(account_id: str, notification_email: str):
    budgets = boto3.client("budgets", region_name="us-east-1")
    budget_name = "usa-immigration-pipeline-limit"

    budget = {
        "BudgetName": budget_name,
        "BudgetLimit": {"Amount": str(BUDGET_LIMIT_USD), "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
    }

    notifications = [
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": (WARN_THRESHOLD / BUDGET_LIMIT_USD) * 100,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": notification_email}],
        },
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100.0,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": notification_email}],
        },
    ]

    try:
        budgets.create_budget(
            AccountId=account_id,
            Budget=budget,
            NotificationsWithSubscribers=notifications,
        )
        log.info("budget_created", name=budget_name, limit=BUDGET_LIMIT_USD)
    except budgets.exceptions.DuplicateRecordException:
        log.info("budget_exists", name=budget_name)


def main():
    sts = boto3.client("sts", region_name="us-east-1")
    identity = sts.get_caller_identity()
    account_id = identity["Account"]
    print(f"AWS Account: {account_id}")
    print(f"Region: {AWS_REGION}")
    print(f"Bucket: {S3_BUCKET}")

    notification_email = input("Enter email for budget alerts: ").strip()
    if not notification_email:
        print("No email provided, skipping budget alerts.")
        notification_email = None

    create_bucket()

    if notification_email:
        create_budget(account_id, notification_email)

    print("\nSetup complete.")
    print(f"  S3 bucket:    s3://{S3_BUCKET}")
    print(f"  Budget limit: ${BUDGET_LIMIT_USD:.2f}")


if __name__ == "__main__":
    main()

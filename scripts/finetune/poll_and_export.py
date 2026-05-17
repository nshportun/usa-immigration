"""
Poll SageMaker fine-tuning job until complete, then export model to HuggingFace.

Run: SAGEMAKER_JOB_NAME=<job_name> python scripts/finetune/poll_and_export.py

Logs status every 60s. On completion, downloads model artifacts from S3
and pushes to HuggingFace (default: nshportun/usa-immigration-llama-3.2-3b-v3).
"""

import boto3
import json
import os
import pathlib
import sys
import tarfile
import time
import urllib.parse

import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

BASE          = pathlib.Path(__file__).resolve().parents[2]
JOB_NAME      = os.getenv("SAGEMAKER_JOB_NAME", "")
REGION        = os.getenv("SAGEMAKER_REGION", "us-west-2")
FT_BUCKET     = os.getenv("SAGEMAKER_BUCKET", "usa-immigration-finetune-2026")
HF_TOKEN      = os.getenv("HF_TOKEN")
HF_USERNAME   = os.getenv("HF_USERNAME", "nshportun")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", f"{HF_USERNAME}/usa-immigration-llama-3.2-3b-v3")


def get_clients():
    kw = dict(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=REGION,
    )
    return boto3.client("sagemaker", **kw), boto3.client("s3", **kw)


def wait_for_job(sm, job_name: str) -> tuple[str, dict]:
    console_url = (
        f"https://{REGION}.console.aws.amazon.com/sagemaker/home"
        f"?region={REGION}#/jobs/{job_name}"
    )
    log.info("polling_job", job=job_name, console=console_url)
    last_secondary = ""
    while True:
        resp      = sm.describe_training_job(TrainingJobName=job_name)
        status    = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        if secondary != last_secondary:
            elapsed = ""
            if resp.get("TrainingStartTime") and resp.get("LastModifiedTime"):
                elapsed_s = (resp["LastModifiedTime"] - resp["TrainingStartTime"]).total_seconds()
                elapsed   = f" ({int(elapsed_s // 60)}m elapsed)"
            log.info("job_status", status=status, secondary=secondary, elapsed=elapsed)
            last_secondary = secondary
        if status in ("Completed", "Failed", "Stopped"):
            return status, resp
        time.sleep(60)


def export_and_push(sm, s3_client, job_name: str):
    resp         = sm.describe_training_job(TrainingJobName=job_name)
    model_s3_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
    log.info("model_artifacts_at", uri=model_s3_uri)

    parsed = urllib.parse.urlparse(model_s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    out_dir     = BASE / "data_local" / "model_artifacts_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    # List objects under the prefix — JumpStart may output loose files or a tar.gz
    paginator = s3_client.get_paginator("list_objects_v2")
    all_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            all_keys.append((obj["Key"], obj["Size"]))

    if not all_keys:
        raise RuntimeError(f"No files found at s3://{bucket}/{prefix}")

    if len(all_keys) == 1 and all_keys[0][0].endswith(".tar.gz"):
        key = all_keys[0][0]
        tar_path = out_dir / "model.tar.gz"
        log.info("downloading_model_tar", bucket=bucket, key=key,
                 size_gb=round(all_keys[0][1] / 1e9, 2))
        s3_client.download_file(bucket, key, str(tar_path))
        log.info("extracting")
        with tarfile.open(tar_path) as tf:
            tf.extractall(str(extract_dir))
        log.info("extraction_complete", files=len(list(extract_dir.rglob("*"))))
    else:
        # Loose files — download each one preserving relative path
        total_gb = round(sum(s for _, s in all_keys) / 1e9, 2)
        log.info("downloading_loose_files",
                 n=len(all_keys), total_gb=total_gb, bucket=bucket, prefix=prefix)
        for i, (key, size) in enumerate(all_keys):
            rel = key[len(prefix):].lstrip("/")
            local_path = extract_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading_file", n=f"{i+1}/{len(all_keys)}",
                     file=rel, size_mb=round(size / 1e6, 1))
            s3_client.download_file(bucket, key, str(local_path))
        log.info("all_files_downloaded", path=str(extract_dir))

    # Push to HuggingFace
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    try:
        api.create_repo(HF_MODEL_REPO, repo_type="model", private=False)
        log.info("hf_repo_created", repo=HF_MODEL_REPO)
    except Exception:
        log.info("hf_repo_exists", repo=HF_MODEL_REPO)

    log.info("uploading_model_folder", path=str(extract_dir))
    api.upload_folder(
        folder_path=str(extract_dir),
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"Add fine-tuned weights — SageMaker job {job_name}",
    )

    url = f"https://huggingface.co/{HF_MODEL_REPO}"
    log.info("model_published", url=url)
    sys.stdout.buffer.write(f"\nModel published: {url}\n".encode("utf-8"))


def main():
    os.chdir(BASE)

    if not JOB_NAME:
        print("ERROR: set SAGEMAKER_JOB_NAME env var to the training job name")
        print("Example: SAGEMAKER_JOB_NAME=immigration-llama32-3b-1234567890 python scripts/finetune/poll_and_export.py")
        sys.exit(1)

    sm, s3 = get_clients()
    status, resp = wait_for_job(sm, JOB_NAME)
    log.info("job_finished", status=status, job=JOB_NAME)

    if status == "Completed":
        export_and_push(sm, s3, JOB_NAME)
    else:
        reason = resp.get("FailureReason", "unknown")
        log.error("job_failed", reason=reason)
        sys.exit(1)


if __name__ == "__main__":
    main()

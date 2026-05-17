"""
Poll SageMaker fine-tuning job until complete, then export model to HuggingFace.
Run: python scripts/finetune/poll_and_export.py

Will log status every 60s. On completion, downloads model artifacts from S3
and pushes to nshportun/usa-immigration-llama-3.2-3b on HuggingFace.
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

log = structlog.get_logger()

BASE        = pathlib.Path(__file__).resolve().parents[2]
JOB_NAME    = os.getenv("SAGEMAKER_JOB_NAME", "immigration-llama32-3b-1778989041")
REGION      = os.getenv("SAGEMAKER_REGION", "us-east-1")
FT_BUCKET   = os.getenv("SAGEMAKER_BUCKET", "sagemaker-immigration-finetune-2026")
HF_TOKEN    = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "nshportun")
HF_MODEL_REPO = f"{HF_USERNAME}/usa-immigration-llama-3.2-3b"


def get_clients():
    from dotenv import load_dotenv
    load_dotenv()
    kw = dict(
        aws_access_key_id=os.getenv("ACCOUNT2_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("ACCOUNT2_AWS_SECRET_ACCESS_KEY"),
        region_name=REGION,
    )
    return boto3.client("sagemaker", **kw), boto3.client("s3", **kw)


def wait_for_job(sm) -> tuple[str, dict]:
    log.info("polling_job", job=JOB_NAME,
             console=f"https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/{JOB_NAME}")
    last_secondary = ""
    while True:
        resp = sm.describe_training_job(TrainingJobName=JOB_NAME)
        status    = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        if secondary != last_secondary:
            log.info("job_status", status=status, secondary=secondary)
            last_secondary = secondary
        if status in ("Completed", "Failed", "Stopped"):
            return status, resp
        time.sleep(60)


def export_and_push(sm, s3_client, resp: dict):
    model_s3_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
    log.info("model_artifacts_at", uri=model_s3_uri)

    parsed = urllib.parse.urlparse(model_s3_uri)
    bucket = parsed.netloc
    key    = parsed.path.lstrip("/")

    out_dir  = BASE / "data_local" / "model_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "model.tar.gz"

    log.info("downloading_model_tar", bucket=bucket, key=key)
    s3_client.download_file(bucket, key, str(tar_path))
    sz = tar_path.stat().st_size
    log.info("download_complete", size_gb=round(sz / 1e9, 2))

    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    log.info("extracting")
    with tarfile.open(tar_path) as tar:
        tar.extractall(str(extract_dir))

    # List extracted files
    extracted_files = list(extract_dir.rglob("*"))
    log.info("extraction_complete", file_count=len(extracted_files))

    # Push to HuggingFace
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    try:
        api.create_repo(HF_MODEL_REPO, repo_type="model", private=False)
        log.info("hf_repo_created", repo=HF_MODEL_REPO)
    except Exception:
        log.info("hf_repo_exists", repo=HF_MODEL_REPO)

    model_card = f"""\
---
language:
- en
license: llama3.2
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- legal
- immigration
- fine-tuned
- llama
- united-states
- lora
datasets:
- {HF_USERNAME}/usa-immigration-law-qa
pipeline_tag: text-generation
---

# USA Immigration Law — Llama 3.2 3B

Fine-tuned from [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
on the [{HF_USERNAME}/usa-immigration-law-qa](https://huggingface.co/datasets/{HF_USERNAME}/usa-immigration-law-qa)
dataset — **17,058 source-grounded Q&A pairs** covering all major U.S. immigration subdomains.

## Training Details

| Setting | Value |
|---------|-------|
| Base model | Llama 3.2 3B Instruct |
| Method | LoRA (r=8, alpha=32, merged into base weights) |
| Training pairs | 16,065 |
| Eval pairs | 993 (stratified across 13 subdomains) |
| Epochs | 1 |
| Batch size | 4 per device |
| Learning rate | 1e-4 |
| Max input length | 1,024 tokens |
| Infrastructure | AWS SageMaker ml.g5.2xlarge (24GB VRAM) |

## Immigration Subdomains Covered

| Subdomain | QA Pairs |
|-----------|----------|
| Family-based immigration | ~3,987 |
| Naturalization | ~2,670 |
| Asylum | ~2,094 |
| Adjustment of status | ~1,727 |
| Removal | ~1,277 |
| Humanitarian | ~894 |
| Employment authorization | ~832 |
| Admissibility | ~553 |
| Nonimmigrant visas | ~548 |
| Travel documents | ~109 |
| Employment-based (EB) | ~74 |
| Appeals | ~66 |
| Statistics | ~141 |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "{HF_MODEL_REPO}"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {{"role": "system", "content": "You are an expert on U.S. immigration law and policy. Answer accurately based on official sources."}},
    {{"role": "user", "content": "Who is eligible to apply for adjustment of status?"}}
]
result = pipe(messages, max_new_tokens=512)
print(result[0]["generated_text"][-1]["content"])
```

## Data Sources

- **USCIS Policy Manual** — primary_official
- **USCIS Forms & Instructions** (I-130, I-485, I-765, N-400, I-589...) — primary_official
- **8 CFR / INA statute text** — primary_official
- **BIA Precedent Decisions** — primary_official
- **harshitha008/US-immigration-laws** (Apache 2.0) — secondary_reputable
- **Law StackExchange immigration posts** — community

## Intended Use

- RAG-based immigration legal assistants
- Domain-specific LLM benchmarking
- Immigration law Q&A research

## Disclaimer

This model is for **research and educational purposes only**.
It does not constitute legal advice. Immigration law is complex and
changes frequently — always consult a licensed immigration attorney.
"""

    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message="Add model card",
    )

    log.info("uploading_model_folder", path=str(extract_dir))
    api.upload_folder(
        folder_path=str(extract_dir),
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"Add fine-tuned weights — SageMaker job {JOB_NAME}",
    )

    url = f"https://huggingface.co/{HF_MODEL_REPO}"
    log.info("model_published", url=url)
    sys.stdout.buffer.write(f"\nModel published: {url}\n".encode("utf-8"))


def main():
    os.chdir(BASE)
    sm, s3 = get_clients()

    status, resp = wait_for_job(sm)
    log.info("job_finished", status=status, job=JOB_NAME)

    if status == "Completed":
        export_and_push(sm, s3, resp)
    else:
        reason = resp.get("FailureReason", "unknown")
        log.error("job_failed", reason=reason)
        sys.exit(1)


if __name__ == "__main__":
    main()

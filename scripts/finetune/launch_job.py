"""
Launch SageMaker JumpStart fine-tune for Llama 3.2 3B Instruct.
Uses boto3 directly — no sagemaker SDK required.

All artifact URIs come from describe_hub_content (already verified).
"""

import boto3
import json
import os
import pathlib
import time
import sys

import structlog
log = structlog.get_logger()

BASE = pathlib.Path(__file__).resolve().parents[2]

REGION        = os.getenv("SAGEMAKER_REGION", "us-west-2")
FT_BUCKET     = os.getenv("SAGEMAKER_BUCKET", "usa-immigration-finetune-2026")
ROLE_ARN      = os.getenv("SAGEMAKER_ROLE_ARN", "")

HF_TOKEN      = os.getenv("HF_TOKEN")
HF_USERNAME   = os.getenv("HF_USERNAME", "nshportun")
HF_MODEL_REPO = f"{HF_USERNAME}/usa-immigration-llama-3.2-3b"

# JumpStart artifact URIs from describe_hub_content (Llama 3.2 3B Instruct v2.7.0)
TRAINING_IMAGE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
)
TRAINING_ARTIFACT_URI = (
    "s3://jumpstart-private-cache-prod-us-west-2/meta-training/"
    "train-meta-textgeneration-llama-3-2-3b-instruct.tar.gz"
)
TRAINING_SCRIPT_URI = (
    "s3://jumpstart-cache-prod-us-west-2/source-directory-tarballs/training/"
    "meta-textgeneration/prepack/inference-meta-textgeneration/v1.2.0/sourcedir.tar.gz"
)
# Gated model artifacts: need accept_eula env var
GATED_MODEL_ENV_URI = (
    "s3://jumpstart-private-cache-prod-us-west-2/meta-training/g5/v1.0.0/"
    "train-meta-textgeneration-llama-3-2-3b-instruct.tar.gz"
)

JOB_NAME      = f"immigration-llama32-3b-{int(time.time())}"
INSTANCE_TYPE = "ml.g5.2xlarge"


def get_clients():
    from dotenv import load_dotenv
    load_dotenv()
    kw = dict(
        aws_access_key_id=os.getenv("ACCOUNT2_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("ACCOUNT2_AWS_SECRET_ACCESS_KEY"),
        region_name=REGION,
    )
    sm  = boto3.client("sagemaker", **kw)
    s3  = boto3.client("s3", **kw)
    return sm, s3, kw


def launch_training_job(sm) -> str:
    log.info("launching_training_job", job=JOB_NAME, instance=INSTANCE_TYPE)

    if not ROLE_ARN:
        raise ValueError("SAGEMAKER_ROLE_ARN env var is required")

    resp = sm.create_training_job(
        TrainingJobName=JOB_NAME,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage": TRAINING_IMAGE,
            "TrainingInputMode": "File",
            "EnableSageMakerMetricsTimeSeries": True,
            "MetricDefinitions": [
                {"Name": "huggingface-textgeneration:eval-loss",
                 "Regex": r"eval_epoch_loss=tensor\(([0-9\.]+)"},
                {"Name": "huggingface-textgeneration:train-loss",
                 "Regex": r"train_epoch_loss=([0-9\.]+)"},
            ],
        },
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{FT_BUCKET}/finetune/train/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/jsonlines",
                "InputMode": "File",
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{FT_BUCKET}/finetune/eval/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/jsonlines",
                "InputMode": "File",
            },
            {
                "ChannelName": "model",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": TRAINING_ARTIFACT_URI.rsplit("/", 1)[0] + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-sagemaker-model",
                "InputMode": "File",
            },
            {
                "ChannelName": "code",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": TRAINING_SCRIPT_URI.rsplit("/", 1)[0] + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-code",
                "InputMode": "File",
            },
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{FT_BUCKET}/output/",
        },
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 256,
        },
        HyperParameters={
            # LoRA fine-tuning settings
            "instruction_tuned": "True",
            "chat_dataset": "True",
            "chat_template": "Llama3.1",
            "epoch": "1",
            "learning_rate": "0.0001",
            "lora_r": "8",
            "lora_alpha": "32",
            "lora_dropout": "0.05",
            "target_modules": "q_proj,v_proj",
            "per_device_train_batch_size": "4",
            "per_device_eval_batch_size": "2",
            "max_input_length": "1024",
            "enable_fsdp": "False",
            "int8_quantization": "False",
            "seed": "42",
            # Required by JumpStart training script
            "sagemaker_program": "transfer_learning.py",
            "sagemaker_submit_directory": "/opt/ml/input/data/code/sourcedir.tar.gz",
            # EULA acceptance for gated model
            "accept_eula": "true",
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 86400,  # 24h max
        },
        EnableNetworkIsolation=True,
        EnableManagedSpotTraining=False,
    )

    log.info("training_job_created", job=JOB_NAME, arn=resp["TrainingJobArn"])
    return JOB_NAME


def wait_for_job(sm, job_name: str) -> tuple[str, dict]:
    log.info("waiting_for_job", job=job_name)
    while True:
        resp = sm.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        elapsed = ""
        if resp.get("TrainingStartTime") and resp.get("LastModifiedTime"):
            import datetime
            elapsed_s = (resp["LastModifiedTime"] - resp["TrainingStartTime"]).total_seconds()
            elapsed = f" ({int(elapsed_s//60)}m elapsed)"
        log.info("job_status", status=status, secondary=secondary, elapsed=elapsed)
        if status in ("Completed", "Failed", "Stopped"):
            return status, resp
        time.sleep(60)


def export_and_push(sm, s3_client, job_name: str):
    resp = sm.describe_training_job(TrainingJobName=job_name)
    model_s3_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
    log.info("model_artifacts_at", uri=model_s3_uri)

    # Download model.tar.gz
    import urllib.parse, tarfile
    parsed = urllib.parse.urlparse(model_s3_uri)
    bucket = parsed.netloc
    key    = parsed.path.lstrip("/")

    out_dir  = BASE / "data_local" / "model_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "model.tar.gz"

    log.info("downloading_model", key=key, size_hint="~3-6GB")
    s3_client.download_file(bucket, key, str(tar_path))
    log.info("download_complete", path=str(tar_path))

    extract_dir = out_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    log.info("extracting_model")
    with tarfile.open(tar_path) as tar:
        tar.extractall(str(extract_dir))
    log.info("extraction_complete", path=str(extract_dir))

    # Push to HuggingFace
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    try:
        api.create_repo(HF_MODEL_REPO, repo_type="model", private=False)
        log.info("hf_repo_created", repo=HF_MODEL_REPO)
    except Exception:
        log.info("hf_repo_exists_or_created", repo=HF_MODEL_REPO)

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

## Training

| Setting | Value |
|---------|-------|
| Base model | Llama 3.2 3B Instruct |
| Method | LoRA (r=8, alpha=32, merged) |
| Training pairs | 16,065 |
| Eval pairs | 993 (stratified) |
| Epochs | 1 |
| Max input length | 1,024 tokens |
| Infrastructure | AWS SageMaker ml.g5.2xlarge |

## Subdomains

Family-based immigration · Adjustment of status · Employment authorization ·
Naturalization · Travel documents · Asylum · Removal · Admissibility ·
Employment-based visas · Nonimmigrant visas · Humanitarian relief · Appeals · Statistics

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
    {{"role": "system", "content": "You are an expert on U.S. immigration law and policy."}},
    {{"role": "user", "content": "Who is eligible to apply for adjustment of status?"}}
]
output = pipe(messages, max_new_tokens=512)
print(output[0]["generated_text"][-1]["content"])
```

## Data Sources

- USCIS Policy Manual (primary_official)
- USCIS Forms & Instructions (primary_official)
- 8 CFR / INA (primary_official)
- BIA Precedent Decisions (primary_official)
- harshitha008/US-immigration-laws (secondary_reputable, Apache 2.0)
- Law StackExchange immigration posts (community)

## Disclaimer

For research and educational purposes only. Not legal advice.
Always consult a licensed immigration attorney for your specific situation.
"""

    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message="Add model card",
    )

    log.info("uploading_model_weights", path=str(extract_dir))
    api.upload_folder(
        folder_path=str(extract_dir),
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"Add fine-tuned weights (SageMaker job: {job_name})",
    )

    url = f"https://huggingface.co/{HF_MODEL_REPO}"
    log.info("model_published", url=url)
    sys.stdout.buffer.write(f"Model published: {url}\n".encode("utf-8"))


def main():
    os.chdir(BASE)
    from dotenv import load_dotenv
    load_dotenv()

    sm, s3, _ = get_clients()

    job_name = launch_training_job(sm)
    log.info("job_launched", job=job_name,
             console=f"https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs/{job_name}")

    status, resp = wait_for_job(sm, job_name)

    if status == "Completed":
        log.info("training_complete", job=job_name)
        export_and_push(sm, s3, job_name)
    else:
        reason = resp.get("FailureReason", "unknown")
        log.error("training_failed", status=status, reason=reason)
        sys.exit(1)


if __name__ == "__main__":
    main()

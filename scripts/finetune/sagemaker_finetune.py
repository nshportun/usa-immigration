"""
Fine-tune Llama 3.2 3B on the USA Immigration Law Q&A dataset via SageMaker JumpStart.

Steps:
1. Upload training/eval JSONL to S3
2. Launch SageMaker JumpStart fine-tuning job (Llama-3-2-3B-Instruct, LoRA)
3. Wait for job completion
4. Export model artifacts from S3
5. Push to HuggingFace as nshportun/usa-immigration-llama-3.2-3b-v3

Cost estimate: ~$10-20 on ml.g5.2xlarge (~2-3 hours for 16K examples, 2 epochs)

Run this script AFTER the dataset is published:
    python scripts/finetune/sagemaker_finetune.py
"""

import boto3
import json
import os
import pathlib
import subprocess
import sys
import time

import structlog

log = structlog.get_logger()

BASE = pathlib.Path(__file__).resolve().parents[2]

# ── AWS credentials ────────────────────────────────────────────────────────────
AWS_KEY    = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION     = os.getenv("SAGEMAKER_REGION", "us-west-2")   # JumpStart Llama available here

# ── S3 ─────────────────────────────────────────────────────────────────────────
S3_BUCKET  = os.getenv("SAGEMAKER_BUCKET", os.getenv("S3_BUCKET", "usa-immigration-finetune-2026"))
S3_FT_PREFIX = "v1/finetune"

# ── SageMaker settings ─────────────────────────────────────────────────────────
# JumpStart model ID for Llama 3.2 3B Instruct
JS_MODEL_ID      = "meta-textgeneration-llama-3-2-3b-instruct"
JS_MODEL_VERSION = "*"   # latest

INSTANCE_TYPE    = "ml.g5.2xlarge"   # 24GB VRAM, ~$1.50/hr
EPOCHS           = 2                  # 2 epochs ~ 2-3h on 16K examples
BATCH_SIZE       = 2
MAX_INPUT_LEN    = 1024
LORA_R           = 32                 # higher capacity for domain adaptation
LEARNING_RATE    = 5e-5               # conservative: prevents catastrophic forgetting

JOB_NAME = f"usa-immigration-llama32-3b-{int(time.time())}"

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN         = os.getenv("HF_TOKEN")
HF_USERNAME      = os.getenv("HF_USERNAME", "nshportun")
HF_MODEL_REPO    = os.getenv("HF_MODEL_REPO", f"{HF_USERNAME}/usa-immigration-llama-3.2-3b-v3")


def get_sagemaker_role(iam_client) -> str:
    """Get or create SageMaker execution role."""
    role_name = "SageMakerExecutionRole-ImmigrationFT"
    try:
        resp = iam_client.get_role(RoleName=role_name)
        arn = resp["Role"]["Arn"]
        log.info("using_existing_role", arn=arn)
        return arn
    except iam_client.exceptions.NoSuchEntityException:
        pass

    log.info("creating_sagemaker_role", role_name=role_name)
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    resp = iam_client.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="SageMaker execution role for immigration fine-tuning",
    )
    arn = resp["Role"]["Arn"]
    # Attach managed policies
    for policy in [
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    ]:
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy)
    log.info("role_created", arn=arn)
    time.sleep(10)   # propagation delay
    return arn


def upload_training_data(s3_client) -> tuple[str, str]:
    """Upload train/eval JSONL to S3, return (train_s3_uri, eval_s3_uri)."""
    finetune_dir = BASE / "data_local" / "finetune"

    for fname, s3key in [
        ("train_chat.jsonl", f"{S3_FT_PREFIX}/train/train.jsonl"),
        ("eval_chat.jsonl",  f"{S3_FT_PREFIX}/eval/eval.jsonl"),
    ]:
        local = finetune_dir / fname
        log.info("uploading_training_data", local=str(local), key=s3key)
        s3_client.upload_file(str(local), S3_BUCKET, s3key)

    train_uri = f"s3://{S3_BUCKET}/{S3_FT_PREFIX}/train/"
    eval_uri  = f"s3://{S3_BUCKET}/{S3_FT_PREFIX}/eval/"
    log.info("training_data_uploaded", train=train_uri, eval=eval_uri)
    return train_uri, eval_uri


def launch_finetune_job(sm_client, role_arn: str, train_uri: str, eval_uri: str) -> str:
    """Launch SageMaker JumpStart fine-tune job, return job name."""
    from sagemaker.jumpstart.estimator import JumpStartEstimator
    import sagemaker

    sess = sagemaker.Session(
        boto_session=boto3.Session(
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            region_name=REGION,
        )
    )

    log.info("launching_jumpstart_finetune", model=JS_MODEL_ID, instance=INSTANCE_TYPE, job=JOB_NAME)

    estimator = JumpStartEstimator(
        model_id=JS_MODEL_ID,
        model_version=JS_MODEL_VERSION,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        role=role_arn,
        sagemaker_session=sess,
        base_job_name=JOB_NAME,
    )

    estimator.set_hyperparameters(
        chat_dataset="True",                              # use chat template (not instruction_tuned — mutually exclusive)
        chat_template="Llama3.1",
        epoch=str(EPOCHS),
        per_device_train_batch_size=str(BATCH_SIZE),
        per_device_eval_batch_size=str(BATCH_SIZE),
        max_input_length=str(MAX_INPUT_LEN),
        lora_r=str(LORA_R),
        lora_alpha=str(LORA_R * 2),                      # alpha = 2×r convention
        lora_dropout="0.05",
        target_modules="q_proj,v_proj,k_proj,o_proj",    # all attention projections
        learning_rate=str(LEARNING_RATE),
        merge_weights="True",                             # merge LoRA into base weights for export
        seed="42",
    )

    estimator.fit(
        {
            "training": train_uri,
            "validation": eval_uri,
        },
        wait=False,
        job_name=JOB_NAME,
    )

    log.info("finetune_job_launched", job_name=JOB_NAME)
    return JOB_NAME


def wait_for_job(sm_client, job_name: str):
    """Poll until job completes or fails."""
    log.info("waiting_for_job", job_name=job_name)
    while True:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        log.info("job_status", status=status, secondary=secondary)
        if status in ("Completed", "Failed", "Stopped"):
            return status, resp
        time.sleep(60)


def export_and_push_to_hf(sm_client, job_name: str):
    """Download model artifacts from S3 and push to HuggingFace."""
    resp = sm_client.describe_training_job(TrainingJobName=job_name)
    model_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
    log.info("model_artifacts_uri", uri=model_uri)

    # Download
    output_dir = BASE / "data_local" / "model_artifacts_v3"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tar = output_dir / "model.tar.gz"

    import urllib.parse
    parsed = urllib.parse.urlparse(model_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = boto3.client("s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=REGION,
    )
    log.info("downloading_model_artifacts", key=key)
    s3.download_file(bucket, key, str(model_tar))

    # Extract
    import tarfile
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(model_tar) as tar:
        tar.extractall(extract_dir)
    log.info("artifacts_extracted", path=str(extract_dir))

    # Push to HF
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    # Create repo if not exists
    try:
        api.create_repo(HF_MODEL_REPO, repo_type="model", private=False)
        log.info("hf_repo_created", repo=HF_MODEL_REPO)
    except Exception:
        log.info("hf_repo_exists", repo=HF_MODEL_REPO)

    # Write model card
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
---

# USA Immigration Law Llama 3.2 3B

A Llama 3.2 3B Instruct model fine-tuned on the
[USA Immigration Law Q&A dataset](https://huggingface.co/datasets/{HF_USERNAME}/usa-immigration-law-qa)
with 17,058 source-grounded question-answer pairs covering all major U.S. immigration subdomains.

## Training Details

- **Base model**: meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning method**: LoRA (r=32, all attention projections, merged into base weights)
- **Training data**: 16,065 Q&A pairs (official USCIS, HF dataset, community)
- **Eval data**: 993 stratified Q&A pairs
- **Epochs**: 2  |  **Learning rate**: 5e-5
- **Instance**: AWS SageMaker ml.g5.2xlarge

## Subdomains Covered

Family-based immigration, Adjustment of status, Employment authorization,
Naturalization, Travel documents, Asylum, Removal, Admissibility,
Employment-based visas, Nonimmigrant visas, Humanitarian relief,
Appeals, and Immigration statistics.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "{HF_MODEL_REPO}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

messages = [
    {{"role": "system", "content": "You are an expert on U.S. immigration law."}},
    {{"role": "user", "content": "Who is eligible for adjustment of status?"}}
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Disclaimer

This model is for research and educational purposes only. It does not constitute
legal advice. Always consult a licensed immigration attorney.
"""

    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message="Add model card",
    )

    # Upload all model files
    api.upload_folder(
        folder_path=str(extract_dir),
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"Add fine-tuned model weights (job: {job_name})",
    )

    log.info("model_published", url=f"https://huggingface.co/{HF_MODEL_REPO}")
    print(f"Model published: https://huggingface.co/{HF_MODEL_REPO}")


def main():
    os.chdir(BASE)

    from dotenv import load_dotenv
    load_dotenv()

    global AWS_KEY, AWS_SECRET
    AWS_KEY    = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

    s3_client = boto3.client("s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    iam_client = boto3.client("iam",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=REGION,
    )
    sm_client = boto3.client("sagemaker",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=REGION,
    )

    role_arn = get_sagemaker_role(iam_client)
    train_uri, eval_uri = upload_training_data(s3_client)
    job_name = launch_finetune_job(sm_client, role_arn, train_uri, eval_uri)

    status, resp = wait_for_job(sm_client, job_name)
    log.info("job_complete", status=status, job=job_name)

    if status == "Completed":
        export_and_push_to_hf(sm_client, job_name)
    else:
        log.error("job_failed", status=status,
                  failure=resp.get("FailureReason", "unknown"))
        sys.exit(1)


if __name__ == "__main__":
    main()

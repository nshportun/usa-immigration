"""
Launch SageMaker JumpStart fine-tune for Llama 3.2 3B Instruct.

Uses JumpStartEstimator (handles EULA acceptance automatically).
Tries ml.g5.2xlarge first; falls back to ml.g4dn.2xlarge if quota is exceeded.

LoRA config: r=32, alpha=64, all attention projections, 2 epochs, lr=5e-5.

Run: python scripts/finetune/launch_job.py

After the job completes, export with:
  SAGEMAKER_JOB_NAME=<job_name> python scripts/finetune/poll_and_export.py
"""

import boto3
import os
import pathlib
import time
import sys

import sagemaker
from sagemaker.jumpstart.estimator import JumpStartEstimator
import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

BASE = pathlib.Path(__file__).resolve().parents[2]

REGION    = os.getenv("SAGEMAKER_REGION", "us-west-2")
FT_BUCKET = os.getenv("SAGEMAKER_BUCKET", "usa-immigration-finetune-2026")
ROLE_ARN  = os.getenv("SAGEMAKER_ROLE_ARN", "")

HF_TOKEN      = os.getenv("HF_TOKEN")
HF_USERNAME   = os.getenv("HF_USERNAME", "nshportun")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", f"{HF_USERNAME}/usa-immigration-llama-3.2-3b-v3")

JS_MODEL_ID = "meta-textgeneration-llama-3-2-3b-instruct"

# Instance preference: g5.2xlarge (A10G 24GB) first, g4dn.2xlarge (T4 16GB) fallback
INSTANCE_PREFERENCE = [
    "ml.g5.2xlarge",
    "ml.g4dn.2xlarge",
]


def make_session() -> sagemaker.Session:
    return sagemaker.Session(
        boto_session=boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=REGION,
        )
    )


def launch(sess: sagemaker.Session) -> str:
    if not ROLE_ARN:
        raise ValueError("SAGEMAKER_ROLE_ARN env var is required")

    job_name = f"immigration-llama32-3b-{int(time.time())}"

    for instance_type in INSTANCE_PREFERENCE:
        log.info("launching",
                 job=job_name, instance=instance_type,
                 lora_r=32, lora_alpha=64, epochs=2, lr="5e-5",
                 target_modules="q_proj,v_proj,k_proj,o_proj")
        try:
            estimator = JumpStartEstimator(
                model_id=JS_MODEL_ID,
                model_version="*",
                instance_type=instance_type,
                instance_count=1,
                role=ROLE_ARN,
                sagemaker_session=sess,
                environment={"accept_eula": "true"},
            )
            estimator.set_hyperparameters(
                chat_dataset="True",
                chat_template="Llama3.1",
                # ── LoRA ──────────────────────────────────────────────────────
                lora_r="32",
                lora_alpha="64",
                lora_dropout="0.05",
                target_modules="q_proj,v_proj,k_proj,o_proj",
                # ── training ──────────────────────────────────────────────────
                epoch="2",
                learning_rate="0.00005",
                per_device_train_batch_size="2",
                per_device_eval_batch_size="2",
                max_input_length="1024",
                # ── output ────────────────────────────────────────────────────
                merge_weights="True",
                seed="42",
            )
            estimator.fit(
                {
                    "training":   f"s3://{FT_BUCKET}/finetune/train/",
                    "validation": f"s3://{FT_BUCKET}/finetune/eval/",
                },
                wait=False,
                job_name=job_name,
            )
            launched_name = estimator.latest_training_job.name
            log.info("job_launched", job=launched_name, instance=instance_type)
            return launched_name

        except Exception as e:
            if "ResourceLimitExceeded" in str(e):
                log.warning("quota_exceeded_trying_next",
                            instance=instance_type, error=str(e)[:120])
                continue
            raise

    log.error("all_instances_quota_exceeded", tried=INSTANCE_PREFERENCE)
    print("\nAll instance types are quota-limited. Request an increase at:")
    print("  https://us-west-2.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas")
    sys.exit(1)


def main():
    os.chdir(BASE)
    sess     = make_session()
    job_name = launch(sess)

    console = (
        f"https://{REGION}.console.aws.amazon.com/sagemaker/home"
        f"?region={REGION}#/jobs/{job_name}"
    )
    log.info("training_started", job=job_name, console=console)
    print(f"\nJob launched:  {job_name}")
    print(f"Monitor at:    {console}")
    print(f"\nAfter training completes (~2-3h), export with:")
    print(f"  SAGEMAKER_JOB_NAME={job_name} python scripts/finetune/poll_and_export.py")


if __name__ == "__main__":
    main()

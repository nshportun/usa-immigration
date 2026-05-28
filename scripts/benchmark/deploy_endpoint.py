"""
Deploy the fine-tuned model as a SageMaker real-time endpoint for benchmarking.
Deletes the endpoint when done to avoid ongoing charges.

Instance fallback: tries ml.g5.2xlarge first, then ml.g4dn.2xlarge if quota exceeded.

Container: HuggingFace PyTorch Inference DLC (transformers 4.48.0, PyTorch 2.3.0)
  - Plain transformers.pipeline() — no Flash Attention dependency, works on T4 (g4dn)
  - Supports Llama 3.2 rope_type (transformers >= 4.45 required)

Usage:
    python scripts/benchmark/deploy_endpoint.py deploy   # create endpoint
    python scripts/benchmark/deploy_endpoint.py delete   # tear down endpoint

Environment variables:
    HF_MODEL_ID             HuggingFace model repo to deploy (default: v2)
    ENDPOINT_NAME           Override default endpoint name
    SAGEMAKER_REGION        AWS region (default: us-west-2)
    SAGEMAKER_ROLE_ARN      SageMaker execution role ARN
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
"""

import boto3
import os
import sys
import time

import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

REGION        = os.getenv("SAGEMAKER_REGION", "us-west-2")
ROLE_ARN      = os.getenv("SAGEMAKER_ROLE_ARN", "")
HF_MODEL_ID   = os.getenv("HF_MODEL_ID", "")
HF_TOKEN      = os.getenv("HF_TOKEN", "")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "immigration-llama32-benchmark")

# Instance preference order — try g5 first, fall back to g4dn
INSTANCE_PREFERENCE = [
    "ml.g5.2xlarge",    # A10G 24GB — preferred
    "ml.g4dn.2xlarge",  # T4  16GB — fallback if g5 endpoint quota is 0
]

# HuggingFace PyTorch Inference DLC — uses plain transformers.pipeline()
#   - transformers 4.48.0 supports Llama 3.2 rope_type (>= 4.45 required)
#   - cu121 works on T4 GPU (sm75 / CUDA 7.5)
#   - No Flash Attention 2 dependency (FA2 requires Ampere / sm80+)
HF_IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/huggingface-pytorch-inference:2.3.0-transformers4.48.0-gpu-py311-cu121-ubuntu22.04"


def get_clients():
    kw = dict(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=REGION,
    )
    return boto3.client("sagemaker", **kw), boto3.client("sagemaker-runtime", **kw)


def _try_create_endpoint(sm, model_name: str, cfg_name: str, instance_type: str):
    """
    Create endpoint config + endpoint for a given instance type.
    Returns True on success, False if ResourceLimitExceeded (quota).
    Re-raises any other exception.
    """
    # Clean up any leftover config from a previous attempt
    try:
        sm.delete_endpoint_config(EndpointConfigName=cfg_name)
    except Exception:
        pass

    log.info("creating_endpoint_config", name=cfg_name, instance=instance_type)
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            "VariantName":          "AllTraffic",
            "ModelName":            model_name,
            "InstanceType":         instance_type,
            "InitialInstanceCount": 1,
        }],
    )

    log.info("creating_endpoint", name=ENDPOINT_NAME)
    try:
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=cfg_name,
        )
        return True
    except sm.exceptions.from_code("ResourceLimitExceeded") as e:
        log.warning("quota_exceeded", instance=instance_type, error=str(e)[:120])
        # Clean up the config we just created
        try:
            sm.delete_endpoint_config(EndpointConfigName=cfg_name)
        except Exception:
            pass
        return False
    except Exception as e:
        # Check by message string (botocore may not expose typed exception)
        if "ResourceLimitExceeded" in str(e):
            log.warning("quota_exceeded", instance=instance_type, error=str(e)[:120])
            try:
                sm.delete_endpoint_config(EndpointConfigName=cfg_name)
            except Exception:
                pass
            return False
        raise


def deploy():
    sm, _ = get_clients()
    ts = int(time.time())
    model_name = f"immigration-llama32-{ts}"

    log.info("creating_model", name=model_name, hf_model=HF_MODEL_ID, image="hf-pytorch-inference")
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            "Image": HF_IMAGE,
            "Environment": {
                # HuggingFace PyTorch Inference DLC environment variables
                "HF_MODEL_ID":    HF_MODEL_ID,
                "HF_TOKEN":       HF_TOKEN,
                "HF_TASK":        "text-generation",
                "SM_NUM_GPUS":    "1",
            },
        },
    )

    chosen_instance = None
    for instance_type in INSTANCE_PREFERENCE:
        cfg_name = f"immigration-llama32-cfg-{ts}-{instance_type.replace('.', '-')}"
        log.info("trying_instance", instance=instance_type)
        success = _try_create_endpoint(sm, model_name, cfg_name, instance_type)
        if success:
            chosen_instance = instance_type
            log.info("endpoint_creation_submitted", instance=chosen_instance)
            break
        else:
            log.warning("trying_next_instance", failed=instance_type)

    if chosen_instance is None:
        log.error("all_instance_types_quota_exceeded", tried=INSTANCE_PREFERENCE)
        print("\nAll instance types are quota-limited. Request a quota increase at:")
        print("  https://us-west-2.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas")
        sys.exit(1)

    # Wait for InService
    log.info("waiting_for_endpoint", endpoint=ENDPOINT_NAME, instance=chosen_instance)
    while True:
        resp   = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        log.info("endpoint_status", status=status)
        if status == "InService":
            log.info("endpoint_ready", name=ENDPOINT_NAME, instance=chosen_instance)
            print(f"\nEndpoint ready: {ENDPOINT_NAME}  ({chosen_instance})")
            print(f"Set env var:    BENCHMARK_SAGEMAKER_ENDPOINT={ENDPOINT_NAME}")
            return
        elif status in ("Failed", "RollingBack"):
            reason = resp.get("FailureReason", "unknown")
            log.error("endpoint_failed", reason=reason)
            sys.exit(1)
        time.sleep(30)


def delete():
    sm, _ = get_clients()
    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        log.info("endpoint_deleted", name=ENDPOINT_NAME)
    except Exception as e:
        log.warning("delete_error", error=str(e))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "deploy"
    if cmd == "deploy":
        deploy()
    elif cmd == "delete":
        delete()
    else:
        print("Usage: python deploy_endpoint.py [deploy|delete]")

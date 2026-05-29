# USA Immigration Law — Dataset & Fine-Tuned LLM

[![Dataset on HuggingFace](https://img.shields.io/badge/Dataset-nshportun%2Fusa--immigration--law--qa-blue?logo=huggingface)](https://huggingface.co/datasets/nshportun/usa-immigration-law-qa)
[![Model on HuggingFace](https://img.shields.io/badge/Model-nshportun%2Fusa--immigration--llama--3.2--3b--v3-orange?logo=huggingface)](https://huggingface.co/nshportun/usa-immigration-llama-3.2-3b-v3)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> A fully reproducible pipeline that builds a **17,058-question Q&A dataset** from official U.S.
> immigration sources and fine-tunes a **Llama 3.2 3B model that outperforms the Llama 3 8B
> zero-shot baseline** on immigration law questions (+27% mean score, 4× more fully-correct answers).

| Resource | Link |
|----------|------|
| Dataset | [`nshportun/usa-immigration-law-qa`](https://huggingface.co/datasets/nshportun/usa-immigration-law-qa) |
| Model | [`nshportun/usa-immigration-llama-3.2-3b-v3`](https://huggingface.co/nshportun/usa-immigration-llama-3.2-3b-v3) |
|arxiv.org|[`https://arxiv.org/submit/7615196/view`](https://arxiv.org/submit/7615196/view)|

---

## Benchmark Results

Evaluated on 101 held-out questions scored 0–3 by Claude Sonnet 4.6 as judge.

| Model | Mean Score (0–3) | % Fully Correct (3) | N |
|-------|-----------------|---------------------|---|
| Claude Sonnet 4.6 (zero-shot) | 1.515 | 24.8% | 101 |
| **Llama 3.2 3B fine-tuned (ours)** | **1.079** | **16.8%** | **101** |
| Llama 3 8B zero-shot | 0.851 | 4.0% | 101 |

**The 3B fine-tuned model beats the 8B zero-shot baseline by +27% on mean score and delivers 4× more fully-correct answers** — demonstrating that domain-specific fine-tuning at 3B scale surpasses a larger general model on this task.

---

## The Problem

U.S. immigration law is one of the most complex legal domains in America — yet chronically underserved by AI tooling. Existing LLMs hallucinate on specific procedural questions (which form to file, what evidence is required, processing times) because:

1. **Fragmented official sources** — USCIS Policy Manual, 8 CFR regulations, BIA precedent decisions, and form instructions are scattered across dozens of government websites
2. **No structured QA dataset** — the few existing datasets lack source grounding, answer types, or subdomain labels needed for fine-tuning and evaluation
3. **No domain-adapted small model** — practitioners who need a locally-runnable or cost-efficient model have nothing to fine-tune from

**This project solves all three.**

---

## Architecture

```
Official Sources         Raw Data             Structured Corpus         Q&A Dataset           Fine-Tuned Model
─────────────────       ──────────           ─────────────────         ───────────           ─────────────────
USCIS Policy Manual ──┐
USCIS Forms & Inst. ──┤
8 CFR / INA statute ──┤  Crawl &   ──►  Normalize &  ──►  Bedrock     ──►  Validate  ──►  SageMaker
BIA Precedent Decs  ──┤  Parse         Dedup & Chunk     Claude             & Split        JumpStart
DHS/CBP Statistics  ──┤                (tiktoken)        (QA gen)                         LoRA FT
harshitha008/HF     ──┤
Law StackExchange   ──┘
```

**What makes this unique:**
- Every Q&A pair carries `source_url`, `source_span`, `authority_level`, and `immigration_subtopic` — enabling citation-aware RAG
- Bedrock Claude generates structured QA in four modes: `faq`, `rule_derived`, `form`, `precedent`
- LoRA fine-tuning merged into base weights → standard `AutoModelForCausalLM` load, no adapter setup
- Single AWS account — Bedrock for QA generation, SageMaker JumpStart for fine-tuning

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              AWS (us-east-1 / us-west-2)                │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │  Crawl   │───►│  Parse   │───►│Normalize │───►│   S3 Bucket      │  │
│  │  Layer   │    │  Layer   │    │  Layer   │    │ usa-immigration   │  │
│  │          │    │          │    │          │    │ -2026            │  │
│  │ USCIS    │    │ HTML→txt │    │ tiktoken │    │                  │  │
│  │ EOIR     │    │ PDF→txt  │    │ dedup    │    │ /v1/raw/         │  │
│  │ 8CFR     │    │ JSON→txt │    │ chunk    │    │ /v1/corpus/      │  │
│  │ HF/SE    │    │          │    │ normalize│    │ /v1/qa_pairs/    │  │
│  └──────────┘    └──────────┘    └──────────┘    └────────┬─────────┘  │
│                                                           │             │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐   │
│  │   Amazon Bedrock     │    │           Amazon SageMaker           │   │
│  │  Claude Sonnet 4.6  │◄───┤         JumpStart Fine-tuning         │   │
│  │                     │    │                                      │   │
│  │  QA Generation:     │    │  • Instance: ml.g5.2xlarge (24GB)    │   │
│  │  - faq mode         │    │  • Base: Llama 3.2 3B Instruct       │   │
│  │  - rule mode        │    │  • LoRA r=32, α=64, all attn proj.   │   │
│  │  - form mode        │    │  • 2 epochs, lr=5e-5                 │   │
│  │  - precedent mode   │    │  • Merged weights → S3 → HF          │   │
│  │                     │    │                                      │   │
│  │  17,058 QA pairs    │    │  usa-immigration-finetune-2026        │   │
│  └─────────────────────┘    └──────────────────┬─────────────────┘   │
└─────────────────────────────────────────────────┼───────────────────────┘
                                                  │
                              ┌───────────────────▼───────────────────────┐
                              │              HuggingFace Hub               │
                              │                                            │
                              │  nshportun/usa-immigration-law-qa          │
                              │  ├── config: qa (train: 16065, eval: 993)  │
                              │  └── config: corpus (10056 docs)           │
                              │                                            │
                              │  nshportun/usa-immigration-llama-3.2-3b-v3 │
                              │  └── Llama 3.2 3B + LoRA merged (6.4GB)   │
                              └────────────────────────────────────────────┘
```

---

## Project Structure

```
usa-immigration/
├── main.py                          # Pipeline orchestrator (CLI)
├── pyproject.toml                   # Dependencies
├── .env.example                     # Environment variable template
├── .gitignore
│
├── scripts/
│   ├── crawl/                       # Source-specific crawlers
│   │   ├── crawl_uscis.py           # USCIS Policy Manual + Forms
│   │   ├── crawl_statutes.py        # 8 CFR / INA text
│   │   ├── crawl_eoir.py            # BIA Precedent Decisions
│   │   ├── crawl_dhs_stats.py       # DHS/CBP statistics
│   │   ├── crawl_secondary.py       # harshitha008/US-immigration-laws (HF)
│   │   └── crawl_community.py       # Law StackExchange Q&A
│   │
│   ├── parse/                       # Format-specific parsers
│   │   ├── parse_html.py
│   │   ├── parse_pdf.py
│   │   └── parse_json.py
│   │
│   ├── normalize/                   # Dedup, chunking, metadata normalization
│   │   ├── normalize_metadata.py
│   │   ├── chunker.py               # tiktoken-based chunker
│   │   └── dedup.py
│   │
│   ├── qa_generation/
│   │   ├── generate_qa.py           # Bedrock Claude QA generation
│   │   └── prompts.py               # QA generation prompts (4 modes)
│   │
│   ├── validation/
│   │   ├── validate.py              # Schema + content validation
│   │   └── coverage_report.py       # Coverage by subdomain
│   │
│   ├── benchmark/
│   │   ├── deploy_endpoint.py       # Deploy/delete SageMaker endpoint
│   │   ├── run_benchmark.py         # LLM-as-judge evaluation
│   │   └── run_benchmark_v3.py      # v3-specific benchmark runner
│   │
│   ├── finetune/
│   │   ├── sagemaker_finetune.py    # Full orchestration (upload→train→export)
│   │   ├── launch_job.py            # SageMaker job launcher
│   │   ├── launch_job_v3.py         # v3 reference launcher
│   │   ├── poll_and_export.py       # Poll job + export model to HF
│   │   └── poll_and_export_v3.py    # v3 reference export script
│   │
│   ├── publish_hf_dataset.py        # Push dataset to HuggingFace Hub
│   ├── budget_monitor.py            # AWS cost tracking
│   ├── aws_config.py                # AWS client factory
│   └── s3_store.py                  # S3 upload/download helpers
│
└── metadata/
    ├── source_registry.json         # Source metadata (URL, license, authority)
    └── licenses.json                # License tracking
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- AWS account with Bedrock and SageMaker access
- HuggingFace account with write token

### 1. Clone and install

```bash
git clone https://github.com/nshportun/usa-immigration.git
cd usa-immigration
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials:
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
#   HF_TOKEN / HF_USERNAME
#   S3_BUCKET
#   SAGEMAKER_ROLE_ARN / SAGEMAKER_BUCKET / SAGEMAKER_REGION
```

### 3. Run the full pipeline

```bash
# Step 1: AWS setup (creates S3 bucket, budget alert)
python main.py setup

# Step 2: Crawl all sources
python main.py crawl

# Step 3: Parse raw files → canonical corpus
python main.py parse

# Step 4: Normalize, dedup, chunk
python main.py normalize

# Step 5: Generate Q&A pairs via Bedrock
python main.py qa

# Step 6: Validate outputs, generate coverage report
python main.py validate

# Step 7: Publish dataset to HuggingFace
python scripts/publish_hf_dataset.py

# Step 8: Fine-tune on SageMaker
python scripts/finetune/launch_job.py
# Monitor the job, then export when complete:
# SAGEMAKER_JOB_NAME=<job_name> python scripts/finetune/poll_and_export.py
```

### 4. Benchmark the fine-tuned model

```bash
# Deploy endpoint (uses ml.g4dn.2xlarge by default)
python scripts/benchmark/deploy_endpoint.py deploy

# Run evaluation (LLM-as-judge with Claude Sonnet 4.6)
python scripts/benchmark/run_benchmark.py

# Delete endpoint when done
python scripts/benchmark/deploy_endpoint.py delete
```

### 5. Use the published model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "nshportun/usa-immigration-llama-3.2-3b-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {"role": "system", "content": "You are an expert on U.S. immigration law and policy. Answer accurately based on USCIS, 8 CFR, and BIA sources."},
    {"role": "user", "content": "Who is eligible to apply for adjustment of status?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Setup Details

### AWS Configuration

| Resource | Purpose |
|----------|---------|
| S3 bucket (`your-s3-bucket-name`) | Raw crawl data, processed corpus, QA pairs |
| Amazon Bedrock (Claude Sonnet 4.6, us-east-1) | QA pair generation |
| SageMaker execution role | `SageMakerExecutionRole` (us-west-2) |
| S3 bucket for SageMaker (must start with `sagemaker`) | Training data + model artifacts |
| ml.g5.2xlarge quota | Requires Service Quotas increase from 0 |

> **Note on GPU quota:** New AWS accounts default to 0 ml.g5.2xlarge capacity.
> Submit a Service Quotas increase request before running the fine-tune step.
> Approval typically takes a few minutes. The launcher automatically falls back to
> `ml.g4dn.2xlarge` if quota is exceeded.

### Required IAM permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["iam:PassRole"],
      "Resource": "arn:aws:iam::*:role/SageMakerExecutionRole"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:ListTrainingJobs"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:*"],
      "Resource": [
        "arn:aws:s3:::sagemaker-*",
        "arn:aws:s3:::sagemaker-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel"],
      "Resource": "*"
    }
  ]
}
```

---

## Dataset Details

The dataset is published at [`nshportun/usa-immigration-law-qa`](https://huggingface.co/datasets/nshportun/usa-immigration-law-qa) with two configs:

### Config: `qa`

17,058 question-answer pairs with full provenance metadata.

| Field | Type | Description |
|-------|------|-------------|
| `qa_id` | string | Unique identifier |
| `question` | string | Immigration law question |
| `answer` | string | Source-grounded answer |
| `answer_type` | string | `factual` / `procedural` / `definition` / `statistics` |
| `extraction_type` | string | `direct` / `rule_derived` / `case_derived` |
| `source_doc_id` | string | Links to corpus doc |
| `source_url` | string | Official source URL |
| `source_span` | string | Verbatim excerpt the answer is grounded in |
| `authority_level` | string | `primary_official` / `secondary_reputable` / `community` |
| `topic_tags` | list[str] | Fine-grained tags |
| `immigration_subtopic` | string | One of 13 subdomains |
| `generation_mode` | string | `faq` / `rule` / `form` / `precedent` / `statistics` |
| `time_sensitive` | bool | Whether answer may change |

### Config: `corpus`

10,056 canonical source documents.

### Subdomain Distribution

| Subdomain | Train | Eval |
|-----------|-------|------|
| Family-based immigration | 3,754 | 233 |
| Naturalization | 2,514 | 156 |
| Asylum | 1,972 | 122 |
| Adjustment of status | 1,626 | 101 |
| Removal | 1,203 | 74 |
| Humanitarian | 842 | 52 |
| Employment authorization | 784 | 48 |
| Admissibility | 521 | 32 |
| Nonimmigrant visas | 516 | 32 |
| Statistics | 133 | 8 |
| Travel documents | 103 | 6 |
| Employment-based (EB) | 70 | 4 |
| Appeals | 63 | 3 |

---

## Fine-Tuning Details

### SageMaker JumpStart Configuration

| Setting | Value |
|---------|-------|
| Base model | `meta-textgeneration-llama-3-2-3b-instruct` |
| Instance | `ml.g5.2xlarge` (24 GB VRAM) |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `o_proj` (all attention) |
| LoRA dropout | 0.05 |
| Learning rate | 5e-5 |
| Epochs | 2 |
| Batch size | 2 per device |
| Max sequence length | 1024 tokens |
| Merge LoRA | Yes (merged into base weights) |
| Training pairs | 16,065 |
| Training time | ~2–3 hours |

### Training Data Format (JumpStart dialog format)

```jsonl
{"dialog": [
  {"role": "system", "content": "You are an expert on U.S. immigration law..."},
  {"role": "user", "content": "What is the filing fee for Form I-485?"},
  {"role": "assistant", "content": "As of 2024, the filing fee for Form I-485..."}
]}
```

### Key Lessons Learned

1. **Conservative learning rate is critical** — `lr=5e-5` (or lower) with all four attention projections and 2 epochs produces stable training. Higher learning rates (≥ 2e-4) with multiple epochs cause catastrophic forgetting: the model collapses to repeating a single token.
2. **`instruction_tuned` and `chat_dataset` are mutually exclusive** — JumpStart raises a `ValueError` if both are `True`. Use `chat_dataset=True` for dialog-formatted data.
3. **LoRA rank matters for domain fit** — `r=32` with all four attention projections (`q+v+k+o`) captures significantly more domain knowledge than `r=8` with only `q+v`. The 4× improvement in fully-correct answers (v1→v3) is largely due to this.
4. **T4 GPU cannot run Flash Attention 2** — Flash Attention 2 requires sm80+ (A100/A10G). When deploying on `ml.g4dn` (T4/sm75), use the HuggingFace PyTorch Inference DLC (`huggingface-pytorch-inference:2.3.0-transformers4.48.0-gpu-py311-cu121-ubuntu22.04`) which falls back to standard attention automatically.
5. **JumpStart outputs loose files, not tar.gz** — newer JumpStart versions write model files directly to an S3 prefix rather than packaging them into `model.tar.gz`. The export script handles both formats via `list_objects_v2`.
6. **SageMaker role + S3 bucket naming** — the execution role only has permissions on buckets with `sagemaker` in the name by default. Use a `sagemaker-*` prefix for the fine-tuning bucket.

---

## Cost Estimate

| Step | Service | Approx. Cost |
|------|---------|-------------|
| Crawl (10K pages) | EC2 / local | ~$0 |
| QA generation (17K pairs) | Bedrock Claude Sonnet | ~$50–80 |
| Fine-tuning (2 epochs, g5.2xlarge) | SageMaker JumpStart | ~$10–20 |
| Benchmarking (101 questions) | Bedrock + SageMaker endpoint | ~$5–10 |
| **Total** | | **~$65–110** |

---

## Data Sources & Licenses

| Source | Type | License |
|--------|------|---------|
| USCIS Policy Manual | primary_official | Public domain (U.S. gov't work) |
| USCIS Forms & Instructions | primary_official | Public domain |
| 8 CFR / INA | primary_official | Public domain |
| BIA Precedent Decisions | primary_official | Public domain |
| DHS/CBP Statistics | primary_official | Public domain |
| harshitha008/US-immigration-laws | secondary_reputable | Apache 2.0 |
| Law StackExchange (immigration tag) | community | CC BY-SA 4.0 |

**Dataset compilation license:** CC BY 4.0

---

## Citation

If you use this dataset or model in your research:

```bibtex
@dataset{nshportun2026usaimmigration,
  author    = {nshportun},
  title     = {USA Immigration Law Q\&A Dataset},
  year      = {2026},
  url       = {https://huggingface.co/datasets/nshportun/usa-immigration-law-qa},
  note      = {17,058 source-grounded Q\&A pairs covering 13 U.S. immigration subdomains}
}

@misc{nshportun2026usaimmigrationmodel,
  author    = {nshportun},
  title     = {USA Immigration Law Llama 3.2 3B (v3)},
  year      = {2026},
  url       = {https://huggingface.co/nshportun/usa-immigration-llama-3.2-3b-v3},
  note      = {Llama 3.2 3B Instruct fine-tuned on usa-immigration-law-qa via AWS SageMaker LoRA; outperforms Llama 3 8B zero-shot baseline}
}
```

---

## Disclaimer

This dataset and model are for **research and educational purposes only**. They do not constitute legal advice. U.S. immigration law is complex and changes frequently — always consult a licensed immigration attorney for your specific situation.

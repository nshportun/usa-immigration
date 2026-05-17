# USA Immigration Law Dataset & Fine-Tuned LLM

[![Dataset on HuggingFace](https://img.shields.io/badge/Dataset-nshportun%2Fusa--immigration--law--qa-blue?logo=huggingface)](https://huggingface.co/datasets/nshportun/usa-immigration-law-qa)
[![Model on HuggingFace](https://img.shields.io/badge/Model-nshportun%2Fusa--immigration--llama--3.2--3b-orange?logo=huggingface)](https://huggingface.co/nshportun/usa-immigration-llama-3.2-3b)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Case Study:** Building a production-quality, source-grounded Q&A dataset for U.S. immigration law, then fine-tuning Llama 3.2 3B on AWS SageMaker — from crawl to published model in a single reproducible pipeline.

---

## The Problem

U.S. immigration law is one of the most complex and consequential legal domains in America — yet it is chronically underserved by AI tooling. Existing LLMs hallucinate on specific procedural questions (which form to file, what evidence is required, processing times) because:

1. **Fragmented official sources** — USCIS Policy Manual, 8 CFR regulations, BIA precedent decisions, and form instructions are scattered across dozens of government websites
2. **No structured QA dataset** — the few existing datasets (e.g., harshitha008/US-immigration-laws) lack source grounding, answer types, or subdomain labels needed for fine-tuning and evaluation
3. **No domain-adapted small model** — practitioners who need a locally-runnable or cost-efficient model have nothing to fine-tune from

**This project solves all three.**

---

## The Approach

A fully automated, end-to-end pipeline:

```
Official Sources         Raw Data             Structured Corpus         Q&A Dataset           Fine-Tuned Model
─────────────────       ──────────           ─────────────────         ───────────           ─────────────────
USCIS Policy Manual ──┐                                                                    
USCIS Forms & Inst. ──┤                                                                    
8 CFR / INA statute ──┤  Crawl &   ──►  Normalize &  ──►  Bedrock     ──►  Validate  ──►  SageMaker
BIA Precedent Decs  ──┤  Parse         Dedup & Chunk     Claude 3.5        & Split        JumpStart
DHS/CBP Statistics  ──┤                (tiktoken)        (QA gen)                         LoRA FT
harshitha008/HF     ──┤
Law StackExchange   ──┘
```

**What makes this unique:**
- Every Q&A pair carries `source_url`, `source_span`, `authority_level`, and `immigration_subtopic` — enabling citation-aware RAG
- Bedrock Claude generates structured QA in four modes: `faq`, `rule_derived`, `form`, `precedent`
- LoRA fine-tuning merged into base weights → standard `AutoModelForCausalLM` load, no adapter setup
- Two-account AWS architecture: Account 1 for crawl/storage (us-east-1), Account 2 for Bedrock + SageMaker compute

---

## Results

| Metric | Value |
|--------|-------|
| Source documents crawled | 10,056 |
| Q&A pairs generated | 17,058 |
| Immigration subdomains covered | 13 |
| Training pairs | 16,065 |
| Eval pairs | 993 (stratified) |
| Base model | Llama 3.2 3B Instruct |
| Fine-tuning method | LoRA r=8, α=32 (merged) |
| Training infrastructure | AWS SageMaker ml.g5.2xlarge |
| Training time | ~1 hr 15 min |
| Total pipeline cost | ~$15–25 (Bedrock QA gen + SageMaker) |

**Published artifacts:**
- Dataset: [`nshportun/usa-immigration-law-qa`](https://huggingface.co/datasets/nshportun/usa-immigration-law-qa) — 17,058 QA pairs + 10,056 corpus docs
- Model: [`nshportun/usa-immigration-llama-3.2-3b`](https://huggingface.co/nshportun/usa-immigration-llama-3.2-3b) — 6.4 GB merged weights

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AWS Account 1 (us-east-1)                       │
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
└───────────────────────────────────────────────────────────┼─────────────┘
                                                            │
┌───────────────────────────────────────────────────────────┼─────────────┐
│                    AWS Account 2 (us-east-1 / us-west-2)  │             │
│                                                           ▼             │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐   │
│  │   Amazon Bedrock     │    │           Amazon SageMaker           │   │
│  │  Claude Sonnet 4.6  │    │         JumpStart Fine-tuning         │   │
│  │                     │    │                                      │   │
│  │  QA Generation:     │    │  • Instance: ml.g5.2xlarge (24GB)    │   │
│  │  - faq mode         │    │  • Base: Llama 3.2 3B Instruct       │   │
│  │  - rule mode        │◄───┤  • LoRA r=8, α=32                    │   │
│  │  - form mode        │    │  • 1 epoch, batch=1, int8            │   │
│  │  - precedent mode   │    │  • Merged weights → S3               │   │
│  │                     │    │                                      │   │
│  │  17,058 QA pairs    │    │  sagemaker-immigration-finetune-2026 │   │
│  └─────────────────────┘    └──────────────────┬─────────────────┘   │
│                                                 │                       │
└─────────────────────────────────────────────────┼───────────────────────┘
                                                  │
                              ┌───────────────────▼───────────────────────┐
                              │              HuggingFace Hub               │
                              │                                            │
                              │  nshportun/usa-immigration-law-qa          │
                              │  ├── config: qa (train: 16065, eval: 993)  │
                              │  └── config: corpus (10056 docs)           │
                              │                                            │
                              │  nshportun/usa-immigration-llama-3.2-3b    │
                              │  └── Llama 3.2 3B + LoRA (merged, 6.4GB)  │
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
│   ├── finetune/
│   │   ├── sagemaker_finetune.py    # Full orchestration (upload→train→export)
│   │   ├── launch_job.py            # Low-level SageMaker job launcher
│   │   └── poll_and_export.py       # Poll job + export model to HF
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
- AWS CLI configured (or `.env` file)
- Two AWS accounts (Account 1: crawl/storage; Account 2: Bedrock + SageMaker)
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
# Edit .env with your actual credentials:
#   ACCOUNT1_AWS_ACCESS_KEY_ID / ACCOUNT1_AWS_SECRET_ACCESS_KEY
#   ACCOUNT2_AWS_ACCESS_KEY_ID / ACCOUNT2_AWS_SECRET_ACCESS_KEY
#   ACCOUNT2_BEDROCK_API_KEY
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

# Step 5: Generate Q&A pairs via Bedrock (Account 2)
# Set ACTIVE_ACCOUNT=2 in .env first
python main.py qa

# Step 6: Validate outputs, generate coverage report
python main.py validate

# Step 7: Publish dataset to HuggingFace
python scripts/publish_hf_dataset.py

# Step 8: Fine-tune on SageMaker and publish model
python scripts/finetune/sagemaker_finetune.py
```

### 4. Use the published model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "nshportun/usa-immigration-llama-3.2-3b"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are an expert on U.S. immigration law and policy. Answer accurately based on official sources."},
    {"role": "user", "content": "Who is eligible to apply for adjustment of status?"}
]
result = pipe(messages, max_new_tokens=512)
print(result[0]["generated_text"][-1]["content"])
```

---

## AWS Setup Details

### Account 1 — Crawl & Storage

| Resource | Purpose |
|----------|---------|
| S3 bucket (`your-s3-bucket-name`) | Raw crawl data, processed corpus, QA pairs |
| AWS Budget | Cost guard — pipeline stops at `$BUDGET_LIMIT_USD` |
| IAM user with S3 + Bedrock read | `ACCOUNT1_*` credentials |

### Account 2 — Compute

| Resource | Purpose |
|----------|---------|
| Amazon Bedrock (Claude Sonnet 4.6) | QA pair generation |
| SageMaker execution role | `SageMakerExecutionRole-ImmigrationFT` |
| S3 bucket (name must start with `sagemaker`) | Training data + model artifacts |
| ml.g5.2xlarge quota | Requires Service Quotas increase from 0 |

> **Note on GPU quota:** New AWS accounts start with 0 ml.g5.2xlarge capacity. Submit a Service Quotas increase request for `ml.g5.2xlarge` training jobs before running the fine-tune step. Approval typically takes a few minutes.

### Required IAM permissions for Account 2 user

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["iam:PassRole"],
      "Resource": "arn:aws:iam::*:role/SageMakerExecutionRole-ImmigrationFT"
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
| Method | LoRA (r=8, α=32, target: q_proj + v_proj) |
| Quantization | int8 (required to fit 24GB VRAM) |
| Batch size | 1 per device |
| Max sequence length | 512 tokens |
| Learning rate | 1e-4 |
| Epochs | 1 |
| Merge LoRA | Yes (merged into base weights) |

### Training Data Format (JumpStart dialog format)

```jsonl
{"dialog": [
  {"role": "system", "content": "You are an expert on U.S. immigration law..."},
  {"role": "user", "content": "What is the filing fee for Form I-485?"},
  {"role": "assistant", "content": "As of 2024, the filing fee for Form I-485..."}
]}
```

### Key Lessons Learned

1. **`instruction_tuned` and `chat_dataset` are mutually exclusive** — JumpStart raises a `ValueError` if both are `True`. Use `chat_dataset=True` for dialog-formatted data.
2. **OOM on ml.g5.2xlarge with Llama 3.2 3B** — requires `int8_quantization=True` + `per_device_train_batch_size=1` + `max_input_length=512`.
3. **SageMaker Studio role + S3 bucket naming** — auto-created Studio roles only allow buckets with `sagemaker` in the name. Use `sagemaker-*` prefix.
4. **ml.g5.2xlarge GPU quota** — new AWS accounts default to 0. Submit Service Quotas increase before launching.

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
  title     = {USA Immigration Law Llama 3.2 3B},
  year      = {2026},
  url       = {https://huggingface.co/nshportun/usa-immigration-llama-3.2-3b},
  note      = {Llama 3.2 3B Instruct fine-tuned on usa-immigration-law-qa via AWS SageMaker LoRA}
}
```

---

## Disclaimer

This dataset and model are for **research and educational purposes only**. They do not constitute legal advice. U.S. immigration law is complex and changes frequently — always consult a licensed immigration attorney for your specific situation.

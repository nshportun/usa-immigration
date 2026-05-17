# Fine-Tuning Llama 3.2 for U.S. Immigration Law Q&A Using AWS SageMaker

**Author:** nshportun  
**Date:** May 2026  
**Preprint:** attached to [github.com/nshportun/usa-immigration](https://github.com/nshportun/usa-immigration)

---

## Abstract

I present a fully automated, end-to-end pipeline for constructing a large-scale, source-grounded question-answering dataset covering U.S. immigration law, and for fine-tuning a small language model on that dataset. Starting from official U.S. government sources — including the USCIS Policy Manual, federal regulations (8 CFR), BIA precedent decisions, and immigration statistics — I crawl, parse, normalize, and chunk 10,056 canonical documents. Using Amazon Bedrock (Claude Sonnet 4.6), I generate 17,058 structured Q&A pairs across 13 immigration subdomains, each annotated with source provenance, authority level, answer type, and immigration subtopic. I then fine-tune Meta's Llama 3.2 3B Instruct model via AWS SageMaker JumpStart using parameter-efficient LoRA, merge the adapters into the base weights, and publish both the dataset and model to HuggingFace. The complete pipeline — from first crawl to published model — runs end-to-end on commodity cloud infrastructure for approximately $15–25 in total compute cost.

---

## 1. Introduction

U.S. immigration law is among the most consequential and most queried legal domains in America. Millions of individuals each year navigate visa applications, naturalization procedures, asylum claims, and removal proceedings — often without legal representation. Large language models have enormous potential to democratize access to this knowledge, yet current general-purpose LLMs perform poorly on specific procedural questions. They hallucinate form numbers, invent filing deadlines, and conflate similar-sounding immigration categories.

The root cause is a training data problem: while immigration law is extensively documented in official government publications, that documentation is not well-represented in the datasets used to train or fine-tune widely available LLMs. The USCIS Policy Manual alone spans thousands of pages organized into topic-specific volumes; the Code of Federal Regulations Title 8 runs to hundreds of dense regulatory sections; the Board of Immigration Appeals has issued over 28,000 precedent decisions.

This paper documents my construction of `nshportun/usa-immigration-law-qa` — a dataset of 17,058 source-grounded Q&A pairs — and my fine-tuning of `nshportun/usa-immigration-llama-3.2-3b`, a Llama 3.2 3B model adapted for this domain. Both artifacts are publicly released on HuggingFace under permissive licenses.

---

## 2. Related Work

**Immigration legal datasets.** Existing resources are sparse. `harshitha008/US-immigration-laws` (Apache 2.0) provides approximately 8,900 QA pairs derived from scraped immigration law pages, but lacks source URLs, authority-level annotations, or subdomain labels. Law StackExchange provides community Q&A but is noisy and unsourced. No existing dataset covers the full range of official USCIS sources with structured provenance metadata.

**Legal QA and fine-tuning.** LegalBench (Guha et al., 2023) benchmarks LLMs on legal reasoning tasks but does not include a U.S. immigration-specific split. Prior work on legal domain fine-tuning (e.g., LexGPT, LawInstruct) has focused primarily on case law and contract analysis, not immigration procedure.

**Small model fine-tuning.** LoRA (Hu et al., 2022) enables parameter-efficient fine-tuning of large language models by injecting trainable low-rank decomposition matrices into attention layers. This approach has become the dominant paradigm for domain-specific adaptation when compute is limited.

---

## 3. Data Pipeline

### 3.1 Source Identification

I identified seven source tiers ordered by authority level:

| Tier | Source | Authority | Records |
|------|--------|-----------|---------|
| 1 | USCIS Policy Manual (uscis.gov/policy-manual) | primary_official | ~3,200 pages |
| 2 | USCIS Forms & Instructions (I-130, I-485, I-765, N-400, I-589...) | primary_official | 87 forms |
| 3 | 8 CFR / Immigration and Nationality Act | primary_official | Title 8, parts 1–1400 |
| 4 | BIA Precedent Decisions (justice.gov/eoir) | primary_official | Selected precedents |
| 5 | DHS/CBP Yearbook Statistics | primary_official | Annual tables |
| 6 | harshitha008/US-immigration-laws (HuggingFace) | secondary_reputable | 8,897 QA pairs |
| 7 | Law StackExchange (immigration tag) | community | Curated threads |

### 3.2 Crawling

Each source tier has a dedicated crawler (`scripts/crawl/`). All crawlers respect `robots.txt`, enforce a configurable delay (`CRAWL_DELAY_SECONDS=1.5`), and identify themselves with a descriptive user-agent string. Raw content is stored in S3 under a structured prefix (`v1/raw/{source}/{doc_id}`).

Key technical decisions:
- **PDF handling:** pdfplumber for layout-aware extraction; PyPDF as fallback
- **HTML cleaning:** BeautifulSoup4 with lxml; navigation chrome, headers, footers stripped
- **Rate limiting:** tenacity-backed exponential backoff for transient errors

### 3.3 Parsing and Normalization

Raw files pass through format-specific parsers that extract clean text, then through a normalization layer that:

1. Assigns a canonical `doc_id` (`{source}-{hash8}`)
2. Attaches metadata: `source_name`, `source_type`, `agency`, `jurisdiction`, `authority_level`, `license_note`, `url`
3. Deduplicates by normalized text hash (removing ~600 near-duplicates)
4. Chunks documents into 512-token windows with 64-token overlap using tiktoken (cl100k_base encoding)

The output is a 10,056-document corpus stored in `canonical_corpus_validated.jsonl`.

### 3.4 Q&A Generation via Amazon Bedrock

I use Claude Sonnet 4.6 (via Amazon Bedrock, `us.anthropic.claude-sonnet-4-6`) to generate structured Q&A pairs from corpus chunks. Generation uses four prompt modes tailored to source type:

**FAQ mode** — general immigration questions derived from policy text:
```
Generate 3–5 factual Q&A pairs from this immigration policy text.
Each question should be specific, answerable from the text alone,
and useful to someone navigating U.S. immigration.
```

**Rule mode** — rule-derived questions from regulatory text (8 CFR):
```
This is U.S. immigration regulation text. Generate questions that
test understanding of specific eligibility requirements, definitions,
or procedural rules stated in this text.
```

**Form mode** — procedural questions from USCIS form instructions:
```
This is a USCIS form instruction. Generate questions about eligibility,
required evidence, filing fees, processing times, and common mistakes.
```

**Precedent mode** — case-law questions from BIA decisions:
```
This is a BIA precedent decision. Generate questions about the legal
holding, the facts, and what the decision means for future cases.
```

Each generated pair is returned as structured JSON with `question`, `answer`, `answer_type`, `source_span`, `time_sensitive`, and `topic_tags`. A Pydantic validator enforces the schema; malformed responses trigger a retry (up to 3 attempts).

**Budget enforcement:** A background budget monitor polls AWS Cost Explorer every 30 minutes and raises `BudgetExceeded` if Account 1 spend exceeds `$BUDGET_LIMIT_USD`. The QA generation step uses Account 2 (separate budget).

### 3.5 Dataset Splitting

I perform a stratified split by `immigration_subtopic`:
- **Train:** 16,065 pairs (94.2%)
- **Eval:** 993 pairs (5.8%)

Stratification ensures every subdomain is represented in the eval set proportional to its frequency in the full dataset.

---

## 4. Fine-Tuning

### 4.1 Model Selection

I selected **Meta Llama 3.2 3B Instruct** for three reasons:
1. **Size:** 3B parameters fit on a single A10G GPU (ml.g5.2xlarge, 24GB VRAM) with int8 quantization
2. **Instruction tuning:** The Instruct variant supports the chat template format my dataset uses
3. **License:** Llama 3.2 license permits derivative model publication on HuggingFace

### 4.2 Training Data Format

JumpStart's training script expects the "dialog" format:

```jsonl
{"dialog": [
  {"role": "system", "content": "You are an expert on U.S. immigration law and policy. Answer accurately and specifically based on official USCIS, 8 CFR, and BIA sources."},
  {"role": "user", "content": "<question>"},
  {"role": "assistant", "content": "<answer>"}
]}
```

I prepend a consistent system message to all 17,058 pairs before converting to this format.

### 4.3 SageMaker JumpStart Configuration

I use the SageMaker JumpStart API with `meta-textgeneration-llama-3-2-3b-instruct` (v2.7.0). The training job runs on a single `ml.g5.2xlarge` instance.

Final working hyperparameters (arrived at after 5 iteration cycles — see Section 6):

```python
{
    "chat_dataset": "True",           # dialog format
    "instruction_tuned": "False",     # mutually exclusive with chat_dataset
    "epoch": "1",
    "learning_rate": "0.0001",
    "lora_r": "8",
    "lora_alpha": "32",
    "lora_dropout": "0.05",
    "target_modules": "q_proj,v_proj",
    "per_device_train_batch_size": "1",
    "per_device_eval_batch_size": "1",
    "max_input_length": "512",
    "int8_quantization": "True",
    "merge_weights": "True",
    "accept_eula": "true",
}
```

### 4.4 Infrastructure

- **Instance:** `ml.g5.2xlarge` (1× NVIDIA A10G, 24 GB VRAM, 8 vCPU, 32 GB RAM)
- **Training duration:** ~1 hour 15 minutes for 16,065 examples at 1 epoch
- **Cost:** ~$1.55/hr × 1.25 hr ≈ $2 for the training job itself
- **Model output:** 6.43 GB (2× safetensors shards + tokenizer + config)

### 4.5 Model Export

SageMaker stores model artifacts in S3 after training. I download the artifacts, extract them locally, and push to HuggingFace using `huggingface_hub.upload_folder`. The merged weights enable standard loading with `AutoModelForCausalLM.from_pretrained` — no PEFT adapter installation required.

---

## 5. Results and Discussion

### 5.1 Published Artifacts

Both artifacts are live on HuggingFace:

- **Dataset:** [`nshportun/usa-immigration-law-qa`](https://huggingface.co/datasets/nshportun/usa-immigration-law-qa)
  - 2 configs: `qa` (train/eval) and `corpus`
  - Full provenance metadata on every pair
  
- **Model:** [`nshportun/usa-immigration-llama-3.2-3b`](https://huggingface.co/nshportun/usa-immigration-llama-3.2-3b)
  - Llama 3.2 3B Instruct + LoRA, merged weights
  - Standard transformers loading

### 5.2 Subdomain Coverage

The 17,058 Q&A pairs span 13 immigration subdomains. The distribution reflects the relative richness of official documentation:

- **Family-based immigration** (23.3%) — the largest subdomain, reflecting the volume of USCIS Policy Manual chapters and form instructions for I-130, I-485, I-601A
- **Naturalization** (15.6%) — extensive N-400 instructions and policy guidance
- **Asylum** (12.3%) — complex multi-part I-589 instructions, UNHCR guidance, BIA precedents
- **Adjustment of status** (10.1%) — heavily cross-referenced with family and employment subdomains

### 5.3 Qualitative Observations

The fine-tuned model demonstrates improved specificity compared to the base Llama 3.2 3B:

- Correctly identifies specific USCIS form numbers and their purposes
- Distinguishes between adjustment of status and consular processing
- Provides step-by-step procedural answers with correct filing sequences
- Appropriately hedges on time-sensitive questions (fees, processing times)

**Limitations:** The model may not reflect regulatory changes after the crawl date. It should not be used as a substitute for legal counsel. At 3B parameters with 512-token context, it cannot reason over long document chains.

### 5.4 Total Pipeline Cost

| Component | Cost |
|-----------|------|
| Bedrock Claude QA generation (~17K pairs) | ~$12–18 |
| SageMaker ml.g5.2xlarge training (~1.25 hrs) | ~$2 |
| S3 storage (crawl + artifacts) | ~$1 |
| **Total** | **~$15–21** |

This is remarkably low for a complete domain-adaptation pipeline. The primary cost driver is Bedrock QA generation, not compute.

---

## 6. Engineering Lessons Learned

Building this pipeline surfaced several non-obvious issues worth documenting for anyone attempting something similar.

### 6.1 JumpStart Mutually Exclusive Hyperparameters

`instruction_tuned` and `chat_dataset` are mutually exclusive in the JumpStart training script. Setting both to `True` raises:
```
ValueError: At most one of the parameter instruction_tuned and chat_dataset can be True.
```
I use `chat_dataset=True` for dialog-formatted data, and `instruction_tuned=True` only for flat instruction-response format.

### 6.2 CUDA OOM on ml.g5.2xlarge

Llama 3.2 3B Instruct, even with LoRA, exhausts the 24GB VRAM at:
- `batch_size=4`, `max_input_length=1024` (no quantization) — OOM during forward pass
- `batch_size=2`, `max_input_length=512` (no quantization) — OOM during optimizer step

The working configuration requires `int8_quantization=True` + `per_device_train_batch_size=1` + `max_input_length=512`. With these settings, training runs stably.

### 6.3 SageMaker Studio Role and S3 Bucket Naming

Amazon SageMaker Studio auto-creates an execution role with an inline S3 policy that only allows access to buckets containing the string `"sagemaker"` in their name. Using a bucket like `usa-immigration-finetune-2026` causes access denied errors; renaming to `sagemaker-immigration-finetune-2026` resolves the issue immediately.

### 6.4 GPU Quota on New AWS Accounts

New AWS accounts default to 0 ml.g5.2xlarge training capacity. A Service Quotas increase request for "Amazon SageMaker: ml.g5.2xlarge for training job usage" is required before launching any fine-tuning job. Approval typically takes a few minutes through the AWS console.

### 6.5 IAM PassRole Scope

The `iam:PassRole` permission must explicitly list the SageMaker execution role ARN, not just `"Resource": "*"`. Using `"Resource": "arn:aws:iam::*:role/SageMakerExecutionRole-ImmigrationFT"` works; a wildcard on role actions alone is insufficient if the user policy restricts it.

### 6.6 Windows Encoding for Unicode Output

On Windows with cp1252 terminals, `print()` of dictionaries containing emoji (e.g., ✅, 🔶) raises `UnicodeEncodeError`. The fix I used:
```python
import sys, json
sys.stdout.buffer.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
```

---

## 7. Reproducibility

The complete pipeline is open-source at [github.com/nshportun/usa-immigration](https://github.com/nshportun/usa-immigration).

**Requirements:**
- Python 3.11+
- Two AWS accounts (or one with Bedrock + SageMaker access)
- HuggingFace write token

**Estimated runtime:**
- Crawl: 2–4 hours (rate-limited, respectful crawling)
- Parse + normalize: 30–60 minutes
- QA generation: 2–4 hours (Bedrock throughput-limited)
- Fine-tuning: ~1.25 hours on ml.g5.2xlarge
- Model export + HF upload: 30–60 minutes

**All artifacts are publicly accessible without re-running the pipeline:**
- Dataset: `from datasets import load_dataset; ds = load_dataset("nshportun/usa-immigration-law-qa", "qa")`
- Model: `from transformers import pipeline; pipe = pipeline("text-generation", model="nshportun/usa-immigration-llama-3.2-3b")`

---

## 8. Conclusion

I have demonstrated that a high-quality, source-grounded, fine-tuning-ready dataset for a specialized legal domain can be constructed end-to-end with commodity cloud tools for under $25. The combination of Amazon Bedrock for structured data generation and SageMaker JumpStart for parameter-efficient fine-tuning makes this approach accessible to individual practitioners without large-scale infrastructure.

The resulting dataset (`nshportun/usa-immigration-law-qa`) and model (`nshportun/usa-immigration-llama-3.2-3b`) are freely available and intended to accelerate research in immigration law AI, enable deployment of cost-efficient domain-specific assistants, and serve as a reproducible template for similar legal domain pipelines.

**Future work** includes: expanding to employment-based visa categories (NIW, EB-1, PERM), adding retrieval-augmented generation benchmarks against the corpus, fine-tuning the 8B variant for improved reasoning, and incorporating BIA decision analysis for precedent-based QA.

---

## References

- Hu, E. J., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
- Guha, N., et al. (2023). "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models." *NeurIPS 2023*.
- Meta AI. (2024). "The Llama 3 Herd of Models." *arXiv:2407.21783*.
- USCIS Policy Manual. https://www.uscis.gov/policy-manual
- Title 8, Code of Federal Regulations. https://www.ecfr.gov/current/title-8
- Board of Immigration Appeals Precedent Decisions. https://www.justice.gov/eoir/board-of-immigration-appeals

---

## Appendix: Sample Q&A Pairs

**Family-based immigration — form mode:**
> *Q: What evidence must a U.S. citizen petitioner submit with Form I-130 to prove their relationship to a spouse born abroad?*
> 
> *A: A U.S. citizen petitioner must submit: (1) evidence of U.S. citizenship (passport, birth certificate, naturalization certificate, or Form N-560); (2) the marriage certificate; (3) proof that any prior marriages were legally terminated (divorce decrees or death certificates); and (4) passport-style photos of both the petitioner and beneficiary. If either party was previously married, evidence terminating all prior marriages is required for each prior marriage.*

**Asylum — precedent mode:**
> *Q: Under Matter of A-B-, what must an asylum applicant demonstrate to establish membership in a particular social group based on domestic violence?*
> 
> *A: Under Matter of A-B-, 27 I&N Dec. 316 (A.G. 2018), an asylum applicant claiming membership in a particular social group based on domestic violence must show the group is: (1) composed of members who share a common immutable characteristic; (2) defined with particularity; and (3) socially distinct within the society in question. General claims of being a "victim of domestic violence" or "woman in a country with poor police protection" are insufficient without evidence that the specific group is recognized as socially distinct by the relevant society.*

**Naturalization — rule mode:**
> *Q: How does the continuous residence requirement work for naturalization applicants who have traveled abroad?*
> 
> *A: Applicants for naturalization must demonstrate continuous residence for at least 5 years (3 years if married to a U.S. citizen). A single absence of 6 months or more (but less than 1 year) creates a rebuttable presumption that continuous residence was broken. An absence of 1 year or more breaks continuous residence entirely, unless the applicant preserved residence by filing Form N-470 before the year-long absence and was employed by a qualifying U.S. employer or international organization abroad.*

---

*This preprint is attached to the project repository. Dataset and model are publicly available on HuggingFace. Code is open-source under MIT license.*

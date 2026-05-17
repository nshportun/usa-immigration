"""
Publish the USA Immigration Law Q&A dataset to HuggingFace.

Uploads three splits:
  - train  (~16,065 Q&A pairs)
  - eval   (~993 Q&A pairs, stratified)
  - corpus (10,056 canonical documents)

Repo: nshportun/usa-immigration-law-qa
"""

import json
import os
import pathlib
import sys
from datasets import Dataset, DatasetDict, Features, Value, Sequence

import structlog
log = structlog.get_logger()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "nshportun")
DATASET_REPO = f"{HF_USERNAME}/usa-immigration-law-qa"

BASE = pathlib.Path(__file__).resolve().parents[1]
SPLITS_DIR = BASE / "data_local" / "splits"
CORPUS_PATH = BASE / "data_local" / "canonical_corpus_validated.jsonl"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def normalize_qa(row: dict) -> dict:
    """Ensure consistent field types for all QA rows."""
    return {
        "qa_id": str(row.get("qa_id") or ""),
        "question": str(row.get("question") or ""),
        "answer": str(row.get("answer") or ""),
        "answer_type": str(row.get("answer_type") or "factual"),
        "extraction_type": str(row.get("extraction_type") or ""),
        "source_doc_id": str(row.get("source_doc_id") or ""),
        "source_url": str(row.get("source_url") or ""),
        "source_span": str(row.get("source_span") or ""),
        "authority_level": str(row.get("authority_level") or ""),
        "topic_tags": [str(t) for t in (row.get("topic_tags") or [])],
        "immigration_subtopic": str(row.get("immigration_subtopic") or ""),
        "generation_mode": str(row.get("generation_mode") or ""),
        "review_status": str(row.get("review_status") or "draft"),
        "time_sensitive": bool(row.get("time_sensitive") or False),
    }


def normalize_corpus(row: dict) -> dict:
    return {
        "doc_id": str(row.get("doc_id") or ""),
        "source_name": str(row.get("source_name") or ""),
        "source_type": str(row.get("source_type") or ""),
        "agency": str(row.get("agency") or ""),
        "jurisdiction": str(row.get("jurisdiction") or "US"),
        "title": str(row.get("title") or ""),
        "url": str(row.get("url") or ""),
        "text": str(row.get("text") or ""),
        "authority_level": str(row.get("authority_level") or ""),
        "license_note": str(row.get("license_note") or ""),
    }


README = """\
---
language:
- en
license: cc-by-4.0
tags:
- legal
- immigration
- question-answering
- retrieval-augmented-generation
- united-states
- fine-tuning
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- text-generation
pretty_name: USA Immigration Law Q&A Dataset
---

# USA Immigration Law Q&A Dataset

A large-scale, source-grounded Q&A dataset covering U.S. immigration law and policy,
built entirely from official government sources, open legal datasets, and curated
community materials.

## Dataset Contents

| Split | Records | Description |
|-------|---------|-------------|
| `train` | 16,065 | Training Q&A pairs |
| `eval` | 993 | Stratified held-out evaluation set |
| `corpus` | 10,056 | Canonical source documents |

## Coverage (Q&A pairs by subdomain)

| Subdomain | Train | Eval |
|-----------|-------|------|
| Family-based immigration | ~3,754 | 233 |
| Naturalization | ~2,514 | 156 |
| Asylum | ~1,972 | 122 |
| General immigration | ~1,964 | 122 |
| Adjustment of status | ~1,626 | 101 |
| Removal | ~1,203 | 74 |
| Humanitarian | ~842 | 52 |
| Employment authorization | ~784 | 48 |
| Admissibility | ~521 | 32 |
| Nonimmigrant visas | ~516 | 32 |
| Statistics | ~133 | 8 |
| Travel documents | ~103 | 6 |
| Employment-based (EB) | ~70 | 4 |
| Appeals | ~63 | 3 |

## Data Sources

- **USCIS Policy Manual** (primary_official) — official adjudication policy
- **USCIS Forms & Instructions** (primary_official) — I-130, I-485, I-765, N-400, I-589, etc.
- **8 CFR / INA** (primary_official) — federal regulations and statute text
- **BIA Precedent Decisions** (primary_official) — Board of Immigration Appeals
- **DHS/CBP Statistics** (primary_official) — yearbook tables, enforcement data
- **harshitha008/US-immigration-laws** (secondary_reputable) — 8,897 QA pairs, Apache 2.0
- **Law StackExchange** (community) — immigration-tagged expert Q&A

## Q&A Pair Schema

```python
{
    "qa_id": "qa-000001",
    "question": "Who may file Form I-130?",
    "answer": "A U.S. citizen or lawful permanent resident...",
    "answer_type": "factual",              # factual | procedural | definition | statistics
    "extraction_type": "rule_derived",     # direct | rule_derived | case_derived
    "source_doc_id": "uscis-form-i130",
    "source_url": "https://www.uscis.gov/i-130",
    "source_span": "Who May File...",
    "authority_level": "primary_official", # primary_official | secondary_reputable | community_non_authoritative
    "topic_tags": ["family-based", "I-130"],
    "immigration_subtopic": "family-based immigration",
    "generation_mode": "rule",             # faq | rule | form | precedent | statistics
    "review_status": "draft",
    "time_sensitive": false
}
```

## Intended Use

- **RAG** — retrieval-augmented generation for immigration legal assistants
- **Fine-tuning** — domain adaptation of small LLMs (Llama 3.2 3B/8B recommended)
- **Benchmarking** — evaluating LLM performance on U.S. immigration law

## ⚠️ Disclaimer

This dataset is for research and educational purposes only. It does not constitute
legal advice. Immigration law is complex and changes frequently — always consult a
licensed immigration attorney for your specific situation.

## License

- Government source text: public domain (U.S. government works)
- HF-sourced data (`harshitha008/US-immigration-laws`): Apache 2.0
- Community data: CC BY-SA 4.0
- Dataset compilation: **CC BY 4.0**

## Citation

```bibtex
@dataset{nshportun2026usaimmigration,
  author = {nshportun},
  title  = {USA Immigration Law Q\\&A Dataset},
  year   = {2026},
  url    = {https://huggingface.co/datasets/nshportun/usa-immigration-law-qa}
}
```
"""


def main():
    os.chdir(BASE)
    log.info("loading_splits")
    train_rows = [normalize_qa(r) for r in load_jsonl(SPLITS_DIR / "train.jsonl")]
    eval_rows  = [normalize_qa(r) for r in load_jsonl(SPLITS_DIR / "eval.jsonl")]
    corpus_rows = [normalize_corpus(r) for r in load_jsonl(CORPUS_PATH)]

    log.info("splits_loaded", train=len(train_rows), eval=len(eval_rows), corpus=len(corpus_rows))

    train_ds  = Dataset.from_list(train_rows)
    eval_ds   = Dataset.from_list(eval_rows)
    corpus_ds = Dataset.from_list(corpus_rows)

    dd = DatasetDict({
        "train":  train_ds,
        "eval":   eval_ds,
        "corpus": corpus_ds,
    })

    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    # DatasetDict requires all splits share same schema; push QA splits together,
    # corpus as a separate config.
    qa_dd = DatasetDict({
        "train": train_ds,
        "eval":  eval_ds,
    })
    corpus_dd = DatasetDict({
        "corpus": corpus_ds,
    })

    log.info("pushing_qa_splits", repo=DATASET_REPO)
    qa_dd.push_to_hub(
        DATASET_REPO,
        config_name="qa",
        token=HF_TOKEN,
        private=False,
        commit_message="Add QA train/eval splits — 17,058 pairs, 13 immigration subdomains",
    )

    log.info("pushing_corpus_split", repo=DATASET_REPO)
    corpus_dd.push_to_hub(
        DATASET_REPO,
        config_name="corpus",
        token=HF_TOKEN,
        private=False,
        commit_message="Add canonical corpus — 10,056 source documents",
    )

    # Upload README / dataset card
    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    log.info("dataset_published", url=f"https://huggingface.co/datasets/{DATASET_REPO}")
    print(f"\n✓ Dataset published: https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    main()

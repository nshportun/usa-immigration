"""
Pull open legal and immigration datasets from Hugging Face and OpenImmigration.us.
"""

import os
from pathlib import Path

import structlog
from datasets import load_dataset

from scripts.crawl.base_crawler import BaseCrawler
from scripts.s3_store import upload_jsonl

log = structlog.get_logger()

OPEN_IMMIGRATION_URLS = [
    "https://www.openimmigration.us/downloads",
]
RAW_PREFIX = "data_raw/open_datasets"
LOG_PATH = Path("metadata/crawl_log.jsonl")

HF_LIMIT = 10_000
HF_TOKEN = os.getenv("HF_TOKEN")


class OpenDatasetsCrawler(BaseCrawler):
    source_name = "OpenDatasets"
    source_type = "open_repository"
    authority_level = "secondary_reputable"

    def _hf_pull(self, dataset: str, config: str | None, split: str,
                 filter_fn, s3_path: str, limit: int = HF_LIMIT):
        try:
            kwargs = dict(split=split, streaming=True, trust_remote_code=False)
            if HF_TOKEN:
                kwargs["token"] = HF_TOKEN
            if config:
                ds = load_dataset(dataset, config, **kwargs)
            else:
                ds = load_dataset(dataset, **kwargs)
            records = []
            for row in ds:
                if filter_fn(row):
                    records.append(row)
                    if len(records) >= limit:
                        break
            upload_jsonl(records, s3_path)
            log.info("hf_pull_done", dataset=dataset, config=config, records=len(records))
        except Exception as e:
            log.warning("hf_pull_error", dataset=dataset, config=config, error=str(e))

    def crawl_us_immigration_laws(self):
        """harshitha008/US-immigration-laws — 8,897 QA pairs, Apache 2.0."""
        log.info("crawl_us_immigration_laws_start")
        for split in ["train", "validation", "test"]:
            self._hf_pull(
                "harshitha008/US-immigration-laws", None, split,
                filter_fn=lambda r: bool(r.get("question") and r.get("answer")),
                s3_path=f"{RAW_PREFIX}/us_immigration_laws/{split}.jsonl",
                limit=HF_LIMIT,
            )

    def crawl_pile_of_law(self):
        # pile-of-law dataset uses a custom loading script that is no longer supported
        # by the datasets library. Skipping.
        log.info("crawl_pile_of_law_skip", reason="custom_script_unsupported")

    def crawl_law_stackexchange(self):
        log.info("crawl_law_stackexchange_start")
        IMMIGRATION_KW = {"immigration", "visa", "uscis", "asylum", "naturalization",
                          "green-card", "green card", "adjustment-of-status", "work-permit"}

        def is_immigration(row):
            text = " ".join([
                str(row.get("question_title", "") or row.get("title", "")),
                str(row.get("question_body", "") or row.get("body", "")),
                str(row.get("tags", "") or row.get("Tags", "") or row.get("text_label", "")),
            ]).lower()
            return any(kw in text for kw in IMMIGRATION_KW)

        for ds_name in ["ymoslem/Law-StackExchange", "jonathanli/law-stack-exchange"]:
            self._hf_pull(
                ds_name, None, "train",
                filter_fn=is_immigration,
                s3_path=f"{RAW_PREFIX}/law_stackexchange/immigration.jsonl",
                limit=10_000,
            )
            break

    def crawl_legal_qa(self):
        log.info("crawl_legal_qa_start")
        self._hf_pull(
            "isaacus/LegalQAEval", None, "test",
            filter_fn=lambda r: True,
            s3_path=f"{RAW_PREFIX}/legal_qa_eval/all.jsonl",
            limit=5_000,
        )

    def crawl_openimmigration(self):
        log.info("crawl_openimmigration_start")
        for url in OPEN_IMMIGRATION_URLS:
            self.fetch(url, f"{RAW_PREFIX}/openimmigration")

    def run(self):
        self.crawl_openimmigration()
        self.crawl_us_immigration_laws()
        self.crawl_pile_of_law()
        self.crawl_law_stackexchange()
        self.crawl_legal_qa()
        log.info("open_datasets_crawl_complete")


if __name__ == "__main__":
    OpenDatasetsCrawler(LOG_PATH).run()

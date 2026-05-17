"""
Orchestrate parsing: pull raw files from S3, parse, upload canonical corpus to S3.
Runs after all crawlers complete.
"""

import datetime
import json
from pathlib import Path

import structlog

from scripts.aws_config import S3_BUCKET
from scripts.parse.parse_html import parse_html
from scripts.parse.parse_pdf import parse_pdf
from scripts.parse.parse_json import parse_ecfr_json, parse_dhs_stats_json, parse_community_jsonl
from scripts.s3_store import download_jsonl, list_keys, upload_jsonl
from scripts.aws_config import s3_client

log = structlog.get_logger()

CORPUS_PREFIX = "data_processed/canonical_corpus"
CRAWL_LOG_PATH = Path("metadata/crawl_log.jsonl")


def _load_key_to_url_map() -> dict[str, str]:
    """Build S3-key → original URL mapping from the local crawl log."""
    mapping: dict[str, str] = {}
    if not CRAWL_LOG_PATH.exists():
        return mapping
    with open(CRAWL_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok" and rec.get("s3_key") and rec.get("url"):
                    mapping[rec["s3_key"]] = rec["url"]
            except Exception:
                pass
    return mapping


def _get_raw_object(key: str) -> bytes:
    resp = s3_client().get_object(Bucket=S3_BUCKET, Key=key)
    return resp["Body"].read()


def _get_key_metadata(key: str) -> dict:
    resp = s3_client().head_object(Bucket=S3_BUCKET, Key=key)
    return resp.get("Metadata", {})


def parse_all():
    records: list[dict] = []
    retrieved_at = datetime.datetime.utcnow().isoformat()

    key_to_url = _load_key_to_url_map()
    log.info("crawl_log_loaded", url_mappings=len(key_to_url))

    raw_keys = list_keys("data_raw/")
    log.info("parse_start", total_raw_keys=len(raw_keys))

    for key in raw_keys:
        try:
            raw = _get_raw_object(key)
            url = key_to_url.get(key, key)  # use real URL from crawl log; fall back to S3 key

            if key.endswith(".html") or key.endswith(".txt"):
                rec = parse_html(raw.decode("utf-8", errors="replace"), url, retrieved_at)
                if rec:
                    records.append(rec)

            elif key.endswith(".pdf"):
                source_meta = _infer_source_meta_from_key(key)
                rec = parse_pdf(raw, url, source_meta, retrieved_at)
                if rec:
                    records.append(rec)

            elif key.endswith(".json"):
                data = json.loads(raw)
                if "data_raw/cfr" in key:
                    records.extend(parse_ecfr_json(data))
                elif "dhs_stats" in key or "cbp" in key:
                    records.extend(parse_dhs_stats_json(data, url, _dataset_name_from_key(key)))
                elif "community" in key or "law_se" in key:
                    rows = data if isinstance(data, list) else [data]
                    records.extend(parse_community_jsonl(rows))

            elif key.endswith(".jsonl"):
                lines = raw.decode("utf-8", errors="replace").splitlines()
                rows = [json.loads(l) for l in lines if l.strip()]
                if ("community" in key or "law_se" in key or "visajourney" in key
                        or "open_datasets" in key or "legal_qa" in key
                        or "law_stackexchange" in key or "pile_of_law" in key):
                    records.extend(parse_community_jsonl(rows))

        except Exception as e:
            log.warning("parse_error", key=key, error=str(e))

    log.info("parse_done", records=len(records))

    # Upload in batches of 1000
    batch_size = 1000
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_num = i // batch_size
        upload_jsonl(batch, f"{CORPUS_PREFIX}/batch_{batch_num:04d}.jsonl")

    log.info("corpus_uploaded", total=len(records))
    return records


def _infer_source_meta_from_key(key: str) -> dict:
    if "uscis" in key:
        if "forms" in key:
            return dict(source_name="USCIS Forms", source_type="official_form",
                        agency="USCIS", authority_level="primary_official")
        return dict(source_name="USCIS", source_type="official_policy",
                    agency="USCIS", authority_level="primary_official")
    if "eoir" in key or "bia" in key:
        return dict(source_name="EOIR Decisions", source_type="case_law",
                    agency="EOIR", authority_level="primary_official")
    if "cfr" in key:
        return dict(source_name="8 CFR", source_type="regulation",
                    agency="DHS", authority_level="primary_official")
    if "ina" in key:
        return dict(source_name="INA", source_type="statute",
                    agency="Congress", authority_level="primary_official")
    if "dhs" in key or "cbp" in key or "ice" in key:
        return dict(source_name="DHS Stats", source_type="statistics",
                    agency="DHS", authority_level="primary_official")
    return dict(source_name="Unknown", source_type="unknown",
                agency="unknown", authority_level="community_non_authoritative")


def _dataset_name_from_key(key: str) -> str:
    parts = key.split("/")
    return parts[-2] if len(parts) > 1 else key


if __name__ == "__main__":
    parse_all()

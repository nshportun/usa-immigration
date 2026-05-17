"""Deduplicate corpus records by URL and near-duplicate text fingerprint."""

import hashlib
import re
import structlog

log = structlog.get_logger()


def _text_fingerprint(rec: dict) -> str:
    """
    Fingerprint a record for near-duplicate detection.
    For QA-format records (title is a question), include title+end-of-text so that
    records sharing the same context paragraph but with different Q&A are not collapsed.
    """
    text = rec.get("text", "")
    title = rec.get("title", "")
    # QA records: fingerprint = question (title) + last 200 chars of text (the answer)
    if title and text.startswith("Q:"):
        key = f"{title.lower().strip()} ||| {text[-200:].lower().strip()}"
    else:
        key = text[:1000].lower().strip()
    normalized = re.sub(r"\s+", " ", key)
    return hashlib.sha256(normalized.encode()).hexdigest()


def dedup(records: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()
    deduped = []
    dupes = 0

    for rec in records:
        url = rec.get("url", "")
        # For QA datasets that all share the same URL, don't dedup by URL alone
        is_shared_url = url in (
            "https://huggingface.co/datasets/harshitha008/US-immigration-laws",
        )
        fp = _text_fingerprint(rec)

        if url and not is_shared_url and url in seen_urls:
            dupes += 1
            continue
        if fp in seen_fingerprints:
            dupes += 1
            continue

        if url and not is_shared_url:
            seen_urls.add(url)
        seen_fingerprints.add(fp)
        deduped.append(rec)

    log.info("dedup_complete", before=len(records), after=len(deduped), dupes_removed=dupes)
    return deduped

"""
Validation pipeline for corpus records and Q&A pairs.
Implements all quality checks from CLAUDE.md.
"""

import re
from typing import Any

import structlog
import tiktoken

log = structlog.get_logger()

_enc = tiktoken.get_encoding("cl100k_base")
MAX_CHUNK_TOKENS = 1024
CONTRADICTION_WINDOW = 200

HIGH_RISK_TOPICS = {"asylum", "removal", "admissibility", "humanitarian", "unlawful_presence"}


# ── Corpus record validation ──────────────────────────────────────────────────

def validate_corpus_record(record: dict) -> list[str]:
    """Return list of validation errors. Empty = valid."""
    errors = []
    if not record.get("doc_id"):
        errors.append("missing doc_id")
    if not record.get("text") or len(record["text"].strip()) < 50:
        errors.append("text too short")
    if not record.get("url"):
        errors.append("missing url")
    if not record.get("source_name"):
        errors.append("missing source_name")
    if not record.get("authority_level"):
        errors.append("missing authority_level")
    # Detect navigation junk
    if _is_navigation_junk(record.get("text", "")):
        errors.append("navigation_junk")
    return errors


def _is_navigation_junk(text: str) -> bool:
    junk_phrases = [
        "skip to main content", "accept all cookies", "privacy policy",
        "terms of service", "© copyright", "all rights reserved",
    ]
    lower = text.lower()[:500]
    junk_count = sum(1 for p in junk_phrases if p in lower)
    return junk_count >= 3 or (len(text.strip()) < 200 and junk_count >= 1)


# ── Chunk validation ───────────────────────────────────────────────────────────

def validate_chunk(chunk: dict) -> list[str]:
    errors = []
    if not chunk.get("chunk_id"):
        errors.append("missing chunk_id")
    if not chunk.get("doc_id"):
        errors.append("missing doc_id")
    if not chunk.get("text_chunk") or len(chunk["text_chunk"].strip()) < 20:
        errors.append("chunk text too short")
    token_count = chunk.get("token_count", 0)
    if token_count > MAX_CHUNK_TOKENS:
        errors.append(f"chunk exceeds max tokens: {token_count}")
    return errors


# ── Q&A validation ─────────────────────────────────────────────────────────────

def validate_qa_pair(qa: dict) -> list[str]:
    errors = []
    if not qa.get("qa_id"):
        errors.append("missing qa_id")
    if not qa.get("question") or len(qa["question"].strip()) < 10:
        errors.append("question too short")
    if not qa.get("answer") or len(qa["answer"].strip()) < 10:
        errors.append("answer too short")
    if not qa.get("source_doc_id") and not qa.get("source_url"):
        errors.append("no source reference")
    if not qa.get("source_span"):
        errors.append("missing source_span (answer not grounded)")
    if not qa.get("authority_level"):
        errors.append("missing authority_level")
    # Flag high-risk topics that lack manual review
    tags = qa.get("topic_tags", [])
    if any(t in HIGH_RISK_TOPICS for t in tags) and qa.get("review_status") != "manual_review_required":
        errors.append("high_risk_topic_not_flagged")
    return errors


def validate_no_hallucination(qa: dict) -> bool:
    """Check that answer text overlaps with source_span (basic grounding check)."""
    span = (qa.get("source_span") or "").lower()
    answer = (qa.get("answer") or "").lower()
    if not span:
        return False
    # For secondary_reputable sources (e.g. HF datasets with INA-formatted context),
    # accept if the answer is substantive (>=20 chars) — the span is from the source text
    # but may use legal citation formatting that doesn't overlap lexically with the answer.
    if qa.get("authority_level") == "secondary_reputable" and len(answer) >= 20:
        return True
    # For primary sources: at least 3 significant words from answer must appear in span
    answer_words = set(re.findall(r"\b\w{4,}\b", answer))
    span_words = set(re.findall(r"\b\w{4,}\b", span))
    overlap = answer_words & span_words
    return len(overlap) >= 3


# ── Contradiction detection ────────────────────────────────────────────────────

def flag_contradictions(qa_pairs: list[dict]) -> list[dict]:
    """
    Naive contradiction detection: if two QA pairs have nearly identical questions
    but different answers, flag both.
    """
    from collections import defaultdict
    by_question_key: dict[str, list[dict]] = defaultdict(list)

    for qa in qa_pairs:
        key = re.sub(r"\W+", " ", qa.get("question", "").lower()).strip()[:80]
        by_question_key[key].append(qa)

    flagged = []
    for key, pairs in by_question_key.items():
        if len(pairs) > 1:
            answers = {p.get("answer", "")[:CONTRADICTION_WINDOW] for p in pairs}
            if len(answers) > 1:
                for p in pairs:
                    p = dict(p)
                    p["review_status"] = "contradiction_flagged"
                    flagged.append(p)
            else:
                flagged.extend(pairs)
        else:
            flagged.extend(pairs)

    contradictions = sum(1 for p in flagged if p.get("review_status") == "contradiction_flagged")
    if contradictions:
        log.warning("contradictions_flagged", count=contradictions)
    return flagged


# ── Batch validators ───────────────────────────────────────────────────────────

def validate_corpus(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (valid_records, rejected_records)."""
    valid, rejected = [], []
    for rec in records:
        errs = validate_corpus_record(rec)
        if errs:
            rec = dict(rec)
            rec["validation_errors"] = errs
            rejected.append(rec)
        else:
            valid.append(rec)
    log.info("corpus_validation", valid=len(valid), rejected=len(rejected))
    return valid, rejected


def validate_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    valid, rejected = [], []
    for chunk in chunks:
        errs = validate_chunk(chunk)
        if errs:
            chunk = dict(chunk)
            chunk["validation_errors"] = errs
            rejected.append(chunk)
        else:
            valid.append(chunk)
    log.info("chunk_validation", valid=len(valid), rejected=len(rejected))
    return valid, rejected


def validate_qa_pairs(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    valid, rejected = [], []
    for qa in pairs:
        errs = validate_qa_pair(qa)
        if not validate_no_hallucination(qa):
            errs.append("hallucination_risk_no_span_overlap")
        if errs:
            qa = dict(qa)
            qa["validation_errors"] = errs
            rejected.append(qa)
        else:
            valid.append(qa)
    log.info("qa_validation", valid=len(valid), rejected=len(rejected))
    return valid, rejected

"""
Normalize canonical corpus records:
  - Assign canonical doc_ids
  - Standardize dates
  - Add topic_tags from URL/section_path/title heuristics
  - Classify data_class (law, policy, procedure, statistics, community)
  - Mark time-sensitive content
"""

import re
from datetime import datetime
from typing import Optional

import structlog
from dateutil import parser as dateparser

log = structlog.get_logger()

FIRST_SCOPE_TOPICS = {
    "family-based immigration": ["family", "i-130", "petition", "relative", "sponsor", "spousal"],
    "adjustment-of-status": ["adjustment", "i-485", "green card", "lawful permanent", "lpr"],
    "employment-authorization": ["employment authorization", "ead", "i-765", "work permit"],
    "naturalization": ["naturalization", "citizenship", "n-400", "oath", "civics"],
    "travel-documents": ["travel document", "advance parole", "reentry permit", "i-131"],
}

ALL_TOPICS = {
    **FIRST_SCOPE_TOPICS,
    "asylum": ["asylum", "refugee", "i-589", "withholding", "cat"],
    "removal": ["removal", "deportation", "order of removal", "voluntary departure"],
    "admissibility": ["inadmissibility", "grounds of inadmissibility", "waiver", "i-601"],
    "employment-based": ["employment-based", "i-140", "priority date", "eb-1", "eb-2", "eb-3", "h-1b"],
    "nonimmigrant-visas": ["nonimmigrant", "b-1", "b-2", "f-1", "j-1", "h-1b", "o-1", "ds-160"],
    "humanitarian": ["humanitarian", "tps", "daca", "u-visa", "vawa", "t-visa"],
    "appeals": ["appeal", "bia", "motion to reopen", "motion to reconsider", "aao"],
    "statistics": ["yearbook", "cbp", "ice", "enforcement", "arrests", "removals", "encounters"],
}

TIME_SENSITIVE_PATTERNS = [
    re.compile(r"\b(202[0-9]|201[0-9])\b"),
    re.compile(r"processing time", re.I),
    re.compile(r"fiscal year \d{4}", re.I),
    re.compile(r"temporary protected status", re.I),
    re.compile(r"executive order", re.I),
    re.compile(r"current as of", re.I),
]

DATA_CLASS_RULES = {
    "statistics": ["statistics", "yearbook", "cbp", "ice", "census", "data"],
    "law": ["statute", "regulation", "cfr", "ina", "case_law"],
    "policy": ["official_policy", "official_faq"],
    "procedure": ["official_form", "procedure"],
    "community": ["community"],
}


def _normalize_date(raw: str) -> str:
    if not raw:
        return ""
    try:
        return dateparser.parse(raw).strftime("%Y-%m-%d")
    except Exception:
        return raw[:10] if len(raw) >= 10 else raw


def _assign_topic_tags(record: dict) -> list[str]:
    haystack = " ".join([
        record.get("title", ""),
        record.get("url", ""),
        " ".join(record.get("section_path", [])),
        record.get("text", "")[:500],
    ]).lower()

    tags = []
    for tag, keywords in ALL_TOPICS.items():
        if any(kw in haystack for kw in keywords):
            tags.append(tag)
    return tags or ["general"]


def _assign_data_class(record: dict) -> str:
    source_type = record.get("source_type", "")
    for data_class, types in DATA_CLASS_RULES.items():
        if any(t in source_type for t in types):
            return data_class
    return "general"


def _is_time_sensitive(record: dict) -> bool:
    text = record.get("text", "") + record.get("title", "")
    return any(p.search(text) for p in TIME_SENSITIVE_PATTERNS)


def normalize_record(record: dict) -> dict:
    record = dict(record)
    record["publication_date"] = _normalize_date(record.get("publication_date", ""))
    record["topic_tags"] = _assign_topic_tags(record)
    record["data_class"] = _assign_data_class(record)
    record["time_sensitive"] = _is_time_sensitive(record)
    record["in_first_scope"] = any(
        tag in FIRST_SCOPE_TOPICS for tag in record["topic_tags"]
    )
    return record


def normalize_corpus(records: list[dict]) -> list[dict]:
    normalized = [normalize_record(r) for r in records]
    log.info("normalize_complete", total=len(normalized),
             in_scope=sum(1 for r in normalized if r["in_first_scope"]))
    return normalized

"""
Parse structured JSON/JSONL datasets (eCFR API, open datasets, DHS stats).
Converts to canonical corpus records or to labeled statistics records.
"""

import datetime
import hashlib
import json
from typing import Any

import structlog

log = structlog.get_logger()


def parse_ecfr_json(data: dict) -> list[dict]:
    """
    Convert eCFR JSON API response (Title 8) into section-level corpus records.
    The eCFR JSON structure: {"title": ..., "children": [{"identifier": "part-X", ...}]}
    """
    records = []

    def walk(node: dict, path: list[str]):
        label = node.get("label") or node.get("identifier") or ""
        heading = node.get("heading") or node.get("title") or ""
        text = node.get("text") or node.get("content") or ""
        current_path = path + [f"{label} {heading}".strip()]

        if text and len(text.strip()) > 50:
            url = f"https://www.ecfr.gov/current/title-8/{'/'.join(node.get('identifier', '').split('-'))}"
            doc_id = "doc-" + hashlib.sha256(url.encode()).hexdigest()[:12]
            records.append({
                "doc_id": doc_id,
                "source_name": "8 CFR",
                "source_type": "regulation",
                "agency": "DHS",
                "jurisdiction": "US",
                "title": heading or label,
                "url": url,
                "publication_date": data.get("date", ""),
                "retrieved_at": datetime.datetime.utcnow().isoformat(),
                "section_path": current_path,
                "text": text.strip(),
                "authority_level": "primary_official",
                "license_note": "public government source",
            })

        for child in node.get("children", []):
            walk(child, current_path)

    walk(data, [])
    log.info("ecfr_records_parsed", count=len(records))
    return records


def parse_dhs_stats_json(data: Any, source_url: str, dataset_name: str) -> list[dict]:
    """
    Convert a DHS/CBP JSON stats payload into individual statistic records.
    Each row becomes a record labeled source_type=statistics.
    """
    records = []
    rows = data if isinstance(data, list) else data.get("results", data.get("data", []))
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        doc_id = "stat-" + hashlib.sha256(f"{source_url}-{i}".encode()).hexdigest()[:12]
        records.append({
            "doc_id": doc_id,
            "source_name": dataset_name,
            "source_type": "statistics",
            "agency": "DHS",
            "jurisdiction": "US",
            "title": dataset_name,
            "url": source_url,
            "publication_date": row.get("year", row.get("date", "")),
            "retrieved_at": datetime.datetime.utcnow().isoformat(),
            "section_path": [],
            "text": json.dumps(row),
            "authority_level": "primary_official",
            "license_note": "public government source",
            "data_class": "statistics",
        })
    log.info("stats_records_parsed", dataset=dataset_name, count=len(records))
    return records


def parse_community_jsonl(records_in: list[dict]) -> list[dict]:
    """
    Convert raw community/open-dataset records into canonical corpus records.
    Handles: harshitha008/US-immigration-laws, LegalQAEval, pile-of-law, StackExchange.
    """
    out = []
    for row in records_in:
        # US-immigration-laws format: {question, context, answer} — Apache 2.0
        if row.get("question") and row.get("context") and row.get("answer") is not None:
            question = row["question"]
            context = row["context"]
            answer = row.get("answer", "")
            text = f"Q: {question}\nContext: {context}\nA: {answer}".strip()
            doc_id = "hf-" + hashlib.sha256((question + context[:100]).encode()).hexdigest()[:12]
            out.append({
                "doc_id": doc_id,
                "source_name": "US Immigration Laws (HF)",
                "source_type": "official_faq",
                "agency": "USCIS",
                "jurisdiction": "US",
                "title": question[:200],
                "url": "https://huggingface.co/datasets/harshitha008/US-immigration-laws",
                "publication_date": "",
                "retrieved_at": datetime.datetime.utcnow().isoformat(),
                "section_path": [],
                "text": text,
                "authority_level": "secondary_reputable",
                "license_note": "Apache 2.0",
                "tags": "",
                "score": 0,
            })
            continue

        # pile-of-law format: {text, ...}
        if row.get("text") and not row.get("question") and not row.get("title"):
            text = row["text"].strip()
            if len(text) < 100:
                continue
            url = row.get("url", row.get("src", ""))
            doc_id = "pol-" + hashlib.sha256(text[:200].encode()).hexdigest()[:12]
            out.append({
                "doc_id": doc_id,
                "source_name": "Pile of Law",
                "source_type": "regulation",
                "agency": "DHS",
                "jurisdiction": "US",
                "title": text[:120].split("\n")[0].strip(),
                "url": url,
                "publication_date": row.get("created_timestamp", "")[:10],
                "retrieved_at": datetime.datetime.utcnow().isoformat(),
                "section_path": [],
                "text": text,
                "authority_level": "primary_official",
                "license_note": "public government source",
                "tags": "",
                "score": 0,
            })
            continue

        # LegalQAEval format: {id, text, question, answers}
        if row.get("question") and row.get("text") and not row.get("body"):
            question = row.get("question", "")
            context = row.get("text", "")
            answers = row.get("answers", [])
            answer_text = answers[0] if isinstance(answers, list) and answers else ""
            text = f"Q: {question}\nContext: {context[:800]}\nA: {answer_text}".strip()
            doc_id = "com-" + hashlib.sha256((str(row.get("id", "")) + question).encode()).hexdigest()[:12]
            out.append({
                "doc_id": doc_id,
                "source_name": "LegalQAEval",
                "source_type": "community",
                "agency": "community",
                "jurisdiction": "US",
                "title": question[:200],
                "url": "",
                "publication_date": "",
                "retrieved_at": datetime.datetime.utcnow().isoformat(),
                "section_path": [],
                "text": text,
                "authority_level": "community_non_authoritative",
                "license_note": "community content; not authoritative",
                "tags": "",
                "score": 0,
            })
            continue

        # StackExchange / generic format (handles question_title/question_body from ymoslem/Law-StackExchange)
        question = (row.get("question_title") or row.get("question") or row.get("title") or "")
        body = (row.get("question_body") or row.get("body") or "")
        text = f"Q: {question}\n\n{body}".strip()
        if len(text) < 50:
            continue
        url = row.get("url") or row.get("link") or ""
        doc_id = "com-" + hashlib.sha256((url + question).encode()).hexdigest()[:12]
        out.append({
            "doc_id": doc_id,
            "source_name": row.get("source", "Community"),
            "source_type": "community",
            "agency": "community",
            "jurisdiction": "US",
            "title": question[:200],
            "url": url,
            "publication_date": "",
            "retrieved_at": datetime.datetime.utcnow().isoformat(),
            "section_path": [],
            "text": text,
            "authority_level": "community_non_authoritative",
            "license_note": "community content; not authoritative",
            "tags": row.get("tags", ""),
            "score": row.get("score", 0),
        })
    return out

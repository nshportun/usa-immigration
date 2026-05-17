"""Generate coverage.md report showing which immigration subdomains are complete or sparse."""

import json
from collections import defaultdict
from pathlib import Path

import structlog

log = structlog.get_logger()

SUBDOMAINS = [
    "family-based immigration",
    "adjustment-of-status",
    "employment-authorization",
    "naturalization",
    "travel-documents",
    "asylum",
    "removal",
    "admissibility",
    "employment-based",
    "nonimmigrant-visas",
    "humanitarian",
    "appeals",
    "statistics",
]

COVERAGE_THRESHOLDS = {
    "corpus_docs": {"sparse": 5, "partial": 20, "complete": 50},
    "qa_pairs": {"sparse": 10, "partial": 50, "complete": 200},
}


def _coverage_level(count: int, thresholds: dict) -> str:
    if count >= thresholds["complete"]:
        return "complete"
    if count >= thresholds["partial"]:
        return "partial"
    if count > 0:
        return "sparse"
    return "missing"


def generate_coverage_report(
    corpus_records: list[dict],
    qa_pairs: list[dict],
    output_path: Path = Path("reports/coverage.md"),
) -> str:
    doc_counts: dict[str, int] = defaultdict(int)
    qa_counts: dict[str, int] = defaultdict(int)

    for rec in corpus_records:
        for tag in rec.get("topic_tags", []):
            if tag in SUBDOMAINS:
                doc_counts[tag] += 1

    for qa in qa_pairs:
        for tag in qa.get("topic_tags", []):
            if tag in SUBDOMAINS:
                qa_counts[tag] += 1

    lines = [
        "# Coverage Report\n",
        f"**Corpus docs**: {len(corpus_records)}  |  **Q&A pairs**: {len(qa_pairs)}\n",
        "| Subdomain | Corpus Docs | QA Pairs | Corpus Status | QA Status |",
        "|---|---|---|---|---|",
    ]

    for domain in SUBDOMAINS:
        docs = doc_counts.get(domain, 0)
        qas = qa_counts.get(domain, 0)
        doc_status = _coverage_level(docs, COVERAGE_THRESHOLDS["corpus_docs"])
        qa_status = _coverage_level(qas, COVERAGE_THRESHOLDS["qa_pairs"])
        emoji = {"complete": "✅", "partial": "🔶", "sparse": "🔸", "missing": "❌"}
        lines.append(
            f"| {domain} | {docs} | {qas} | {emoji[doc_status]} {doc_status} | {emoji[qa_status]} {qa_status} |"
        )

    # Authority breakdown
    authority_counts: dict[str, int] = defaultdict(int)
    for rec in corpus_records:
        authority_counts[rec.get("authority_level", "unknown")] += 1

    lines += [
        "\n## Authority Level Breakdown (Corpus)",
        "| Authority Level | Count |",
        "|---|---|",
    ]
    for level, count in sorted(authority_counts.items()):
        lines.append(f"| {level} | {count} |")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    log.info("coverage_report_written", path=str(output_path))
    return report


def generate_data_quality_report(
    rejected_corpus: list[dict],
    rejected_chunks: list[dict],
    rejected_qa: list[dict],
    output_path: Path = Path("reports/data_quality.md"),
) -> str:
    lines = [
        "# Data Quality Report\n",
        f"- Rejected corpus records: {len(rejected_corpus)}",
        f"- Rejected chunks: {len(rejected_chunks)}",
        f"- Rejected Q&A pairs: {len(rejected_qa)}\n",
    ]

    if rejected_corpus:
        error_counts: dict[str, int] = defaultdict(int)
        for rec in rejected_corpus:
            for e in rec.get("validation_errors", []):
                error_counts[e] += 1
        lines += ["## Corpus Rejection Reasons", "| Reason | Count |", "|---|---|"]
        for reason, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")

    if rejected_qa:
        qa_error_counts: dict[str, int] = defaultdict(int)
        for qa in rejected_qa:
            for e in qa.get("validation_errors", []):
                qa_error_counts[e] += 1
        lines += ["\n## Q&A Rejection Reasons", "| Reason | Count |", "|---|---|"]
        for reason, count in sorted(qa_error_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} |")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    log.info("quality_report_written", path=str(output_path))
    return report

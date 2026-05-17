"""
Chunk canonical corpus records into retrieval units for RAG.
Target: ~512 tokens, overlap: ~64 tokens, max: 1024 tokens.
Chunk boundaries prefer heading lines (all-caps or markdown #) when present.
"""

import hashlib
import os
import re
from typing import Iterator

import structlog
import tiktoken

log = structlog.get_logger()

TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "512"))
OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "1024"))

_enc = tiktoken.get_encoding("cl100k_base")

HEADING_RE = re.compile(r"^(#{1,4}\s+.+|[A-Z][A-Z\s\d:]{10,})$", re.MULTILINE)


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) pairs."""
    parts = HEADING_RE.split(text)
    if len(parts) <= 1:
        return [("", text)]
    result = []
    # parts alternates: [pre, heading1, body1, heading2, body2, ...]
    result.append(("", parts[0]))
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        result.append((heading, body))
    return result


def _chunk_text(text: str) -> Iterator[str]:
    """Yield overlapping token-bounded chunks."""
    tokens = _enc.encode(text)
    step = TARGET_TOKENS - OVERLAP_TOKENS
    start = 0
    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        yield _enc.decode(tokens[start:end])
        if end >= len(tokens):
            break
        start += step


def chunk_record(record: dict) -> list[dict]:
    """Split a canonical record into chunk records."""
    text = record.get("text", "")
    if not text:
        return []

    sections = _split_by_headings(text)
    chunks = []
    chunk_index = 0

    for heading, body in sections:
        context = f"{heading}\n\n{body}".strip() if heading else body.strip()
        if not context:
            continue

        if _token_count(context) <= MAX_TOKENS:
            segments = [context]
        else:
            segments = list(_chunk_text(context))

        for segment in segments:
            if not segment.strip():
                continue
            chunk_id = (
                record["doc_id"]
                + "-"
                + hashlib.sha256(segment.encode()).hexdigest()[:8]
            )
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": record["doc_id"],
                "chunk_index": chunk_index,
                "text_chunk": segment,
                "token_count": _token_count(segment),
                "heading_context": heading,
                "section_path": record.get("section_path", []),
                "source_name": record.get("source_name", ""),
                "source_type": record.get("source_type", ""),
                "authority_level": record.get("authority_level", ""),
                "topic_tags": record.get("topic_tags", []),
                "url": record.get("url", ""),
                "publication_date": record.get("publication_date", ""),
                "time_sensitive": record.get("time_sensitive", False),
                "data_class": record.get("data_class", "general"),
            })
            chunk_index += 1

    return chunks


def chunk_corpus(records: list[dict]) -> list[dict]:
    all_chunks = []
    for rec in records:
        all_chunks.extend(chunk_record(rec))
    log.info("chunking_complete", docs=len(records), chunks=len(all_chunks))
    return all_chunks

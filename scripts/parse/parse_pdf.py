"""Parse PDF bytes into canonical corpus records using pdfplumber."""

import datetime
import hashlib
import io
import re
from typing import Optional

import pdfplumber
import structlog

log = structlog.get_logger()


def parse_pdf(pdf_bytes: bytes, url: str, source_meta: dict, retrieved_at: str = "") -> Optional[dict]:
    """
    Extract text from PDF bytes.
    source_meta must contain: source_name, source_type, agency, authority_level.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
            full_text = "\n\n".join(p for p in pages_text if p.strip())
    except Exception as e:
        log.warning("pdf_parse_error", url=url, error=str(e))
        return None

    if not full_text or len(full_text.strip()) < 100:
        return None

    # Clean up whitespace artifacts common in PDFs
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    doc_id = "doc-" + hashlib.sha256(url.encode()).hexdigest()[:12]

    # Try to extract title from first non-empty line
    first_line = next((l.strip() for l in full_text.split("\n") if l.strip()), "")
    title = first_line[:200] if first_line else url.split("/")[-1]

    return {
        "doc_id": doc_id,
        **source_meta,
        "jurisdiction": "US",
        "title": title,
        "url": url,
        "publication_date": "",
        "retrieved_at": retrieved_at or datetime.datetime.utcnow().isoformat(),
        "section_path": [],
        "text": full_text,
        "license_note": (
            "public government source"
            if source_meta.get("authority_level") == "primary_official"
            else "see source"
        ),
        "format": "pdf",
    }

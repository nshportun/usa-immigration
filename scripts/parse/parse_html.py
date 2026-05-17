"""
Parse raw HTML from S3 into canonical corpus records.
Handles USCIS Policy Manual, FAQs, statutes, EOIR pages.
"""

import datetime
import hashlib
import re
from typing import Optional
from urllib.parse import urlparse

import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger()

JUNK_PATTERNS = [
    re.compile(r"skip to (main )?content", re.I),
    re.compile(r"javascript is (required|disabled)", re.I),
    re.compile(r"^\s*$"),
]

# Only checked against first 200 chars — short pages with no real content
_LEDE_JUNK_PATTERNS = [
    re.compile(r"(cookie|privacy) (policy|notice|banner)", re.I),
]


def _is_junk(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 20:
        return True
    # Full-text checks
    for p in JUNK_PATTERNS:
        if p.search(stripped):
            return True
    # Lede-only checks (privacy/cookie notices in footer don't disqualify a page)
    for p in _LEDE_JUNK_PATTERNS:
        if p.search(stripped[:200]):
            return True
    return False


def _extract_date(soup: BeautifulSoup, url: str) -> str:
    """Try multiple heuristics to extract a publication or last-updated date."""
    # meta tags
    for name in ["article:modified_time", "article:published_time", "dc.date", "last-modified"]:
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            return tag["content"][:10]
    # visible "last updated" text
    for candidate in soup.find_all(string=re.compile(r"last (updated|revised|modified)", re.I)):
        match = re.search(r"(\d{4}-\d{2}-\d{2}|\w+ \d+, \d{4})", str(candidate))
        if match:
            return match.group(1)
    return ""


def _extract_section_path(soup: BeautifulSoup, url: str) -> list[str]:
    """Build a breadcrumb/section path from the page structure."""
    breadcrumb = soup.find(class_=re.compile(r"breadcrumb", re.I))
    if breadcrumb:
        parts = [a.get_text(strip=True) for a in breadcrumb.find_all("a")]
        parts += [li.get_text(strip=True) for li in breadcrumb.find_all("li") if not li.find("a")]
        return [p for p in parts if p]
    # Fall back to URL path segments
    path = urlparse(url).path.strip("/").split("/")
    return [seg.replace("-", " ").title() for seg in path if seg]


def _classify_source(url: str) -> dict:
    """Return source_name, source_type, agency, authority_level from the URL."""
    if "uscis.gov" in url:
        if "policy-manual" in url:
            return dict(source_name="USCIS Policy Manual", source_type="official_policy",
                        agency="USCIS", authority_level="primary_official")
        if "/forms/" in url:
            return dict(source_name="USCIS Forms", source_type="official_form",
                        agency="USCIS", authority_level="primary_official")
        return dict(source_name="USCIS", source_type="official_faq",
                    agency="USCIS", authority_level="primary_official")
    if "my.uscis.gov" in url:
        return dict(source_name="myUSCIS Help", source_type="official_faq",
                    agency="USCIS", authority_level="primary_official")
    if "ecfr.gov" in url or "cfr" in url.lower():
        return dict(source_name="8 CFR", source_type="regulation",
                    agency="DHS", authority_level="primary_official")
    if "law.cornell.edu" in url:
        return dict(source_name="INA (LII)", source_type="statute",
                    agency="Congress", authority_level="primary_official")
    if "justice.gov/eoir" in url:
        if "bia" in url.lower():
            return dict(source_name="BIA Precedent Decisions", source_type="case_law",
                        agency="EOIR", authority_level="primary_official")
        if "ag-decision" in url.lower():
            return dict(source_name="AG Decisions", source_type="case_law",
                        agency="DOJ", authority_level="primary_official")
        return dict(source_name="EOIR", source_type="case_law",
                    agency="EOIR", authority_level="primary_official")
    if "dhs.gov" in url:
        return dict(source_name="DHS", source_type="statistics",
                    agency="DHS", authority_level="primary_official")
    if "cbp.gov" in url:
        return dict(source_name="CBP", source_type="statistics",
                    agency="CBP", authority_level="primary_official")
    if "ice.gov" in url:
        return dict(source_name="ICE", source_type="statistics",
                    agency="ICE", authority_level="primary_official")
    if "nilc.org" in url or "ilrc.org" in url or "trac.syr.edu" in url:
        return dict(source_name="Nonprofit Legal Guide", source_type="secondary",
                    agency="nonprofit", authority_level="secondary_reputable")
    if "visajourney.com" in url or "stackexchange.com" in url:
        return dict(source_name="Community Forum", source_type="community",
                    agency="community", authority_level="community_non_authoritative")
    return dict(source_name="Unknown", source_type="unknown",
                agency="unknown", authority_level="community_non_authoritative")


def parse_html(raw_html: str, url: str, retrieved_at: str = "") -> Optional[dict]:
    """
    Parse raw HTML into a canonical corpus record.
    Returns None if the page yields no useful content.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove nav, footer, scripts, styles
    for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Prefer main content area
    main = (
        soup.find("main")
        or soup.find(id="main-content")
        or soup.find(class_=re.compile(r"(content|article|body)", re.I))
        or soup.body
    )
    if not main:
        return None

    text = main.get_text(separator="\n").strip()
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    if _is_junk(text) or len(text) < 100:
        return None

    source_meta = _classify_source(url)
    doc_id = "doc-" + hashlib.sha256(url.encode()).hexdigest()[:12]

    return {
        "doc_id": doc_id,
        **source_meta,
        "jurisdiction": "US",
        "title": title,
        "url": url,
        "publication_date": _extract_date(soup, url),
        "retrieved_at": retrieved_at or datetime.datetime.utcnow().isoformat(),
        "section_path": _extract_section_path(soup, url),
        "text": text,
        "license_note": "public government source" if source_meta["authority_level"] == "primary_official" else "see source",
    }

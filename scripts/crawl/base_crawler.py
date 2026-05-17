"""Base HTTP crawler with retry, rate-limiting, dedup, and S3 raw-save."""

import hashlib
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from scripts.aws_config import S3_BUCKET
from scripts.s3_store import upload_text

log = structlog.get_logger()

DELAY = float(os.getenv("CRAWL_DELAY_SECONDS", "1.5"))
TIMEOUT = float(os.getenv("CRAWL_TIMEOUT_SECONDS", "30"))
MAX_RETRIES = int(os.getenv("CRAWL_MAX_RETRIES", "3"))
USER_AGENT = os.getenv(
    "CRAWL_USER_AGENT",
    "USAImmigrationDatasetBot/1.0 (research; non-commercial)",
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def url_to_key(url: str) -> str:
    """Deterministic S3 key fragment from URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


class BaseCrawler:
    source_name: str = "base"
    source_type: str = "unknown"
    authority_level: str = "primary_official"

    def __init__(self, crawl_log_path: Path):
        self.crawl_log_path = crawl_log_path
        self.crawl_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._seen_urls: set[str] = self._load_seen()
        self._client = httpx.Client(
            headers=HEADERS,
            timeout=TIMEOUT,
            follow_redirects=True,
        )

    def _load_seen(self) -> set[str]:
        if not self.crawl_log_path.exists():
            return set()
        seen = set()
        with open(self.crawl_log_path) as f:
            import json
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("status") == "ok":
                            seen.add(rec["url"])
                    except Exception:
                        pass
        return seen

    def _log_result(self, url: str, status: str, s3_key: str = "", error: str = ""):
        import json
        import datetime
        rec = {
            "url": url,
            "status": status,
            "s3_key": s3_key,
            "error": error,
            "ts": datetime.datetime.utcnow().isoformat(),
        }
        with open(self.crawl_log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=2, max=30))
    def _get(self, url: str) -> httpx.Response:
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp

    def fetch(self, url: str, raw_s3_prefix: str) -> Optional[str]:
        """
        Fetch URL, save raw HTML/text to S3, return text content.
        Returns None if already crawled or on error.
        """
        if url in self._seen_urls:
            log.debug("skip_seen", url=url)
            return None

        time.sleep(DELAY)
        try:
            resp = self._get(url)
        except Exception as e:
            log.warning("fetch_error", url=url, error=str(e))
            self._log_result(url, "error", error=str(e))
            return None

        content_type = resp.headers.get("content-type", "")
        text = resp.text
        fragment = url_to_key(url)
        ext = ".html" if "html" in content_type else ".txt"
        s3_path = f"{raw_s3_prefix}/{fragment}{ext}"

        try:
            key = upload_text(text, s3_path, content_type=content_type)
        except Exception as e:
            log.warning("s3_upload_error", url=url, error=str(e))
            self._log_result(url, "s3_error", error=str(e))
            return None

        self._seen_urls.add(url)
        self._log_result(url, "ok", s3_key=key)
        log.info("crawled", url=url, s3_key=key, bytes=len(text))
        return text

    def fetch_pdf(self, url: str, raw_s3_prefix: str) -> Optional[bytes]:
        """Fetch a PDF and store raw bytes on S3."""
        if url in self._seen_urls:
            return None
        time.sleep(DELAY)
        try:
            resp = self._get(url)
        except Exception as e:
            log.warning("fetch_pdf_error", url=url, error=str(e))
            self._log_result(url, "error", error=str(e))
            return None

        fragment = url_to_key(url)
        s3_path = f"{raw_s3_prefix}/{fragment}.pdf"
        try:
            from scripts.aws_config import s3_client
            s3_client().put_object(
                Bucket=S3_BUCKET,
                Key=f"v1/{s3_path}",
                Body=resp.content,
                ContentType="application/pdf",
            )
        except Exception as e:
            log.warning("s3_pdf_upload_error", url=url, error=str(e))
            self._log_result(url, "s3_error", error=str(e))
            return None

        self._seen_urls.add(url)
        self._log_result(url, "ok", s3_key=s3_path)
        return resp.content

"""
Crawl INA (house.gov) and Title 8 CFR (eCFR dated API + specific parts).
"""

import datetime
import re
from pathlib import Path
from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup

from scripts.crawl.base_crawler import BaseCrawler

log = structlog.get_logger()

INA_ROOT = "https://uscode.house.gov/view.xhtml?path=/prelim@title8&edition=prelim"
INA_SECTION_PATTERN = re.compile(r"path=/prelim@title8/chapter\d")

ECFR_HTML = "https://www.ecfr.gov/current/title-8"
ECFR_API = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-8.json"

# Priority CFR parts per CLAUDE.md
CFR_PARTS = [
    103, 204, 205, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
    245, 248, 249, 274, 316, 319, 322, 324,
]

RAW_INA = "data_raw/statutes"
RAW_CFR = "data_raw/cfr"
LOG_PATH = Path("metadata/crawl_log.jsonl")


class StatutesCrawler(BaseCrawler):
    source_name = "INA_8CFR"
    source_type = "statute_regulation"
    authority_level = "primary_official"

    def crawl_ina(self):
        log.info("crawl_ina_start")
        html = self.fetch(INA_ROOT, RAW_INA)
        if not html:
            return
        soup = BeautifulSoup(html, "lxml")
        section_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "title8" in href and ("chapter" in href or "section" in href):
                full = urljoin("https://uscode.house.gov", href)
                section_links.add(full)
        log.info("ina_links_found", count=len(section_links))
        for url in sorted(section_links):
            self.fetch(url, RAW_INA)

    def crawl_cfr_api(self):
        today = datetime.date.today().isoformat()
        api_url = ECFR_API.format(date=today)
        log.info("crawl_cfr_api", url=api_url)
        self.fetch(api_url, RAW_CFR)

    def crawl_cfr_parts(self):
        log.info("crawl_cfr_parts_start", parts=len(CFR_PARTS))
        # Top-level HTML for discovery
        html = self.fetch(ECFR_HTML, RAW_CFR)
        # Fetch each priority part directly
        for part in CFR_PARTS:
            url = f"https://www.ecfr.gov/current/title-8/chapter-I/part-{part}"
            self.fetch(url, f"{RAW_CFR}/parts")
            # Also try alternate chapter patterns
            alt = f"https://www.ecfr.gov/current/title-8/part-{part}"
            self.fetch(alt, f"{RAW_CFR}/parts")

    def run(self):
        self.crawl_ina()
        self.crawl_cfr_api()
        self.crawl_cfr_parts()
        log.info("statutes_crawl_complete")


if __name__ == "__main__":
    StatutesCrawler(LOG_PATH).run()

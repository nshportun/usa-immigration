"""
Crawl secondary / nonprofit / research sources:
  - TRAC immigration data
  - NILC practice explainers
  - ILRC advisories
  - Deportation Data Project
  - University library guides (discovery only)
"""

import re
from pathlib import Path
from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup

from scripts.crawl.base_crawler import BaseCrawler

log = structlog.get_logger()

TRAC_ROOT = "https://trac.syr.edu/immigration"
NILC_ROOT = "https://www.nilc.org/issues/immigration-policy/"
ILRC_ROOT = "https://www.ilrc.org/publications"
DEPORTATION_DATA = "https://deportationdata.org/"
UNIV_GUIDES = [
    "https://guides.library.cornell.edu/immigration",
    "https://libguides.law.ucla.edu/immigration",
]

RAW_PREFIX = "data_raw/open_datasets"
LOG_PATH = Path("metadata/crawl_log.jsonl")


class SecondaryCrawler(BaseCrawler):
    source_name = "Secondary_Nonprofit"
    source_type = "secondary_reputable"
    authority_level = "secondary_reputable"

    def crawl_trac(self):
        log.info("crawl_trac_start")
        html = self.fetch(TRAC_ROOT, f"{RAW_PREFIX}/trac")
        if not html:
            return
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "trac.syr.edu/immigration" in href or href.startswith("/immigration"):
                self.fetch(urljoin(TRAC_ROOT, href), f"{RAW_PREFIX}/trac")

    def crawl_nilc(self):
        log.info("crawl_nilc_start")
        html = self.fetch(NILC_ROOT, f"{RAW_PREFIX}/nilc")
        if not html:
            return
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "nilc.org" in href and re.search(r"\.(html?|pdf)$", href, re.I):
                if re.search(r"\.pdf$", href, re.I):
                    self.fetch_pdf(href, f"{RAW_PREFIX}/nilc/pdf")
                else:
                    self.fetch(href, f"{RAW_PREFIX}/nilc")

    def crawl_ilrc(self):
        log.info("crawl_ilrc_start")
        html = self.fetch(ILRC_ROOT, f"{RAW_PREFIX}/ilrc")
        if not html:
            return
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"\.pdf$", href, re.I) and "ilrc.org" in href:
                self.fetch_pdf(href, f"{RAW_PREFIX}/ilrc/pdf")

    def crawl_deportation_data(self):
        log.info("crawl_deportation_data_start")
        html = self.fetch(DEPORTATION_DATA, f"{RAW_PREFIX}/deportation_data")
        if not html:
            return
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"\.(csv|json|xlsx?)$", href, re.I):
                self.fetch(urljoin(DEPORTATION_DATA, href), f"{RAW_PREFIX}/deportation_data/files")

    def crawl_univ_guides(self):
        log.info("crawl_univ_guides_start")
        for url in UNIV_GUIDES:
            self.fetch(url, f"{RAW_PREFIX}/univ_guides")

    def run(self):
        self.crawl_trac()
        self.crawl_nilc()
        self.crawl_ilrc()
        self.crawl_deportation_data()
        self.crawl_univ_guides()
        log.info("secondary_crawl_complete")


if __name__ == "__main__":
    SecondaryCrawler(LOG_PATH).run()

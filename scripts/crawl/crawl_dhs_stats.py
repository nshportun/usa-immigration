"""
Crawl DHS, CBP, ICE, and Census statistical datasets.
Stores structured tables as raw JSON/CSV on S3, tagged as statistics.
"""

import re
from pathlib import Path
from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup

from scripts.crawl.base_crawler import BaseCrawler

log = structlog.get_logger()

DHS_YEARBOOK = "https://www.dhs.gov/immigration-statistics/yearbook"
CBP_STATS = "https://www.cbp.gov/newsroom/stats/southwest-land-border-encounters"
ICE_STATS = "https://www.ice.gov/doclib/eroe/ice-fy2023-enforcement-and-removal-operations-report.pdf"
CENSUS_FOREIGN_BORN = "https://data.census.gov/table/ACSDT1Y2023.B05002"
DATA_GOV_SEARCH = "https://catalog.data.gov/api/3/action/package_search?q=immigration&rows=50&fq=res_format:JSON"

RAW_PREFIX = "data_raw/dhs_stats"
LOG_PATH = Path("metadata/crawl_log.jsonl")


class DHSStatsCrawler(BaseCrawler):
    source_name = "DHS_Stats"
    source_type = "statistics"
    authority_level = "primary_official"

    def crawl_dhs_yearbook(self):
        log.info("crawl_dhs_yearbook_start")
        html = self.fetch(DHS_YEARBOOK, f"{RAW_PREFIX}/yearbook")
        if not html:
            return
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"\.(xlsx?|csv|json|pdf)$", href, re.I):
                full = urljoin("https://www.dhs.gov", href)
                if re.search(r"\.(pdf)$", href, re.I):
                    self.fetch_pdf(full, f"{RAW_PREFIX}/yearbook/pdf")
                else:
                    self.fetch(full, f"{RAW_PREFIX}/yearbook/data")
            elif "/yearbook/" in href:
                self.fetch(urljoin("https://www.dhs.gov", href), f"{RAW_PREFIX}/yearbook")

    def crawl_cbp(self):
        log.info("crawl_cbp_start")
        html = self.fetch(CBP_STATS, f"{RAW_PREFIX}/cbp")
        if not html:
            return
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"\.(xlsx?|csv|json)$", href, re.I):
                self.fetch(urljoin("https://www.cbp.gov", href), f"{RAW_PREFIX}/cbp/data")

    def crawl_ice(self):
        log.info("crawl_ice_start")
        self.fetch_pdf(ICE_STATS, f"{RAW_PREFIX}/ice/pdf")
        ice_stats_page = "https://www.ice.gov/statistics"
        html = self.fetch(ice_stats_page, f"{RAW_PREFIX}/ice")
        if html:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.search(r"\.(xlsx?|csv|pdf)$", href, re.I):
                    full = urljoin("https://www.ice.gov", href)
                    if re.search(r"\.pdf$", href, re.I):
                        self.fetch_pdf(full, f"{RAW_PREFIX}/ice/pdf")
                    else:
                        self.fetch(full, f"{RAW_PREFIX}/ice/data")

    def crawl_data_gov(self):
        log.info("crawl_data_gov_start")
        self.fetch(DATA_GOV_SEARCH, f"{RAW_PREFIX}/data_gov")

    def run(self):
        self.crawl_dhs_yearbook()
        self.crawl_cbp()
        self.crawl_ice()
        self.crawl_data_gov()
        log.info("dhs_stats_crawl_complete")


if __name__ == "__main__":
    DHSStatsCrawler(LOG_PATH).run()

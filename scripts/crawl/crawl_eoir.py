"""
Crawl EOIR / DOJ: all 28 BIA precedent volumes, precedent charts, CourtListener API.
"""

import re
import time
from pathlib import Path
from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup

from scripts.crawl.base_crawler import BaseCrawler

log = structlog.get_logger()

EOIR_AG_BIA = "https://www.justice.gov/eoir/ag-bia-decisions"
BIA_VOL_PDF = "https://www.justice.gov/sites/default/files/eoir/legacy/2012/08/14/Vol{n}.pdf"
BIA_CHARTS = [
    "https://www.justice.gov/eoir/bia-precedent-chart",
    "https://www.justice.gov/eoir/bia-precedent-chart-d-i",
    "https://www.justice.gov/eoir/bia-precedent-chart-j-r",
    "https://www.justice.gov/eoir/bia-precedent-chart-s-z",
]
EOIR_DATA_GOV = [
    "https://catalog.data.gov/dataset/eoir-immigration-court-data",
    "https://catalog.data.gov/dataset/ag-bia-precedent-decisions",
]

# CourtListener API — immigration opinions from high-volume circuits
CL_API = "https://www.courtlistener.com/api/rest/v3/opinions/"
CL_COURTS = ["ca9", "ca5", "ca2", "ca11", "ca4", "ca1", "ca3", "ca6", "ca7", "ca8", "ca10", "cadc"]
CL_PAGE_SIZE = 20
CL_MAX_PAGES = 10  # 200 opinions per circuit

RAW_EOIR = "data_raw/eoir_bia"
RAW_CL = "data_raw/courtlistener"
LOG_PATH = Path("metadata/crawl_log.jsonl")


class EOIRCrawler(BaseCrawler):
    source_name = "EOIR_DOJ"
    source_type = "case_law"
    authority_level = "primary_official"

    def crawl_bia_index(self):
        log.info("crawl_bia_index_start")
        self.fetch(EOIR_AG_BIA, f"{RAW_EOIR}/index")

    def crawl_bia_volumes(self):
        log.info("crawl_bia_volumes_start")
        fetched = 0
        for n in range(1, 29):
            url = BIA_VOL_PDF.format(n=n)
            result = self.fetch_pdf(url, f"{RAW_EOIR}/volumes")
            if result:
                fetched += 1
        log.info("crawl_bia_volumes_done", fetched=fetched)

    def crawl_bia_charts(self):
        log.info("crawl_bia_charts_start")
        for url in BIA_CHARTS:
            self.fetch(url, f"{RAW_EOIR}/charts")

    def crawl_eoir_data_gov(self):
        log.info("crawl_eoir_data_gov_start")
        for url in EOIR_DATA_GOV:
            self.fetch(url, f"{RAW_EOIR}/data_gov")

    def crawl_courtlistener(self):
        log.info("crawl_courtlistener_start", circuits=len(CL_COURTS))
        for court in CL_COURTS:
            for page in range(1, CL_MAX_PAGES + 1):
                url = (
                    f"{CL_API}?court={court}&type=10"
                    f"&page={page}&page_size={CL_PAGE_SIZE}"
                    f"&fields=id,absolute_url,case_name,date_filed,plain_text"
                )
                html = self.fetch(url, f"{RAW_CL}/{court}")
                if not html:
                    break
                # Stop if no results
                if '"count": 0' in html or '"results": []' in html:
                    break
                time.sleep(0.5)

    def run(self):
        self.crawl_bia_index()
        self.crawl_bia_volumes()
        self.crawl_bia_charts()
        self.crawl_eoir_data_gov()
        self.crawl_courtlistener()
        log.info("eoir_crawl_complete")


if __name__ == "__main__":
    EOIRCrawler(LOG_PATH).run()

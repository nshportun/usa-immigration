"""
Community Q&A crawl — for question paraphrases and intent mining only.
Outputs are labeled community_non_authoritative.

Sources:
  - VisaJourney public forums (timeline and process threads)
  - Law Stack Exchange (already in open_datasets via HF, this handles direct crawl fallback)

Note: Reddit is excluded here — their API requires OAuth and rate limits are strict.
      Use the Pushshift/HF Reddit dataset if needed in a separate step.
"""

import re
from pathlib import Path
from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup

from scripts.crawl.base_crawler import BaseCrawler

log = structlog.get_logger()

VISAJOURNEY_FORUMS = [
    "https://www.visajourney.com/forums/forum/169-us-immigration-general-discussion/",
    "https://www.visajourney.com/forums/forum/170-family-based-immigration/",
    "https://www.visajourney.com/forums/forum/172-adjustment-of-status/",
    "https://www.visajourney.com/forums/forum/173-employment-based-immigration/",
    "https://www.visajourney.com/forums/forum/174-naturalization-citizenship/",
]
LAW_SE_IMMIGRATION = (
    "https://law.stackexchange.com/questions/tagged/immigration?tab=votes&pagesize=50"
)

RAW_PREFIX = "data_raw/community"
LOG_PATH = Path("metadata/crawl_log.jsonl")
MAX_PAGES = 10  # per forum section


class CommunityCrawler(BaseCrawler):
    source_name = "Community_QA"
    source_type = "community_non_authoritative"
    authority_level = "community_non_authoritative"

    def crawl_visajourney(self):
        log.info("crawl_visajourney_start")
        for base_url in VISAJOURNEY_FORUMS:
            html = self.fetch(base_url, f"{RAW_PREFIX}/visajourney")
            if not html:
                continue
            soup = BeautifulSoup(html, "lxml")
            thread_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.search(r"/topic/\d+", href):
                    thread_links.add(urljoin("https://www.visajourney.com", href))
            for url in list(thread_links)[:50]:
                self.fetch(url, f"{RAW_PREFIX}/visajourney/threads")

    def crawl_law_se(self):
        log.info("crawl_law_se_start")
        url = LAW_SE_IMMIGRATION
        for page in range(1, MAX_PAGES + 1):
            page_url = f"{url}&page={page}"
            html = self.fetch(page_url, f"{RAW_PREFIX}/law_se")
            if not html:
                break
            soup = BeautifulSoup(html, "lxml")
            q_links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.match(r"^/questions/\d+/", href):
                    q_links.add(urljoin("https://law.stackexchange.com", href))
            for url2 in q_links:
                self.fetch(url2, f"{RAW_PREFIX}/law_se/questions")

    def run(self):
        self.crawl_visajourney()
        self.crawl_law_se()
        log.info("community_crawl_complete")


if __name__ == "__main__":
    CommunityCrawler(LOG_PATH).run()

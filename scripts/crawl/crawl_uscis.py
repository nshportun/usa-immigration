"""
USCIS crawler — full chapter-level Policy Manual + all topic hubs + form PDFs.
Implements the complete URL inventory from CLAUDE.md Part 1-3.
"""

import datetime
import re
from pathlib import Path
from urllib.parse import urljoin

import structlog
from bs4 import BeautifulSoup

from scripts.crawl.base_crawler import BaseCrawler

log = structlog.get_logger()

# ── Policy Manual ─────────────────────────────────────────────────────────────
PM_EXPORT = "https://www.uscis.gov/policy-manual/export"
PM_BASE = "https://www.uscis.gov/policy-manual"

VOLUMES = {
    1:  list("ABCDEFGH"),
    2:  list("ABCDEFGHIJKLMNOPQRST"),
    3:  list("ABCD"),
    4:  list("ABCDEF"),
    5:  list("ABCDE"),
    6:  list("ABCDEFGHIJKLM"),
    7:  list("ABCDEFGHI"),
    8:  list("ABCDEFGHIJKLM"),
    9:  list("ABCDEFG"),
    10: list("ABCDEFGHI"),
    11: list("ABCDE"),
    12: list("ABCDEFGHIJK"),
}

# ── Form instruction PDFs ─────────────────────────────────────────────────────
FORM_INSTR_BASE = "https://www.uscis.gov/sites/default/files/document/forms"
FORM_INSTR_PDFS = [
    "i-485instr.pdf", "i-130instr.pdf", "i-765instr.pdf", "i-131instr.pdf",
    "n-400instr.pdf", "i-589instr.pdf", "i-864instr.pdf", "i-140instr.pdf",
    "i-129instr.pdf", "i-90instr.pdf",  "i-360instr.pdf", "i-601instr.pdf",
    "i-821dinstr.pdf", "i-918instr.pdf",
]
ALL_FORMS_PAGE = "https://www.uscis.gov/forms/all-forms"

# ── Topic hub pages ───────────────────────────────────────────────────────────
FAQ_PAGES = [
    "https://www.uscis.gov/citizenship/learn-about-citizenship/citizenship-and-naturalization/frequently-asked-questions",
    "https://www.uscis.gov/green-card/frequently-asked-questions",
    "https://www.uscis.gov/working-in-the-united-states/information-for-employers-and-employees/frequently-asked-questions-about-form-i-9",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum/asylum/questions-and-answers-asylum",
    "https://www.uscis.gov/family/frequently-asked-questions-family-petitions",
]

EMPLOYMENT_HUB = [
    "https://www.uscis.gov/working-in-the-united-states",
    "https://www.uscis.gov/working-in-the-united-states/information-for-employers-and-employees",
    "https://www.uscis.gov/working-in-the-united-states/information-for-employers-and-employees/employment-authorization",
    "https://www.uscis.gov/green-card/employment-authorization-document",
    "https://www.uscis.gov/working-in-the-united-states/temporary-workers",
    "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations",
    "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-2a-temporary-agricultural-workers",
    "https://www.uscis.gov/working-in-the-united-states/temporary-workers/h-2b-temporary-non-agricultural-workers",
    "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors",
    "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/students-and-employment",
    "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students",
    "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt",
    "https://www.uscis.gov/working-in-the-united-states/permanent-workers",
    "https://www.uscis.gov/working-in-the-united-states/permanent-workers/employment-based-immigration-first-preference-eb-1",
    "https://www.uscis.gov/working-in-the-united-states/permanent-workers/employment-based-immigration-second-preference-eb-2",
    "https://www.uscis.gov/working-in-the-united-states/permanent-workers/employment-based-immigration-third-preference-eb-3",
    "https://www.uscis.gov/working-in-the-united-states/permanent-workers/employment-based-immigration-fourth-preference-eb-4",
    "https://www.uscis.gov/working-in-the-united-states/permanent-workers/employment-based-immigration-fifth-preference-eb-5",
]

TRAVEL_HUB = [
    "https://www.uscis.gov/travel",
    "https://www.uscis.gov/travel-abroad",
    "https://www.uscis.gov/travel-abroad/reentry-permits",
    "https://www.uscis.gov/travel-abroad/advance-parole",
    "https://www.uscis.gov/travel-abroad/refugee-travel-documents",
    "https://www.uscis.gov/travel-abroad/carrier-documentation",
    "https://www.uscis.gov/travel-abroad/automatic-revalidation",
    "https://www.uscis.gov/green-card/after-we-grant-your-green-card/international-travel-as-a-permanent-resident",
]

FAMILY_HUB = [
    "https://www.uscis.gov/family",
    "https://www.uscis.gov/family/family-of-us-citizens",
    "https://www.uscis.gov/family/family-of-us-citizens/immediate-relatives-of-us-citizens",
    "https://www.uscis.gov/family/family-of-us-citizens/family-preference-immigrants",
    "https://www.uscis.gov/family/family-of-us-citizens/petition-for-your-spouse",
    "https://www.uscis.gov/family/family-of-us-citizens/petition-for-your-child",
    "https://www.uscis.gov/family/family-of-us-citizens/petition-for-your-parent",
    "https://www.uscis.gov/family/family-of-us-citizens/petition-for-your-sibling",
    "https://www.uscis.gov/family/family-of-us-citizens/petition-for-your-fiance",
    "https://www.uscis.gov/family/family-of-permanent-residents",
    "https://www.uscis.gov/family/family-of-permanent-residents/petition-for-your-spouse-and-children",
    "https://www.uscis.gov/family/vawa-based-immigration",
    "https://www.uscis.gov/family/abused-spouses-children-and-parents",
]

AOS_HUB = [
    "https://www.uscis.gov/green-card",
    "https://www.uscis.gov/green-card/green-card-processes-and-procedures",
    "https://www.uscis.gov/green-card/green-card-processes-and-procedures/adjustment-of-status",
    "https://www.uscis.gov/green-card/green-card-processes-and-procedures/concurrent-filing",
    "https://www.uscis.gov/green-card/green-card-processes-and-procedures/national-visa-center",
    "https://www.uscis.gov/green-card/green-card-processes-and-procedures/visa-bulletin",
    "https://www.uscis.gov/green-card/green-card-processes-and-procedures/interview",
    "https://www.uscis.gov/green-card/after-we-grant-your-green-card",
    "https://www.uscis.gov/green-card/after-we-grant-your-green-card/conditions-on-permanent-residence",
    "https://www.uscis.gov/green-card/after-we-grant-your-green-card/conditions-on-permanent-residence/removing-conditions-on-permanent-residence-based-on-marriage",
    "https://www.uscis.gov/green-card/green-card-through-family",
    "https://www.uscis.gov/green-card/green-card-through-job",
    "https://www.uscis.gov/green-card/green-card-through-refugee-or-asylee-status",
    "https://www.uscis.gov/green-card/green-card-through-special-immigrant-status",
    "https://www.uscis.gov/green-card/green-card-through-diversity-visa",
    "https://www.uscis.gov/green-card/green-card-through-other-categories",
]

NATURALIZATION_HUB = [
    "https://www.uscis.gov/citizenship",
    "https://www.uscis.gov/citizenship/apply-for-citizenship",
    "https://www.uscis.gov/citizenship/learn-about-citizenship",
    "https://www.uscis.gov/citizenship/learn-about-citizenship/citizenship-and-naturalization",
    "https://www.uscis.gov/citizenship/learn-about-citizenship/10-steps-to-naturalization",
    "https://www.uscis.gov/citizenship/learn-about-citizenship/citizenship-and-naturalization/naturalization-eligibility",
    "https://www.uscis.gov/citizenship/civics-test",
    "https://www.uscis.gov/citizenship/find-study-materials-and-resources",
    "https://www.uscis.gov/citizenship/find-study-materials-and-resources/study-for-the-test",
]

HUMANITARIAN_HUB = [
    "https://www.uscis.gov/humanitarian",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum/asylum",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum/asylum/asylum-processes",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum/asylum/affirmative-asylum-process",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum/asylum/defensive-asylum",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum/asylum/withholding-of-removal",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum/refugees",
    "https://www.uscis.gov/humanitarian/refugees-and-asylum/refugees/eligibility",
    "https://www.uscis.gov/humanitarian/temporary-protected-status",
    "https://www.uscis.gov/humanitarian/deferred-action",
    "https://www.uscis.gov/humanitarian/deferred-action-for-childhood-arrivals-daca",
    "https://www.uscis.gov/humanitarian/victims-of-human-trafficking-and-other-crimes",
    "https://www.uscis.gov/humanitarian/victims-of-human-trafficking-and-other-crimes/victims-of-criminal-activity-u-nonimmigrant-status",
    "https://www.uscis.gov/humanitarian/victims-of-human-trafficking-and-other-crimes/victims-of-human-trafficking-t-nonimmigrant-status",
    "https://www.uscis.gov/humanitarian/special-immigrant-juveniles",
]

MY_USCIS_HELP = [
    "https://my.uscis.gov/explorations/topics/green_card",
    "https://my.uscis.gov/explorations/topics/citizenship",
    "https://my.uscis.gov/explorations/topics/work_in_us",
    "https://my.uscis.gov/explorations/topics/visitor_tourist",
    "https://my.uscis.gov/explorations/topics/student_exchange",
    "https://my.uscis.gov/explorations/topics/humanitarian",
]

DATA_PAGES = [
    "https://egov.uscis.gov/processing-times/",
    "https://www.uscis.gov/forms/filing-fees",
]

RAW_PM   = "data_raw/uscis_policy_manual"
RAW_FORM = "data_raw/uscis_forms"
RAW_TOPIC = "data_raw/uscis_topic_pages"
LOG_PATH = Path("metadata/crawl_log.jsonl")


class USCISCrawler(BaseCrawler):
    source_name = "USCIS"
    source_type = "official_policy"
    authority_level = "primary_official"

    # ── Policy Manual ──────────────────────────────────────────────────────────

    def crawl_policy_manual(self):
        log.info("pm_export_attempt")
        html = self.fetch(PM_EXPORT, RAW_PM)
        if html and len(html) > 100_000:
            log.info("pm_export_success", bytes=len(html))
            return
        log.info("pm_export_blocked_or_small", fallback="chapter_by_chapter")
        self._crawl_pm_chapter_by_chapter()

    def _crawl_pm_chapter_by_chapter(self):
        total = 0
        for vol_num, parts in VOLUMES.items():
            for part in parts:
                part_url = f"{PM_BASE}/volume-{vol_num}-part-{part.lower()}"
                html = self.fetch(part_url, RAW_PM)
                if not html:
                    continue
                # Extract chapter links from the part index page
                soup = BeautifulSoup(html, "lxml")
                chapter_urls = []
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if re.search(rf"volume-{vol_num}-part-{part.lower()}-chapter-\d+", href, re.I):
                        full = urljoin("https://www.uscis.gov", href)
                        chapter_urls.append(full)
                # Also try chapters 1-20 directly if none found via links
                if not chapter_urls:
                    for ch in range(1, 21):
                        chapter_urls.append(
                            f"{PM_BASE}/volume-{vol_num}-part-{part.lower()}-chapter-{ch}"
                        )
                for ch_url in chapter_urls:
                    result = self.fetch(ch_url, RAW_PM)
                    if result:
                        total += 1
        log.info("pm_chapter_crawl_done", pages_fetched=total)

    # ── Forms ──────────────────────────────────────────────────────────────────

    def crawl_forms(self):
        log.info("crawl_forms_start")
        # All-forms discovery
        html = self.fetch(ALL_FORMS_PAGE, RAW_FORM)
        if html:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if re.match(r"^/forms/[a-z0-9-]+$", href):
                    self.fetch(urljoin("https://www.uscis.gov", href), RAW_FORM)

        # Direct instruction PDFs
        for pdf_name in FORM_INSTR_PDFS:
            url = f"{FORM_INSTR_BASE}/{pdf_name}"
            self.fetch_pdf(url, f"{RAW_FORM}/instructions")

    # ── Topic hubs ─────────────────────────────────────────────────────────────

    def _crawl_url_list(self, urls: list[str], prefix: str):
        for url in urls:
            self.fetch(url, prefix)

    def crawl_topic_hubs(self):
        log.info("crawl_topic_hubs_start")
        # Employment and travel first (zero-doc gaps from previous run)
        self._crawl_url_list(EMPLOYMENT_HUB, f"{RAW_TOPIC}/employment")
        self._crawl_url_list(TRAVEL_HUB,     f"{RAW_TOPIC}/travel")
        self._crawl_url_list(FAMILY_HUB,     f"{RAW_TOPIC}/family")
        self._crawl_url_list(AOS_HUB,        f"{RAW_TOPIC}/adjustment")
        self._crawl_url_list(NATURALIZATION_HUB, f"{RAW_TOPIC}/naturalization")
        self._crawl_url_list(HUMANITARIAN_HUB,   f"{RAW_TOPIC}/humanitarian")
        self._crawl_url_list(FAQ_PAGES,      f"{RAW_TOPIC}/faqs")
        self._crawl_url_list(MY_USCIS_HELP,  f"{RAW_TOPIC}/my_uscis")
        self._crawl_url_list(DATA_PAGES,     f"{RAW_TOPIC}/data")

    def run(self):
        self.crawl_policy_manual()
        self.crawl_forms()
        self.crawl_topic_hubs()
        log.info("uscis_crawl_complete")


if __name__ == "__main__":
    USCISCrawler(LOG_PATH).run()

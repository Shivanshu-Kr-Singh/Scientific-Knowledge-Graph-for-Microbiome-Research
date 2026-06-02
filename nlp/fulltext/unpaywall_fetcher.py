"""
nlp/fulltext/unpaywall_fetcher.py
----------------------------------
Fetches legal open-access full text via the Unpaywall API and PDF parser.

Unpaywall tracks legal open-access versions of papers across institutional
repositories, author manuscripts, preprint servers, and publisher OA pages.
No API key required — just a registered email address.

API docs: https://unpaywall.org/products/api
"""

import os
import requests
from loguru import logger

from nlp.fulltext.pdf_parser import PDFParser
from nlp.fulltext.web_scraper import WebScraper

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_EMAIL = os.getenv("NCBI_EMAIL", "research@example.com")


class UnpaywallFetcher:
    """
    Queries Unpaywall for a legal open-access PDF or HTML URL,
    then fetches the full text from that URL.

    Strategy:
      1. GET https://api.unpaywall.org/v2/{doi}?email={email}
      2. From response, collect all oa_locations sorted by best_oa_location first
      3. Try PDF URL first (pdf_parser), then landing page (web_scraper)
      4. Return first successful result
    """

    def __init__(self) -> None:
        self._pdf = PDFParser()
        self._web = WebScraper()

    def fetch(self, doi: str) -> dict | None:
        """
        Attempt to fetch full text for a paper via Unpaywall.

        Args:
            doi: DOI string, e.g. "10.1038/s41586-024-07999-z"

        Returns:
            dict with full_text/section keys and fetch_source/status,
            or None if no OA version found.
        """
        if not doi or not doi.strip():
            return None

        doi_clean = doi.strip().lstrip("https://doi.org/").lstrip("doi:")

        try:
            resp = requests.get(
                f"{UNPAYWALL_BASE}/{doi_clean}",
                params={"email": _EMAIL},
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()

            # Not open access at all
            if not data.get("is_oa", False):
                return None

            # Collect OA locations — best first
            locations = []
            best = data.get("best_oa_location")
            if best:
                locations.append(best)
            for loc in data.get("oa_locations", []):
                if loc != best:
                    locations.append(loc)

            for loc in locations:
                # Try PDF first
                pdf_url = loc.get("url_for_pdf")
                if pdf_url:
                    result = self._pdf.fetch(pdf_url)
                    if result:
                        result["fetch_source"] = "unpaywall_pdf"
                        result["fetch_status"] = "success"
                        logger.debug(
                            "[Unpaywall] Full text via PDF for DOI {}", doi_clean
                        )
                        return result

                # Try landing page
                landing_url = loc.get("url_for_landing_page") or loc.get("url")
                if landing_url:
                    result = self._web.fetch(landing_url)
                    if result:
                        result["fetch_source"] = "unpaywall_web"
                        result["fetch_status"] = "success"
                        logger.debug(
                            "[Unpaywall] Full text via web scrape for DOI {}", doi_clean
                        )
                        return result

        except Exception as exc:
            logger.warning("[Unpaywall] Failed for DOI {}: {}", doi_clean, exc)

        return None

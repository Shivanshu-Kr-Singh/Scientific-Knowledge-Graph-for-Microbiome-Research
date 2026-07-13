"""
nlp/fulltext/openaire_fetcher.py
-----------------------------------
Fetches legal open-access full text via the OpenAIRE Graph.

OpenAIRE aggregates open-access research outputs from hundreds of European
repositories, institutional archives, DOAJ, BASE, and preprint servers.
It's a different index than Unpaywall — overlapping but not identical
coverage, so it's tried as an additional Tier 2 strategy, not a duplicate.

No API key required for the legacy search endpoint used here.

API docs: https://graph.openaire.eu/docs/apis/search-api/
Endpoint: https://api.openaire.eu/search/publications?doi={doi}

RESPONSE SHAPE (relevant parts only):
  response.results.result[0].metadata."oaf:entity"."oaf:result"
    .bestaccessright.@classid        → "OPEN" | "CLOSED" | "EMBARGO" | ...
    .children.result[].instance[]    → list of hosted copies, each with:
        .accessright.@classid        → per-instance access level
        .webresource.url.$ (or .url.$) → direct URL to the full text/PDF

Multiple instances can exist (different repositories hosting the same
paper) — all OPEN-access URLs are tried in order until one yields
parseable text.
"""

from typing import Optional
from loguru import logger
import requests

from nlp.fulltext.pdf_parser import PDFParser
from nlp.fulltext.web_scraper import WebScraper
from nlp.fulltext.domain_throttle import throttle as domain_throttle

OPENAIRE_BASE = "https://api.openaire.eu/search/publications"


class OpenAIREFetcher:
    """
    Queries the OpenAIRE Graph for open-access copies of a paper by DOI,
    then fetches full text from whichever hosted URL responds first.

    Strategy:
      1. GET https://api.openaire.eu/search/publications?doi={doi}&format=json
      2. Walk result.metadata.oaf:entity.oaf:result.children.result[].instance[]
      3. Keep only instances with accessright == OPEN
      4. Try each URL: PDF parser first (many OpenAIRE URLs are direct PDFs),
         fall back to web scraper for landing pages
      5. Return the first successful result
    """

    def __init__(self) -> None:
        self._pdf = PDFParser()
        self._web = WebScraper()

    def fetch(self, doi: str) -> Optional[dict]:
        """
        Attempt to fetch full text for a paper via OpenAIRE.

        Args:
            doi: DOI string, e.g. "10.1371/journal.pone.0000308"

        Returns:
            dict with full_text/section keys and fetch_source/status,
            or None if no open-access copy was found or fetchable.
        """
        if not doi or not doi.strip():
            return None

        doi_clean = doi.strip()

        urls = self._lookup_urls(doi_clean)
        if not urls:
            return None

        for url in urls:
            # Try PDF parsing first — many OpenAIRE-hosted URLs are direct
            # PDF links (e.g. repository /article/file?type=printable links)
            result = self._pdf.fetch(url)
            if result:
                result["fetch_source"] = "openaire_pdf"
                result["fetch_status"] = "success"
                logger.debug(f"[OpenAIRE] Full text via PDF for DOI {doi_clean}")
                return result

            # Fall back to web scraping the landing page
            result = self._web.fetch(url)
            if result:
                result["fetch_source"] = "openaire_web"
                result["fetch_status"] = "success"
                logger.debug(f"[OpenAIRE] Full text via web scrape for DOI {doi_clean}")
                return result

        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _lookup_urls(self, doi: str) -> list[str]:
        """
        Queries OpenAIRE for a DOI and returns a list of open-access URLs,
        ordered as returned by the API (best/most complete instance first).
        """
        try:
            domain_throttle(OPENAIRE_BASE)
            resp = requests.get(
                OPENAIRE_BASE,
                params={"doi": doi, "format": "json"},
                timeout=15,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
        except Exception as e:
            logger.debug(f"[OpenAIRE] Lookup failed for DOI {doi}: {e}")
            return []

        try:
            results = (data.get("response") or {}).get("results") or {}
            result_list = self._as_list(results.get("result"))
            if not result_list:
                return []

            oaf_result = (
                result_list[0]
                .get("metadata", {})
                .get("oaf:entity", {})
                .get("oaf:result", {})
            )

            # Skip entirely if the aggregate access right isn't open
            best_access = (oaf_result.get("bestaccessright") or {}).get("@classid", "")
            if best_access and best_access.upper() != "OPEN":
                return []

            children = self._as_list(oaf_result.get("children", {}).get("result"))
            instances = []
            for child in children:
                instances.extend(self._as_list(child.get("instance")))

            urls: list[str] = []
            for inst in instances:
                access = (inst.get("accessright") or {}).get("@classid", "")
                if access and access.upper() != "OPEN":
                    continue

                url = self._extract_url(inst)
                if url and url not in urls:
                    urls.append(url)

            return urls

        except Exception as e:
            logger.debug(f"[OpenAIRE] Response parse failed for DOI {doi}: {e}")
            return []

    @staticmethod
    def _as_list(value) -> list:
        """
        OpenAIRE's XML-derived JSON collapses single-element repeating
        fields into a bare dict instead of a one-item list. Normalizes
        None/dict/list into a consistent list so callers never have to
        special-case the shape.
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _extract_url(self, instance: dict) -> Optional[str]:
        """Pulls the actual URL string out of an instance's varying shapes."""
        webresource = instance.get("webresource")
        if webresource:
            url_field = webresource.get("url")
            if isinstance(url_field, dict):
                url = url_field.get("$")
                if url:
                    return url

        url_field = instance.get("url")
        if isinstance(url_field, dict):
            return url_field.get("$")
        if isinstance(url_field, str):
            return url_field
        if isinstance(url_field, list) and url_field:
            first = url_field[0]
            if isinstance(first, dict):
                return first.get("$")
            if isinstance(first, str):
                return first

        return None

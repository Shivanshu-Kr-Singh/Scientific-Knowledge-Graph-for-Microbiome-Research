"""
nlp/fulltext/ncbi_pmc_fetcher.py
----------------------------------
Fetches full text from PubMed Central via the NCBI Entrez API.

The PMC OA subset accessible via NCBI is larger than what EuropePMC exposes —
it includes NIH-mandate papers, US journal deposits, and author manuscripts
that EuropePMC doesn't carry. Using both as sequential fallbacks maximises
full-text coverage for papers with a PMCID.

API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}
Rate limits: 3 req/sec without NCBI_API_KEY, 10/sec with it.
"""

import os
import time
from loguru import logger

try:
    from Bio import Entrez
    from bs4 import BeautifulSoup
    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False

_EMAIL   = os.getenv("NCBI_EMAIL",   "research@example.com")
_API_KEY = os.getenv("NCBI_API_KEY", "")
_DELAY   = 0.12 if _API_KEY else 0.35   # seconds between NCBI calls


class NCBIPMCFetcher:
    """
    Fetches JATS XML full text from PubMed Central via Entrez efetch.

    Returns a dict with section keys (abstract, methods, results, discussion)
    and fetch_source="ncbi_pmc".  Returns None when the paper is not in the
    PMC OA subset or the fetch fails.
    """

    def __init__(self) -> None:
        if ENTREZ_AVAILABLE:
            Entrez.email = _EMAIL
            if _API_KEY:
                Entrez.api_key = _API_KEY

    def fetch(self, pmcid: str) -> dict | None:
        """
        Fetch full-text XML from PMC for the given PMCID.

        Args:
            pmcid: PMC ID string, with or without the "PMC" prefix,
                   e.g. "PMC9876543" or "9876543".

        Returns:
            dict with section text and fetch metadata, or None on failure.
        """
        if not ENTREZ_AVAILABLE:
            return None

        if not pmcid or not str(pmcid).strip():
            return None

        # Normalise — strip "PMC" prefix; Entrez db=pmc expects bare numbers
        pmcid_str = str(pmcid).strip().upper().lstrip("PMC")
        if not pmcid_str.isdigit():
            return None

        try:
            handle = Entrez.efetch(
                db="pmc",
                id=pmcid_str,
                rettype="full",
                retmode="xml",
            )
            xml_text = handle.read()
            handle.close()
            time.sleep(_DELAY)

            if not xml_text:
                return None

            return self._parse_jats_xml(xml_text, pmcid_str)

        except Exception as exc:
            logger.warning("[NCBI PMC] Fetch failed for PMCID {}: {}", pmcid_str, exc)
            return None

    # ── XML parser ────────────────────────────────────────────────────────────

    def _parse_jats_xml(self, xml_bytes: bytes, pmcid: str) -> dict | None:
        """
        Parse JATS XML into section text dict.

        JATS section types we look for:
          <abstract>           → abstract
          <sec sec-type="methods">  / <sec> with title "Methods"  → methods
          <sec sec-type="results">  / <sec> with title "Results"  → results
          <sec sec-type="discussion"> / "Discussion"              → discussion
          <sec sec-type="conclusions"> / "Conclusion"             → discussion
        """
        try:
            soup = BeautifulSoup(xml_bytes, "xml")
        except Exception as exc:
            logger.warning("[NCBI PMC] XML parse error for PMCID {}: {}", pmcid, exc)
            return None

        def _text(tag) -> str:
            return tag.get_text(" ", strip=True) if tag else ""

        # ── Abstract ──────────────────────────────────────────────────────────
        abstract = _text(soup.find("abstract"))

        # ── Body sections ─────────────────────────────────────────────────────
        methods    = ""
        results    = ""
        discussion = ""
        full_body  = ""

        body = soup.find("body")
        if body:
            full_body = _text(body)

            for sec in body.find_all("sec"):
                sec_type = (sec.get("sec-type") or "").lower()
                title_tag = sec.find("title")
                title_text = _text(title_tag).lower() if title_tag else ""

                is_methods    = "method" in sec_type or "method" in title_text or "material" in title_text
                is_results    = "result" in sec_type or "result" in title_text or "finding" in title_text
                is_discussion = ("discussion" in sec_type or "discussion" in title_text
                                 or "conclusion" in sec_type or "conclusion" in title_text)

                if is_methods and not methods:
                    methods = _text(sec)
                elif is_results and not results:
                    results = _text(sec)
                elif is_discussion and not discussion:
                    discussion = _text(sec)

        # Must have at least abstract or body to be useful
        if not abstract and not full_body:
            return None

        result: dict = {
            "fetch_source": "ncbi_pmc",
            "fetch_status": "success",
        }
        if abstract:
            result["abstract"] = abstract
        if methods:
            result["methods"] = methods
        if results:
            result["results"] = results
        if discussion:
            result["discussion"] = discussion
        if full_body:
            result["full_text"] = full_body

        logger.debug(
            "[NCBI PMC] Full text fetched for PMCID {} "
            "(abstract={}, methods={}, results={}, discussion={})",
            pmcid,
            bool(abstract), bool(methods), bool(results), bool(discussion),
        )
        return result

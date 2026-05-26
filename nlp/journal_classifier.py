"""
nlp/journal_classifier.py
---------------------------
Looks up journal metadata: impact factor, quartile (Q1-Q4), field, open access status.

WHY THIS MATTERS:
  Q1 journals are the top 25% by impact factor in their field.
  This lets you query the graph: "show me only Q1 papers" or
  "filter out Q3/Q4 results" — critical for evidence quality assessment.

APPROACH:
  We maintain a curated list of the top microbiome-relevant journals with
  their known metadata. For journals not in our list, we query CrossRef
  to get basic info (ISSN, publisher, open access status).

  Impact factors change yearly. Our list uses 2023/2024 values.
  Update annually or pull from Scimago programmatically.

  Scimago data is available free at: https://www.scimagojr.com/
"""

import re
import time
from typing import Optional, Dict
from loguru import logger

from nlp.enriched_record import JournalInfo


# ── Curated journal database ──────────────────────────────────────────────────
# Top journals for human microbiome research with known metadata.
# Key: lowercase journal name or ISSN
# This covers ~80% of papers in our collection.

JOURNAL_DB: Dict[str, dict] = {
    # ── Top-tier general science ──────────────────────────────────────────────
    "nature":                       {"if": 64.8,  "q": "Q1", "field": "Multidisciplinary"},
    "science":                      {"if": 56.9,  "q": "Q1", "field": "Multidisciplinary"},
    "cell":                         {"if": 45.5,  "q": "Q1", "field": "Cell Biology"},
    "nature medicine":              {"if": 58.7,  "q": "Q1", "field": "Medicine"},
    "nature microbiology":          {"if": 28.3,  "q": "Q1", "field": "Microbiology"},
    "nature communications":        {"if": 16.6,  "q": "Q1", "field": "Multidisciplinary"},
    "science advances":             {"if": 13.6,  "q": "Q1", "field": "Multidisciplinary"},
    "pnas":                         {"if": 11.1,  "q": "Q1", "field": "Multidisciplinary"},
    "proceedings of the national academy of sciences": {"if": 11.1, "q": "Q1", "field": "Multidisciplinary"},

    # ── Microbiome-specific ───────────────────────────────────────────────────
    "microbiome":                   {"if": 13.8,  "q": "Q1", "field": "Microbiology", "oa": True},
    "gut microbes":                 {"if": 12.2,  "q": "Q1", "field": "Microbiology"},
    "npj biofilms and microbiomes": {"if": 7.8,   "q": "Q1", "field": "Microbiology", "oa": True},
    "msystems":                     {"if": 6.1,   "q": "Q1", "field": "Microbiology", "oa": True},
    "mbio":                         {"if": 6.4,   "q": "Q1", "field": "Microbiology", "oa": True},
    "microbial genomics":           {"if": 4.0,   "q": "Q2", "field": "Microbiology", "oa": True},

    # ── Gastroenterology ─────────────────────────────────────────────────────
    "gut":                          {"if": 24.5,  "q": "Q1", "field": "Gastroenterology"},
    "gastroenterology":             {"if": 29.4,  "q": "Q1", "field": "Gastroenterology"},
    "cell host & microbe":          {"if": 30.3,  "q": "Q1", "field": "Microbiology"},
    "cell host and microbe":        {"if": 30.3,  "q": "Q1", "field": "Microbiology"},
    "journal of crohn's and colitis": {"if": 9.2, "q": "Q1", "field": "Gastroenterology"},
    "alimentary pharmacology & therapeutics": {"if": 7.6, "q": "Q1", "field": "Gastroenterology"},
    "united european gastroenterology journal": {"if": 4.7, "q": "Q2", "field": "Gastroenterology"},

    # ── Bioinformatics / Computational ───────────────────────────────────────
    "bioinformatics":               {"if": 5.8,   "q": "Q1", "field": "Bioinformatics"},
    "genome biology":               {"if": 17.4,  "q": "Q1", "field": "Genomics", "oa": True},
    "genome research":              {"if": 6.5,   "q": "Q1", "field": "Genomics"},
    "nucleic acids research":       {"if": 14.9,  "q": "Q1", "field": "Biochemistry", "oa": True},
    "plos computational biology":   {"if": 4.5,   "q": "Q2", "field": "Bioinformatics", "oa": True},
    "briefings in bioinformatics":  {"if": 9.5,   "q": "Q1", "field": "Bioinformatics"},

    # ── Immunology / Infectious disease ──────────────────────────────────────
    "immunity":                     {"if": 25.5,  "q": "Q1", "field": "Immunology"},
    "journal of immunology":        {"if": 4.4,   "q": "Q2", "field": "Immunology"},
    "infection and immunity":       {"if": 3.0,   "q": "Q2", "field": "Microbiology", "oa": True},

    # ── Open access megajournals ─────────────────────────────────────────────
    "plos one":                     {"if": 3.7,   "q": "Q2", "field": "Multidisciplinary", "oa": True},
    "plos biology":                 {"if": 9.8,   "q": "Q1", "field": "Biology", "oa": True},
    "elife":                        {"if": 7.7,   "q": "Q1", "field": "Biology", "oa": True},
    "frontiers in microbiology":    {"if": 5.2,   "q": "Q2", "field": "Microbiology", "oa": True},
    "frontiers in cellular and infection microbiology": {"if": 5.7, "q": "Q1", "field": "Microbiology", "oa": True},
    "bmc microbiology":             {"if": 4.0,   "q": "Q2", "field": "Microbiology", "oa": True},
    "bmc genomics":                 {"if": 4.0,   "q": "Q2", "field": "Genomics", "oa": True},

    # ── Clinical nutrition / metabolomics ────────────────────────────────────
    "cell metabolism":              {"if": 27.3,  "q": "Q1", "field": "Metabolism"},
    "nature metabolism":            {"if": 18.9,  "q": "Q1", "field": "Metabolism"},
    "american journal of clinical nutrition": {"if": 8.5, "q": "Q1", "field": "Nutrition"},
    "nutrients":                    {"if": 5.9,   "q": "Q2", "field": "Nutrition", "oa": True},
}

# ── Predatory journal signals ─────────────────────────────────────────────────
# Journals known or suspected to be predatory (no peer review).
# Papers in these get flagged, not excluded — the researcher decides.
PREDATORY_SIGNALS = [
    "omics international",
    "longdom",
    "imedpub",
    "scitechnol",
    "gavin publishers",
]

# CrossRef API for journal lookup (fallback for journals not in our DB)
CROSSREF_JOURNALS_URL = "https://api.crossref.org/journals"


class JournalClassifier:
    """
    Classifies journals and attaches impact factor, quartile, and field metadata.
    Uses local DB first, CrossRef API as fallback.
    """

    def __init__(self):
        self._crossref_cache: Dict[str, dict] = {}

    def classify(self, journal_name: Optional[str], issn: Optional[str]) -> JournalInfo:
        """
        Returns a JournalInfo with as much metadata as we can find.
        
        LOOKUP ORDER:
          1. Exact match on lowercased journal name in our DB
          2. Partial name match (handles abbreviations)
          3. CrossRef API lookup by ISSN or name
          4. Return partial info with what we know
        """
        if not journal_name and not issn:
            return JournalInfo()

        journal_lower = (journal_name or "").lower().strip()

        # ── Step 1: Exact match ────────────────────────────────────────────────
        if journal_lower in JOURNAL_DB:
            return self._build_info(journal_name, issn, JOURNAL_DB[journal_lower])

        # ── Step 2: Partial / cleaned match ───────────────────────────────────
        # Handles cases like "Gut" matching "gut" or "The Gut Journal"
        cleaned = self._clean_journal_name(journal_lower)
        for db_name, meta in JOURNAL_DB.items():
            if cleaned in db_name or db_name in cleaned:
                return self._build_info(journal_name, issn, meta)

        # ── Step 3: CrossRef API lookup ────────────────────────────────────────
        crossref_meta = self._lookup_crossref(journal_name, issn)
        if crossref_meta:
            return self._build_info(journal_name, issn, crossref_meta)

        # ── Step 4: Return what we know (unknown quartile) ────────────────────
        is_predatory = self._check_predatory(journal_lower)
        is_oa = self._guess_open_access(journal_lower, issn)

        return JournalInfo(
            name=journal_name,
            issn=issn,
            quartile="unknown",
            is_predatory=is_predatory,
            is_open_access=is_oa,
        )

    def _build_info(self, name: Optional[str], issn: Optional[str], meta: dict) -> JournalInfo:
        """Builds a JournalInfo from a metadata dict."""
        return JournalInfo(
            name=name,
            issn=issn,
            impact_factor=meta.get("if"),
            quartile=meta.get("q", "unknown"),
            field=meta.get("field"),
            is_open_access=meta.get("oa", False),
            is_predatory=self._check_predatory((name or "").lower()),
        )

    def _clean_journal_name(self, name: str) -> str:
        """Removes common prefixes/articles that vary between sources."""
        name = re.sub(r"^(the|journal of the|journal of)\s+", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _check_predatory(self, journal_lower: str) -> bool:
        """Returns True if the journal name contains known predatory signals."""
        return any(signal in journal_lower for signal in PREDATORY_SIGNALS)

    def _guess_open_access(self, journal_lower: str, issn: Optional[str]) -> bool:
        """
        Guesses open access status from journal name patterns.
        Not foolproof — CrossRef is more reliable.
        """
        oa_signals = ["plos", "bmc", "frontiers", "elife", "open", "public library"]
        return any(s in journal_lower for s in oa_signals)

    def _lookup_crossref(self, journal_name: Optional[str], issn: Optional[str]) -> Optional[dict]:
        """
        Queries CrossRef for journal metadata.
        CrossRef is free, no API key needed, rate limit ~50 req/sec.
        We use it as a fallback to get basic publisher info.

        NOTE: CrossRef does NOT provide impact factors (those are proprietary
        to Clarivate/Web of Science). We only get: ISSN, publisher, OA status.
        """
        import requests

        cache_key = issn or journal_name
        if cache_key in self._crossref_cache:
            return self._crossref_cache[cache_key]

        try:
            time.sleep(0.1)   # Be polite to CrossRef
            params = {}
            if issn:
                url = f"{CROSSREF_JOURNALS_URL}/{issn}"
            else:
                url = CROSSREF_JOURNALS_URL
                params["query"] = journal_name
                params["rows"] = 1

            resp = requests.get(url, params=params, timeout=10,
                                headers={"User-Agent": "MicrobiomeMiner/1.0 (mailto:your@email.com)"})

            if resp.status_code != 200:
                return None

            data = resp.json()
            # CrossRef response structure varies by endpoint
            item = data.get("message", {})
            if "items" in item:
                items = item["items"]
                if not items:
                    return None
                item = items[0]

            # Extract what we can
            meta = {
                "field": None,
                "oa": False,
            }

            # Check if DOAJ-indexed (indicator of open access)
            flags = item.get("flags", {})
            if flags.get("is-oa"):
                meta["oa"] = True

            # Subject from CrossRef
            subjects = item.get("subjects", [])
            if subjects:
                meta["field"] = subjects[0].get("name")

            self._crossref_cache[cache_key] = meta
            return meta

        except Exception as e:
            logger.debug(f"[journal_classifier] CrossRef lookup failed: {e}")
            return None

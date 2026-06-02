"""
nlp/fulltext/ncbi_abstract_fetcher.py
---------------------------------------
Fetches complete abstracts from NCBI PubMed via the Entrez API.

Some collectors (EuropePMC, Semantic Scholar) return truncated abstracts.
PubMed always has the complete abstract for any paper indexed there.
This fetcher retrieves the full abstract and structured sections when available.

No paywall — any paper with a PMID is accessible.
Rate limits: 3 req/sec without API key, 10 req/sec with NCBI_API_KEY.
"""

import os
import time
from loguru import logger

try:
    from Bio import Entrez, Medline
    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False

_EMAIL = os.getenv("NCBI_EMAIL", "research@example.com")
_API_KEY = os.getenv("NCBI_API_KEY", "")
_DELAY = 0.12 if _API_KEY else 0.35  # seconds between requests


class NCBIAbstractFetcher:
    """
    Fetches complete abstracts from PubMed via Entrez efetch.

    Returns a dict compatible with the fulltext orchestrator schema,
    with the abstract in the "abstract" key and fetch_source="ncbi_abstract".

    Also attempts to extract structured sections if the PubMed record
    contains them (e.g. background, methods, results, conclusions).
    """

    def __init__(self) -> None:
        if ENTREZ_AVAILABLE:
            Entrez.email = _EMAIL
            if _API_KEY:
                Entrez.api_key = _API_KEY

    def fetch(self, pmid: str) -> dict | None:
        """
        Fetch complete abstract for a paper by PMID.

        Args:
            pmid: PubMed ID string, e.g. "38765432"

        Returns:
            dict with abstract and structured sections if available,
            or None if fetch fails.
        """
        if not ENTREZ_AVAILABLE:
            return None

        if not pmid or not str(pmid).strip():
            return None

        pmid_str = str(pmid).strip()

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid_str,
                rettype="medline",
                retmode="text",
            )
            records = list(Medline.parse(handle))
            handle.close()
            time.sleep(_DELAY)

            if not records:
                return None

            rec = records[0]

            # Full abstract — AB field in MEDLINE format
            abstract = rec.get("AB", "")
            if not abstract:
                return None

            # Try to extract structured sections from the abstract
            # PubMed structured abstracts use labels like "BACKGROUND:", "METHODS:", etc.
            sections = self._parse_structured_abstract(abstract)

            result = {
                "abstract": abstract,
                "fetch_source": "ncbi_abstract",
                "fetch_status": "success",
            }

            # Add structured sections if found
            if sections.get("methods"):
                result["methods"] = sections["methods"]
            if sections.get("results"):
                result["results"] = sections["results"]
            if sections.get("discussion") or sections.get("conclusions"):
                result["discussion"] = sections.get("discussion") or sections.get("conclusions", "")

            logger.debug("[NCBI] Full abstract fetched for PMID {}", pmid_str)
            return result

        except Exception as exc:
            logger.warning("[NCBI] Abstract fetch failed for PMID {}: {}", pmid_str, exc)
            return None

    def _parse_structured_abstract(self, abstract: str) -> dict:
        """
        Parse a structured PubMed abstract into sections.

        PubMed structured abstracts look like:
        "BACKGROUND: ... METHODS: ... RESULTS: ... CONCLUSIONS: ..."
        """
        import re

        sections = {}
        # Common section labels in PubMed structured abstracts
        label_patterns = [
            ("background", r"(?:BACKGROUND|INTRODUCTION|AIM|PURPOSE|OBJECTIVE[S]?)\s*:"),
            ("methods", r"(?:METHODS?|MATERIALS? AND METHODS?|STUDY DESIGN|PATIENTS? AND METHODS?)\s*:"),
            ("results", r"(?:RESULTS?|FINDINGS?|OUTCOMES?)\s*:"),
            ("discussion", r"(?:DISCUSSION|CONCLUSIONS?|CONCLUSION AND RELEVANCE|INTERPRETATION|SIGNIFICANCE)\s*:"),
        ]

        # Find all section boundaries
        boundaries = []
        for section_name, pattern in label_patterns:
            for match in re.finditer(pattern, abstract, re.IGNORECASE):
                boundaries.append((match.start(), match.end(), section_name))

        if not boundaries:
            return {}

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        # Extract content between boundaries
        for i, (start, end, name) in enumerate(boundaries):
            next_start = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(abstract)
            content = abstract[end:next_start].strip()
            if content:
                sections[name] = content

        return sections

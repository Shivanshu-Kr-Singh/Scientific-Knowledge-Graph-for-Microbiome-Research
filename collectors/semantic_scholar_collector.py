"""
collectors/semantic_scholar_collector.py
-----------------------------------------
Fetches papers from Semantic Scholar.

WHY SEMANTIC SCHOLAR?
  1. Citation counts and reference lists — essential for building the
     citation network in the knowledge graph.
  2. Paper "influence" scores — useful for ranking papers by importance.
  3. Covers CS + AI venues that PubMed misses (e.g. NeurIPS papers on
     microbiome ML).
  4. Free API with generous rate limits for registered users.

API DOCS: https://api.semanticscholar.org/graph/v1/
Registration: https://www.semanticscholar.org/product/api
"""

from typing import Optional
from loguru import logger

from config import SEMANTIC_SCHOLAR_API_KEY
from models import PaperRecord
from collectors.base_collector import BaseCollector


S2_BASE = "https://api.semanticscholar.org/graph/v1"

# Fields we want from Semantic Scholar.
# Listing only what you need reduces response size and speeds up queries.
S2_FIELDS = ",".join([
    "paperId",
    "externalIds",         # Contains DOI, PubMed ID, PMCID, ArXiv ID
    "title",
    "abstract",
    "year",
    "publicationDate",
    "authors",
    "venue",               # Journal / conference name
    "journal",
    "publicationTypes",
    "openAccessPdf",       # Direct PDF link if available
    "citationCount",
    "referenceCount",
    "fieldsOfStudy",       # ["Biology", "Medicine"] etc.
    "isOpenAccess",
])


class SemanticScholarCollector(BaseCollector):
    """Collects papers from Semantic Scholar via their Graph API."""

    source_name = "semantic_scholar"

    def __init__(self):
        super().__init__()
        if SEMANTIC_SCHOLAR_API_KEY:
            # With an API key you get 100 req/min instead of 1 req/sec
            self.session.headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
            logger.info("[semantic_scholar] API key loaded")
        else:
            logger.warning("[semantic_scholar] No API key — rate limited to ~1 req/sec")

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Semantic Scholar's keyword search. Less expressive than PubMed MeSH,
        but the citation data it provides is worth it.
        """
        year_from = date_from[:4]
        year_to   = date_to[:4]
        return {
            "query":    f"human microbiome metagenomics microbiota",
            "year":     f"{year_from}-{year_to}",   # Semantic Scholar date filter
        }

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        Uses Semantic Scholar's /paper/search endpoint.
        Offset-based pagination: offset = page * page_size.
        """
        params = {
            "query":  query_params["query"],
            "fields": S2_FIELDS,
            "limit":  page_size,
            "offset": page * page_size,
        }
        if query_params.get("year"):
            params["year"] = query_params["year"]

        response = self._get(f"{S2_BASE}/paper/search", params=params)
        data = response.json()

        self._save_raw(f"page_{page}", data)
        return data

    def _extract_items(self, raw_page: dict) -> list:
        """Semantic Scholar returns results under 'data' key."""
        return raw_page.get("data", [])

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one Semantic Scholar paper JSON record.

        SEMANTIC SCHOLAR JSON STRUCTURE:
          {
            "paperId": "a1b2c3...",
            "externalIds": {
              "DOI": "10.1038/...",
              "PubMed": "38765432",
              "ArXiv": "2024.12345"
            },
            "title": "...",
            "abstract": "...",
            "year": 2024,
            "publicationDate": "2024-03-15",
            "authors": [{"authorId": "...", "name": "John Smith"}],
            "venue": "Nature",
            "journal": {"name": "Nature", "volume": "625", "pages": "1-10"},
            "publicationTypes": ["JournalArticle"],
            "openAccessPdf": {"url": "https://...", "status": "GREEN"},
            "citationCount": 42,
            "referenceCount": 87,
            "isOpenAccess": true
          }
        """
        try:
            title = (raw.get("title") or "").strip()
            if not title:
                return None

            # ── External IDs ──────────────────────────────────────────────────
            ext_ids = raw.get("externalIds", {}) or {}
            doi     = ext_ids.get("DOI")
            pmid    = ext_ids.get("PubMed")
            pmcid   = ext_ids.get("PubMedCentral")
            arxiv   = ext_ids.get("ArXiv")

            # ── Authors ───────────────────────────────────────────────────────
            authors = [
                a["name"] for a in (raw.get("authors") or [])
                if a.get("name")
            ]

            # ── Journal ───────────────────────────────────────────────────────
            journal_data = raw.get("journal") or {}
            journal  = journal_data.get("name") or raw.get("venue")
            volume   = journal_data.get("volume")
            pages    = journal_data.get("pages")

            # ── Open Access PDF ───────────────────────────────────────────────
            oa_pdf = raw.get("openAccessPdf") or {}
            pdf_url = oa_pdf.get("url")

            # ── Article Types ─────────────────────────────────────────────────
            pub_types = raw.get("publicationTypes") or []

            return PaperRecord(
                pmid=str(pmid) if pmid else None,
                pmcid=str(pmcid) if pmcid else None,
                doi=doi,
                arxiv_id=arxiv,
                title=title,
                abstract=raw.get("abstract"),
                authors=authors,
                journal=journal,
                publication_date=raw.get("publicationDate"),
                publication_year=raw.get("year"),
                volume=str(volume) if volume else None,
                pages=str(pages) if pages else None,
                article_types=pub_types,
                is_open_access=raw.get("isOpenAccess", False),
                pdf_url=pdf_url,
                citation_count=raw.get("citationCount"),
                reference_count=raw.get("referenceCount"),
                is_preprint=bool(arxiv),   # If it has an ArXiv ID, it's likely a preprint
            )

        except Exception as e:
            logger.warning(f"[semantic_scholar] Failed to parse record: {e}")
            return None

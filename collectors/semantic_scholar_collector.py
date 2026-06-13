"""
collectors/semantic_scholar_collector.py
-----------------------------------------
Fetches papers from Semantic Scholar using the correct bulk search endpoint.

OFFICIAL API: https://api.semanticscholar.org/graph/v1/paper/search/bulk

KEY DIFFERENCES FROM RELEVANCE SEARCH:
  - Endpoint: /paper/search/bulk  (NOT /paper/search)
  - Pagination: token-based (response returns a 'token' for next page)
    NOT offset-based
  - Recommended for bulk data retrieval (more efficient, higher limits)
  - Supports sorting by paperId, publicationDate, citationCount
  - Supports publicationDateOrYear filter for date ranges

API KEY:
  Sent as header: x-api-key: <your_key>
  Rate limit with key: 1 request/second across all endpoints

DOCS: https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_bulk_search
"""

from typing import Optional, List
from loguru import logger

from config import SEMANTIC_SCHOLAR_API_KEY
from models import PaperRecord
from collectors.base_collector import BaseCollector


S2_BASE = "https://api.semanticscholar.org/graph/v1"

# Fields to request — only what we need (fewer fields = faster response)
S2_FIELDS = ",".join([
    "paperId",
    "externalIds",        # Contains DOI, PubMed, ArXiv IDs
    "title",
    "abstract",
    "year",
    "publicationDate",
    "authors",
    "venue",
    "journal",
    "publicationTypes",
    "openAccessPdf",
    "citationCount",
    "referenceCount",
    "isOpenAccess",
])


class SemanticScholarCollector(BaseCollector):
    """
    Collects papers from Semantic Scholar via the bulk search endpoint.
    Uses token-based pagination as documented in the official tutorial.
    """

    source_name = "semantic_scholar"

    def __init__(self):
        super().__init__()
        if SEMANTIC_SCHOLAR_API_KEY:
            # API key sent as x-api-key header (as per official tutorial)
            self.session.headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
            logger.info("[semantic_scholar] API key loaded — 1 req/sec rate limit | 1000 papers/request")
        else:
            logger.warning(
                "[semantic_scholar] No API key — sharing rate limit with all "
                "unauthenticated users (much slower). "
                "Get a free key at: https://www.semanticscholar.org/product/api"
            )

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Builds query params for the bulk search endpoint.

        YEAR FORMAT (from official tutorial):
          "2023-"       → 2023 and later
          "2024-2026"   → between 2024 and 2026 inclusive
          "-2023"       → up to 2023

        QUERY SYNTAX (bulk search supports special operators):
          "exact phrase"   → exact match
          term1 +term2     → must contain term2
          term1 -term2     → must not contain term2
          term*            → prefix match
        """
        year_from = date_from[:4]   # "2024/01/01" → "2024"
        year_to   = date_to[:4]     # "2026/12/31" → "2026"

        return {
            "query":    "human microbiome metagenomics microbiota",
            "year":     f"{year_from}-{year_to}",    # e.g. "2024-2026"
            "fields":   S2_FIELDS,
            # Filter to biology/medicine fields only
            "fieldsOfStudy": "Biology,Medicine",
        }

    def collect(
        self,
        query: str,
        date_from: str,
        date_to: str,
        max_results: int = 500,
        page_size: int = 1000,   # Bulk endpoint max is 1000 per request
        start_offset: int = 0,   # ignored — S2 uses token pagination; cursor stored separately
    ) -> List[PaperRecord]:
        """
        Override base collect() to use token-based pagination correctly.

        HOW TOKEN PAGINATION WORKS (from official tutorial):
          1. Send first request — no token
          2. Response contains: {total, token, data: [...]}
          3. If 'token' is in response → more pages exist
          4. Include token in next request to get next page
          5. Stop when no token in response

        NOTE ON start_offset:
          Semantic Scholar uses opaque continuation tokens, not numeric offsets.
          The cursor for S2 is stored separately as "semantic_scholar_token" in
          the cursor file. start_offset is accepted for interface compatibility
          but is not used here.
        """
        logger.info(
            f"[semantic_scholar] Starting bulk collection | "
            f"query='{query}' | {date_from} → {date_to}"
        )

        query_params = self.build_query(query, date_from, date_to)
        url = f"{S2_BASE}/paper/search/bulk"

        import hashlib, datetime as dt

        # Request exactly as many as needed per page, capped at API max of 1000
        per_page = min(max_results, page_size, 1000)

        params = {
            "query":  query_params["query"],
            "fields": query_params["fields"],
            "year":   query_params["year"],
            "sort":   "publicationDate:desc",
            "limit":  per_page,
        }

        papers: List[PaperRecord] = []
        page = 0

        # Resume from a previously saved continuation token if available.
        # The orchestrator stores this as "semantic_scholar_token" in the cursor file.
        resume_token = getattr(self, "_resume_token", None)
        token = resume_token
        if resume_token:
            logger.info(f"[semantic_scholar] Resuming from saved continuation token")
        self._last_token = None  # will be set to the final token after collection

        while len(papers) < max_results:
            if token:
                params["token"] = token
            else:
                # Recalculate exactly how many we still need, capped at page_size
                params["limit"] = min(max_results - len(papers), page_size, 1000)

            try:
                self._wait_for_rate_limit()
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 429:
                    import time
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"[semantic_scholar] Rate limited. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                if response.status_code == 403:
                    logger.error(
                        "[semantic_scholar] 403 Forbidden.\n"
                        "  1. Key not activated yet (wait 30-60 min)\n"
                        "  2. SEMANTIC_SCHOLAR_API_KEY not set in .env\n"
                        "  3. Header should be x-api-key"
                    )
                    break

                response.raise_for_status()
                data = response.json()

            except Exception as e:
                logger.error(f"[semantic_scholar] Request failed: {e}")
                break

            if page == 0:
                total = data.get("total", "unknown")
                logger.info(f"[semantic_scholar] Total results available: {total}")

            batch = data.get("data", [])
            if not batch:
                logger.info("[semantic_scholar] Empty batch — end of results")
                break

            for raw in batch:
                if len(papers) >= max_results:
                    break                   # hard stop — never exceed MAX_PER_SOURCE
                paper = self.parse_record(raw)
                if paper:
                    paper.source = self.source_name
                    content = f"{paper.title}|{paper.abstract}|{','.join(paper.authors)}"
                    paper.content_hash = hashlib.md5(content.encode()).hexdigest()
                    paper.fetched_at = dt.datetime.utcnow().isoformat()
                    papers.append(paper)

            logger.info(
                f"[semantic_scholar] Page {page}: {len(batch)} records | "
                f"Total so far: {len(papers)}"
            )

            # Save raw response
            self._save_raw(f"bulk_page_{page}", {"page": page, "count": len(batch)})

            # Check for next page token
            token = data.get("token")
            self._last_token = token  # persist so orchestrator can save it
            if not token:
                logger.info("[semantic_scholar] No more pages (no token in response)")
                break

            page += 1

        logger.success(f"[semantic_scholar] Collection complete: {len(papers)} papers")
        return papers

    # ── Required interface methods (not used in overridden collect()) ──────────

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """Not used — overridden by collect() above. Required by base class."""
        return {"data": []}

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one Semantic Scholar bulk search result.

        RESPONSE FORMAT:
          {
            "paperId": "abc123",
            "externalIds": {"DOI": "10.1038/...", "PubMed": "38765432"},
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

            # External IDs
            ext_ids = raw.get("externalIds") or {}
            doi   = ext_ids.get("DOI")
            pmid  = ext_ids.get("PubMed")
            pmcid = ext_ids.get("PubMedCentral")
            arxiv = ext_ids.get("ArXiv")

            # Authors
            authors = [
                a["name"] for a in (raw.get("authors") or [])
                if a.get("name")
            ]

            # Journal
            journal_data = raw.get("journal") or {}
            journal = journal_data.get("name") or raw.get("venue")
            volume  = journal_data.get("volume")
            pages   = journal_data.get("pages")

            # PDF
            oa_pdf  = raw.get("openAccessPdf") or {}
            pdf_url = oa_pdf.get("url")

            # Article types
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
                is_open_access=bool(raw.get("isOpenAccess")),
                pdf_url=pdf_url,
                citation_count=raw.get("citationCount"),
                reference_count=raw.get("referenceCount"),
                is_preprint=bool(arxiv),
            )

        except Exception as e:
            logger.warning(f"[semantic_scholar] Parse error: {e}")
            return None

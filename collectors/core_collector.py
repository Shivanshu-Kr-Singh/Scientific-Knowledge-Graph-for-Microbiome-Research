"""
collectors/core_collector.py
------------------------------
Fetches open-access papers from CORE — the world's largest aggregator
of open access research.

WHY CORE?
  CORE aggregates full-text content from 10,000+ repositories and journals.
  Key advantages over our existing collectors:

  1. FULL TEXT: Unlike PubMed/Crossref which give abstracts, CORE provides
     the actual parsed full-text when available (via the 'fullText' field).
     This directly feeds Layer 2 NLP — extracting from Methods, Results,
     and Data Availability sections, not just abstracts.

  2. OPEN ACCESS COVERAGE: 200M+ open access papers. Covers institutional
     repositories, disciplinary repos, preprint servers, journals.

  3. DOWNLOAD LINKS: Direct PDF download URLs via 'downloadUrl'.

  4. SDG CLASSIFICATION: CORE auto-tags papers with UN Sustainable
     Development Goals — useful metadata for thematic analysis.

  5. REPOSITORY METADATA: Which institution deposited the paper.

API DOCS: https://api.core.ac.uk/docs/v3
Base URL:  https://api.core.ac.uk/v3
Endpoint:  POST /search/works   (deduplicated, enriched records)

Rate limits:
  No key (unauthenticated): 100 tokens/day, 10/min ← very limited
  Free registered key:      1,000 tokens/day, 25/min ← acceptable
  Academic key:             5,000 tokens/day, 10/min

HOW TO GET A FREE API KEY:
  Register at: https://core.ac.uk/services/api
  Set CORE_API_KEY in .env

IMPORTANT: We use the /search/works endpoint (not /search/outputs).
  - Works = deduplicated, enriched, harmonised records (like PubMed records)
  - Outputs = raw repository-specific records (not deduplicated)
  Works is the right choice for our use case.
"""

import os
import time
from datetime import datetime
from typing import List, Optional
from loguru import logger

from models import PaperRecord
from collectors.base_collector import BaseCollector

CORE_BASE    = "https://api.core.ac.uk/v3"
CORE_API_KEY = os.getenv("CORE_API_KEY", "")


class CoreCollector(BaseCollector):
    """
    Collects open-access papers from CORE using the /search/works endpoint.

    Uses POST-based search with offset pagination:
      POST /v3/search/works
      Body: {"q": "...", "limit": 100, "offset": 0, "sort": "..."}
    """

    source_name = "core"

    def __init__(self):
        super().__init__()
        if CORE_API_KEY:
            self.session.headers["Authorization"] = f"Bearer {CORE_API_KEY}"
            logger.info("[core] API key loaded — up to 5,000 tokens/day")
        else:
            logger.warning(
                "[core] No CORE_API_KEY in .env — limited to 100 tokens/day (10/min). "
                "Get a free key at: https://core.ac.uk/services/api"
            )

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Builds the CORE search query body for /search/works.

        CORE QUERY SYNTAX (Elasticsearch-style):
          title:"human microbiome"       → exact phrase in title
          abstract:"microbiota"          → in abstract
          yearPublished>=2024            → year range
          _exists_:fullText              → has full text
          AND / OR operators supported

        STRATEGY:
          Use a targeted microbiome query with year filter and require
          full text to be present (so we get the most value from CORE's
          unique capability).
        """
        year_from = int(date_from[:4])
        year_to   = int(date_to[:4])

        # Target human microbiome papers with full text available
        # yearPublished range ensures we stay within our study window
        search_q = (
            f'(title:"microbiome" OR title:"microbiota" OR '
            f'title:"gut microbiome" OR title:"human microbiome" OR '
            f'abstract:"human microbiome" OR abstract:"human microbiota") '
            f'AND yearPublished>={year_from} AND yearPublished<={year_to}'
        )

        return {
            "q":         search_q,
            "year_from": year_from,
            "year_to":   year_to,
        }

    def collect(
        self,
        query: str,
        date_from: str,
        date_to: str,
        max_results: int = 500,
        page_size: int = 100,
        start_offset: int = 0,
    ) -> List[PaperRecord]:
        """
        Override to use POST-based pagination for CORE /search/works.

        CORE /search/works pagination:
          POST body: {"q": "...", "limit": 100, "offset": 0, ...}
          Response:  {"totalHits": N, "results": [...]}

        We can't use the base collect() since that calls GET via fetch_page.
        CORE requires POST for search, so we manage the loop here directly.
        """
        logger.info(
            f"[{self.source_name}] Starting collection | "
            f"{date_from} → {date_to} | "
            f"full-text open access papers"
        )
        if start_offset > 0:
            logger.info(
                f"[{self.source_name}] Resuming from offset {start_offset}"
            )

        query_params = self.build_query(query, date_from, date_to)
        papers: List[PaperRecord] = []
        offset = start_offset

        while len(papers) < max_results:
            limit = min(page_size, max_results - len(papers), 100)  # CORE max=100

            payload = {
                "q":      query_params["q"],
                "limit":  limit,
                "offset": offset,
                "sort":   "yearPublished:desc",
            }

            self._wait_for_rate_limit()

            try:
                response = self.session.post(
                    f"{CORE_BASE}/search/works",
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(
                        f"[core] Rate limited. Waiting {retry_after}s. "
                        f"Consider getting a free API key at https://core.ac.uk/services/api"
                    )
                    time.sleep(retry_after)
                    continue

                if response.status_code == 401:
                    logger.error(
                        "[core] 401 Unauthorized — CORE_API_KEY is invalid or missing. "
                        "Get a free key at https://core.ac.uk/services/api"
                    )
                    break

                response.raise_for_status()
                data = response.json()

            except Exception as e:
                logger.error(f"[core] Request failed at offset {offset}: {e}")
                break

            # Log total on first call
            if offset == start_offset:
                total_hits = data.get("totalHits", "unknown")
                logger.info(
                    f"[core] Total results available: {total_hits} | "
                    f"collecting up to {max_results}"
                )

            results = data.get("results", [])
            if not results:
                logger.info(f"[core] No more results at offset {offset}")
                break

            batch_count = 0
            for raw in results:
                if len(papers) >= max_results:
                    break
                paper = self.parse_record(raw)
                if paper:
                    paper.source       = self.source_name
                    paper.content_hash = self._compute_hash(paper)
                    paper.fetched_at   = datetime.utcnow().isoformat()
                    papers.append(paper)
                    batch_count += 1

            logger.info(
                f"[core] Offset {offset}: {batch_count} records | "
                f"Total so far: {len(papers)}"
            )

            self._save_raw(f"offset_{offset}", {
                "totalHits": data.get("totalHits"),
                "offset":    offset,
                "count":     batch_count,
            })

            # If we got fewer than limit, we've exhausted results
            if len(results) < limit:
                logger.info(f"[core] Reached end of results at offset {offset}")
                break

            offset += limit

        logger.success(f"[core] Collection complete: {len(papers)} papers")
        return papers

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """Build query params — defined above, repeated here to satisfy BaseCollector."""
        year_from = int(date_from[:4])
        year_to   = int(date_to[:4])
        search_q = (
            f'(title:"microbiome" OR title:"microbiota" OR '
            f'title:"gut microbiome" OR title:"human microbiome" OR '
            f'abstract:"human microbiome" OR abstract:"human microbiota") '
            f'AND yearPublished>={year_from} AND yearPublished<={year_to}'
        )
        return {"q": search_q, "year_from": year_from, "year_to": year_to}

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """Not used — overridden by collect() above. Required by BaseCollector."""
        return {"results": []}

    def _extract_items(self, raw_page: dict) -> list:
        return raw_page.get("results", [])

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one CORE /search/works result.

        CORE WORKS JSON STRUCTURE (key fields):
          {
            "id":            12345678,
            "doi":           "10.1038/...",
            "title":         "Paper title",
            "abstract":      "Abstract text...",
            "authors": [{"name": "Smith, John"}],
            "yearPublished": 2024,
            "publishedDate": "2024-03-15T00:00:00",
            "journals": [{"title": "Nature", "identifiers": ["issn:0028-0836"]}],
            "publisher":     {"name": "Springer Nature"},
            "downloadUrl":   "https://core.ac.uk/download/pdf/12345.pdf",
            "fullText":      "Full text of the paper...",  ← unique to CORE
            "isOpenAccess":  true,
            "citationCount": 42,
            "documentType":  "research article",
            "dataProviders": [{"name": "Open Research Online", ...}],
            "sourceFulltextUrls": ["https://..."],
            "links": [{"type": "download", "url": "https://..."}]
          }

        NOTE ON fullText:
          When present, this is the full parsed text from the PDF.
          We store it in the abstract field as a fallback (or could add
          a dedicated field later) — primarily useful for Layer 2 NLP.
        """
        try:
            title = (raw.get("title") or "").strip()
            if not title:
                return None

            doi = (raw.get("doi") or "").strip() or None

            # ── Abstract (prefer abstract, fall back to beginning of fullText) ──
            abstract = (raw.get("abstract") or "").strip() or None
            if not abstract and raw.get("fullText"):
                # Take first 2000 chars of full text as a proxy abstract
                abstract = raw["fullText"][:2000].strip()

            # ── Authors ────────────────────────────────────────────────────────
            authors = []
            for author in (raw.get("authors") or []):
                name = (author.get("name") or "").strip()
                if name:
                    authors.append(name)

            # ── Journal ────────────────────────────────────────────────────────
            journals = raw.get("journals") or []
            journal  = journals[0].get("title") if journals else None

            # Extract ISSN from journal identifiers
            issn = None
            if journals:
                for identifier in (journals[0].get("identifiers") or []):
                    if str(identifier).startswith("issn:"):
                        issn = str(identifier).replace("issn:", "")
                        break

            # ── Publisher ──────────────────────────────────────────────────────
            publisher_obj = raw.get("publisher") or {}
            # publisher name stored in keywords as metadata — not a PaperRecord field

            # ── Date ───────────────────────────────────────────────────────────
            pub_year = raw.get("yearPublished")
            pub_date = raw.get("publishedDate")
            if pub_date:
                # Normalize "2024-03-15T00:00:00" → "2024-03-15"
                pub_date = pub_date[:10]
            elif pub_year:
                pub_date = str(pub_year)

            # ── Full text / PDF ────────────────────────────────────────────────
            download_url = raw.get("downloadUrl")

            # Also check links array for download URL
            if not download_url:
                for link in (raw.get("links") or []):
                    if link.get("type") == "download":
                        download_url = link.get("url")
                        break

            # Source fulltext URLs as fallback
            source_urls = raw.get("sourceFulltextUrls") or []
            full_text_url = download_url or (source_urls[0] if source_urls else None)

            # ── Open Access ────────────────────────────────────────────────────
            # All CORE papers are open access by definition, but flag explicitly
            is_oa = bool(raw.get("isOpenAccess", True))

            # ── Document type ──────────────────────────────────────────────────
            doc_type = raw.get("documentType") or "research"
            article_types = [doc_type] if doc_type else ["research"]

            return PaperRecord(
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                issn=issn,
                publication_date=pub_date,
                publication_year=pub_year,
                article_types=article_types,
                is_open_access=is_oa,
                full_text_url=full_text_url,
                pdf_url=download_url,
                citation_count=raw.get("citationCount"),
                is_preprint=False,
            )

        except Exception as e:
            logger.warning(f"[core] Failed to parse record: {e}")
            return None

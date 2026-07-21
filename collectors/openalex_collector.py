"""
collectors/openalex_collector.py
---------------------------------
Fetches papers from OpenAlex — the open, comprehensive academic graph.

WHY OPENALEX?
  OpenAlex indexes 250M+ scholarly works and is rapidly becoming the
  de-facto replacement for Semantic Scholar as a general academic graph.
  Key advantages over our existing collectors:

  1. SCALE: 250M works vs PubMed's ~36M. Covers books, datasets, conference
     papers and grey literature that PubMed misses.

  2. CONCEPTS & TOPICS: Each paper is tagged with a concept hierarchy
     (e.g. Microbiota → Gastroenterology → Medicine). Perfect for our
     relevance filter — we can filter by concept ID, not just keywords.

  3. CITATION GRAPH: Provides cited_by_count AND referenced_works list,
     enabling forward/backward citation traversal for the KG.

  4. INSTITUTIONS & FUNDING: Author affiliations, country, funder — useful
     for Layer 3 graph nodes.

  5. OPEN ACCESS: Returns best_oa_location with a direct PDF URL when
     available — feeds directly into our full-text pipeline.

  6. FREE: No API key required. Polite pool: 10 req/sec. With email param
     in User-Agent (polite pool): 100 req/sec.

API DOCS: https://docs.openalex.org/api-entities/works/search-works
Base URL:  https://api.openalex.org/works
"""

from typing import Optional, List
from loguru import logger

from models import PaperRecord
from collectors.base_collector import BaseCollector
from config import NCBI_EMAIL  # Reuse email for polite pool identification

OPENALEX_BASE = "https://api.openalex.org"

# OpenAlex concept IDs for human microbiome research.
# Using concept IDs is more precise than keyword search — they're stable
# identifiers in OpenAlex's controlled vocabulary hierarchy.
#
# How to find concept IDs:
#   https://api.openalex.org/concepts?search=human+microbiome
#
MICROBIOME_CONCEPTS = [
    "C2778793"  ,  # Human microbiome (most specific)
    "C185592680",  # Gut microbiota
    "C2776943"  ,  # Microbiota
    "C2781022"  ,  # Metagenomics
    "C2781029"  ,  # 16S rRNA
]


class OpenAlexCollector(BaseCollector):
    """
    Collects papers from OpenAlex using concept-filtered search.

    Uses cursor-based pagination (same principle as Semantic Scholar's token
    pagination) — each response contains a next_cursor for the following page.
    """

    source_name = "openalex"

    def __init__(self):
        super().__init__()
        # OpenAlex polite pool: include email in User-Agent or mailto param
        # This bumps rate limit from 10 req/sec to 100 req/sec
        self._polite_email = NCBI_EMAIL
        logger.info(
            f"[openalex] Collector initialized | "
            f"polite pool email: {self._polite_email}"
        )

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Builds OpenAlex filter params.

        FILTER STRATEGY (two-layer, same approach as EuropePMC):
          Layer A — Concept filter: require at least one microbiome concept tag.
          Layer B — Date filter: publication_year range from the query.
        """
        year_from = date_from[:4]
        year_to   = date_to[:4]
        concept_filter = "|".join(MICROBIOME_CONCEPTS)
        return {
            "concept_filter": concept_filter,
            "year_from":      year_from,
            "year_to":        year_to,
            "query":          query,
        }

    def collect(
        self,
        query: str,
        date_from: str,
        date_to: str,
        max_results: int = 500,
        page_size: int = 200,    # OpenAlex max per page is 200
        start_offset: int = 0,
    ) -> List[PaperRecord]:
        """
        Override base collect() to use OpenAlex cursor-based pagination.

        CURSOR PERSISTENCE ACROSS RUNS:
          OpenAlex uses opaque cursor strings, not numeric offsets.
          We save the last cursor string in the cursor file as a special
          key "openalex_cursor". start_offset is used only to detect
          "fresh start vs resume" — 0 = fresh, >0 = check for saved cursor.
        """
        import hashlib
        import datetime as dt

        logger.info(
            f"[{self.source_name}] Starting collection | "
            f"query='{query}' | {date_from} → {date_to} | max={max_results}"
        )

        query_params = self.build_query(query, date_from, date_to)

        filters = [
            f"concepts.id:{query_params['concept_filter']}",
            f"publication_year:{query_params['year_from']}-{query_params['year_to']}",
            "type:article",
        ]

        base_params = {
            "filter":   ",".join(filters),
            "per_page": min(page_size, 200),
            "sort":     "publication_date:desc",
            "mailto":   self._polite_email,
            "select":   ",".join([
                "id", "doi", "ids", "title", "abstract_inverted_index",
                "authorships", "publication_date", "publication_year",
                "primary_location", "open_access", "cited_by_count",
                "referenced_works_count", "type", "concepts",
                "best_oa_location",
            ]),
        }

        # Use saved cursor string if resuming, otherwise start fresh
        saved_cursor = getattr(self, "_resume_cursor", None)
        cursor = saved_cursor if (start_offset > 0 and saved_cursor) else "*"
        if cursor != "*":
            logger.info(f"[openalex] Resuming from saved cursor")

        papers: List[PaperRecord] = []
        page = 0
        self._last_cursor = None   # will be updated after each page

        while len(papers) < max_results:
            params = {**base_params, "cursor": cursor}

            try:
                self._wait_for_rate_limit()
                response = self.session.get(
                    f"{OPENALEX_BASE}/works", params=params, timeout=30
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                logger.error(f"[openalex] Request failed on page {page}: {e}")
                break

            if page == 0:
                total = data.get("meta", {}).get("count", "unknown")
                logger.info(f"[openalex] Total results available: {total}")

            results = data.get("results", [])
            if not results:
                logger.info(f"[openalex] No more results at page {page}")
                break

            batch_count = 0
            for raw in results:
                if len(papers) >= max_results:
                    break
                paper = self.parse_record(raw)
                if paper:
                    paper.source       = self.source_name
                    paper.content_hash = self._compute_hash(paper)
                    paper.fetched_at   = dt.datetime.utcnow().isoformat()
                    papers.append(paper)
                    batch_count += 1

            logger.info(
                f"[openalex] Page {page}: {batch_count} records | "
                f"Total so far: {len(papers)}"
            )

            # Advance cursor and save it for cross-run persistence
            cursor = data.get("meta", {}).get("next_cursor")
            self._last_cursor = cursor   # orchestrator saves this to cursor file

            if not cursor:
                logger.info("[openalex] Cursor exhausted — no more pages")
                break

            page += 1

        logger.success(f"[openalex] Collection complete: {len(papers)} papers")
        return papers

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        Fetches one page from OpenAlex using cursor-based pagination.

        CURSOR PAGINATION:
          First request: cursor=*  (wildcard = start from beginning)
          Subsequent: cursor=<value from previous response's meta.next_cursor>
          End of results: next_cursor is null

        We cache the cursor on the instance so consecutive pages chain correctly.
        On page=0 we reset to '*' (start over), otherwise use stored cursor.
        """
        if page == 0:
            self._next_cursor = "*"

        # Build filter string
        filters = [
            f"concepts.id:{query_params['concept_filter']}",
            f"publication_year:{query_params['year_from']}-{query_params['year_to']}",
            "type:article",          # Exclude books, datasets, etc.
        ]

        params = {
            "filter":       ",".join(filters),
            "per_page":     min(page_size, 200),   # OpenAlex max is 200 per page
            "cursor":       getattr(self, "_next_cursor", "*"),
            "sort":         "publication_date:desc",
            # Polite pool identification
            "mailto":       self._polite_email,
            # Select only fields we need — reduces response size significantly
            "select": ",".join([
                "id", "doi", "ids", "title", "abstract_inverted_index",
                "authorships", "publication_date", "publication_year",
                "primary_location", "open_access", "cited_by_count",
                "referenced_works_count", "type", "concepts",
                "best_oa_location",
            ]),
        }

        response = self._get(f"{OPENALEX_BASE}/works", params=params)
        data = response.json()

        # Log total on first page
        if page == 0:
            total = data.get("meta", {}).get("count", "unknown")
            logger.info(f"[openalex] Total results available: {total}")

        # Store cursor for next page
        self._next_cursor = data.get("meta", {}).get("next_cursor")
        if not self._next_cursor:
            logger.info("[openalex] No more pages (cursor exhausted)")

        self._save_raw(f"page_{page}", {
            "count":  data.get("meta", {}).get("count"),
            "cursor": self._next_cursor,
            "page":   page,
        })

        return {"results": data.get("results", []), "meta": data.get("meta", {})}

    def _extract_items(self, raw_page: dict) -> list:
        return raw_page.get("results", [])

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one OpenAlex work object into a PaperRecord.

        OPENALEX WORK STRUCTURE:
          {
            "id":        "https://openalex.org/W2741809807",
            "doi":       "https://doi.org/10.1038/...",
            "ids":       {
              "openalex": "https://openalex.org/W2741809807",
              "doi":      "https://doi.org/10.1038/...",
              "pmid":     "https://pubmed.ncbi.nlm.nih.gov/17375194"
            },
            "title":     "...",
            "abstract_inverted_index": {"word": [pos, ...], ...},
            "authorships": [
              {
                "author": {"display_name": "John Smith"},
                "institutions": [{"display_name": "MIT", "country_code": "US"}]
              }
            ],
            "publication_date": "2024-03-15",
            "publication_year": 2024,
            "primary_location": {
              "source": {"display_name": "Nature", "issn_l": "0028-0836"},
              "is_oa": true
            },
            "open_access": {"is_oa": true, "oa_url": "https://..."},
            "cited_by_count": 42,
            "referenced_works_count": 87,
            "type": "article",
            "concepts": [
              {"id": "C2778793", "display_name": "Human microbiome", "score": 0.9}
            ],
            "best_oa_location": {"pdf_url": "https://..."}
          }

        NOTE ON ABSTRACT:
          OpenAlex stores abstracts as an inverted index (word → [positions]).
          We reconstruct the abstract by sorting words by their first position.
          This is lossless — the original abstract is fully recoverable.

        NOTE ON PMID:
          OpenAlex returns a PMID (when the work is indexed in PubMed) inside
          the `ids` object, at zero extra query cost. Previously this field
          was requested via `select` but never parsed, meaning every OpenAlex
          paper always had pmid=None regardless of whether OpenAlex actually
          had it — silently blocking FullTextOrchestrator's Tier 3 NCBI
          abstract fallback for every paper from this collector. Verified
          directly against the live API that "ids.pmid" is populated as a
          full PubMed URL (e.g. "https://pubmed.ncbi.nlm.nih.gov/17375194"),
          not a bare ID — stripped down to the numeric ID below.
        """
        try:
            title = (raw.get("title") or "").strip()
            if not title:
                return None

            # ── DOI ───────────────────────────────────────────────────────────
            doi_raw = raw.get("doi") or ""
            # OpenAlex includes "https://doi.org/" prefix — strip it.
            # In rare cases OpenAlex returns unparsed HTML anchor tags in the
            # doi field (e.g. '10.xxxx/yyy">10.xxxx/yyy</a></p'). Strip all
            # HTML before the first quote or angle bracket to recover the clean
            # DOI. Example broken value:
            #   '10.1016/j.jnucmat.2026.156769">10.1016/j.jnucmat.2026.156769</a></p'
            import re as _re
            doi_clean = doi_raw.replace("https://doi.org/", "")
            # If there is any HTML markup, take only the substring before it
            doi_clean = _re.split(r'["\'>< ]', doi_clean)[0].strip()
            doi = doi_clean or None

            # ── PMID (from the "ids" object, free — no extra API call) ────────
            ids_obj  = raw.get("ids") or {}
            pmid_raw = ids_obj.get("pmid") or ""
            pmid     = pmid_raw.replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip() or None

            # ── Abstract reconstruction ───────────────────────────────────────
            abstract = self._reconstruct_abstract(
                raw.get("abstract_inverted_index") or {}
            )

            # ── Authors ───────────────────────────────────────────────────────
            authors = []
            for authorship in (raw.get("authorships") or []):
                name = (authorship.get("author") or {}).get("display_name")
                if name:
                    authors.append(name)

            # ── Journal ───────────────────────────────────────────────────────
            primary = raw.get("primary_location") or {}
            source  = primary.get("source") or {}
            journal = source.get("display_name")
            issn    = source.get("issn_l")

            # ── Open Access ───────────────────────────────────────────────────
            oa_info  = raw.get("open_access") or {}
            is_oa    = bool(oa_info.get("is_oa"))
            oa_url   = oa_info.get("oa_url")

            best_oa  = raw.get("best_oa_location") or {}
            pdf_url  = best_oa.get("pdf_url")

            # ── Concepts as keywords ──────────────────────────────────────────
            # Use concept display names as keywords — they're human-readable
            # and more standardized than raw author keywords
            keywords = [
                c["display_name"]
                for c in (raw.get("concepts") or [])
                if c.get("score", 0) >= 0.3   # Only reasonably confident concepts
            ]

            return PaperRecord(
                doi=doi,
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                keywords=keywords,
                journal=journal,
                issn=issn,
                publication_date=raw.get("publication_date"),
                publication_year=raw.get("publication_year"),
                article_types=[raw.get("type", "article")],
                is_open_access=is_oa,
                full_text_url=oa_url,
                pdf_url=pdf_url,
                citation_count=raw.get("cited_by_count"),
                reference_count=raw.get("referenced_works_count"),
                is_preprint=False,
            )

        except Exception as e:
            logger.warning(f"[openalex] Failed to parse record: {e}")
            return None

    def _reconstruct_abstract(self, inverted_index: dict) -> Optional[str]:
        """
        Reconstructs a plain-text abstract from OpenAlex's inverted index format.

        OpenAlex stores: {"The": [0], "gut": [1, 15], "microbiome": [2], ...}
        We need: "The gut microbiome ..."

        Algorithm: flatten all (word, position) pairs, sort by position,
        join words with spaces.
        """
        if not inverted_index:
            return None
        try:
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            return " ".join(word for _, word in word_positions)
        except Exception:
            return None

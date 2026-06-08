"""
collectors/crossref_collector.py
---------------------------------
Fetches papers from Crossref — the DOI registration authority.

WHY CROSSREF?
  Crossref is where DOIs are born. Every paper with a DOI has a Crossref
  record, making it the most authoritative source for:

  1. DOI COMPLETENESS: Papers that PubMed indexed without a DOI often have
     one in Crossref. Critical for deduplication accuracy.

  2. JOURNAL METADATA: Publisher, ISSN, volume, issue, pages — Crossref
     has the canonical version of these fields.

  3. FUNDING INFO: Funder names and grant IDs — unique to Crossref among
     our collectors. Essential for Layer 3 funding graph nodes.

  4. LICENSE: CC-BY, CC-BY-NC etc. — tells us reuse rights for full text.

  5. REFERENCE LISTS: Structured reference data for citation graph building.

  6. COVERAGE: 150M+ DOI-registered works. Catches specialized journals
     that PubMed and OpenAlex may not fully index (food science, environmental
     journals publishing microbiome work).

API DOCS: https://api.crossref.org/swagger-ui/index.html
Base URL:  https://api.crossref.org/works
Rate limit: 50 req/sec (polite pool with User-Agent email)

NOTE ON SEARCH QUALITY:
  Crossref's full-text search is weaker than PubMed or OpenAlex — it only
  searches title and abstract. We compensate with a highly specific query
  and post-fetch relevance filtering.
"""

from typing import Optional
from loguru import logger

from models import PaperRecord
from collectors.base_collector import BaseCollector
from config import NCBI_EMAIL

CROSSREF_BASE = "https://api.crossref.org"


class CrossrefCollector(BaseCollector):
    """
    Collects papers from Crossref using the /works endpoint.
    Uses offset-based pagination (rows + offset params).
    """

    source_name = "crossref"

    def __init__(self):
        super().__init__()
        # Crossref polite pool: include email in User-Agent for 50 req/sec
        # vs anonymous 3 req/sec. Critical difference at scale.
        self.session.headers.update({
            "User-Agent": (
                f"MicrobiomeMiner/1.0 (Academic research; mailto:{NCBI_EMAIL})"
            )
        })
        logger.info(
            f"[crossref] Collector initialized | "
            f"polite pool via User-Agent mailto:{NCBI_EMAIL}"
        )

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Builds Crossref query params.

        CROSSREF QUERY SYNTAX:
          query=           → free-text search across title + abstract
          filter=          → structured filters (date, type, ISSN, etc.)

        FILTER STRATEGY:
          - from-pub-date / until-pub-date: publication date range
          - type:journal-article: exclude books, datasets, conference papers
            (we only want peer-reviewed articles)

        DATE FORMAT: Crossref uses YYYY-MM-DD for date filters.
        """
        year_from = date_from[:4]
        year_to   = date_to[:4]

        # Use highly specific query terms to improve precision
        # Crossref search is less sophisticated than PubMed so we need
        # specific phrases rather than broad terms
        search_query = (
            "human microbiome microbiota gut metagenomics 16S rRNA "
            "intestinal microbiome dysbiosis probiotics"
        )

        return {
            "query":      search_query,
            "date_from":  f"{year_from}-01-01",
            "date_to":    f"{year_to}-12-31",
        }

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        Fetches one page from Crossref /works endpoint.

        CROSSREF PAGINATION:
          Uses offset (rows + offset), not cursor.
          offset = page * page_size
          rows   = page_size (how many per page, max 1000)

        SORT: score (relevance) by default — best for our use case since
        we want the most relevant papers first.
        """
        offset = page * page_size

        params = {
            "query":           query_params["query"],
            "filter":          (
                f"from-pub-date:{query_params['date_from']},"
                f"until-pub-date:{query_params['date_to']},"
                f"type:journal-article"
            ),
            "rows":            min(page_size, 1000),  # Crossref max is 1000
            "offset":          offset,
            "sort":            "score",
            "order":           "desc",
            # Select only fields we need — dramatically reduces response size
            "select": ",".join([
                "DOI", "title", "abstract", "author", "published",
                "container-title", "ISSN", "volume", "issue", "page",
                "type", "is-referenced-by-count", "references-count",
                "license", "funder", "link",
            ]),
        }

        response = self._get(f"{CROSSREF_BASE}/works", params=params)
        data = response.json()

        # Log total on first page
        if page == 0:
            total = data.get("message", {}).get("total-results", "unknown")
            logger.info(f"[crossref] Total results available: {total}")

        self._save_raw(f"page_{page}", {
            "total":  data.get("message", {}).get("total-results"),
            "offset": offset,
            "page":   page,
        })

        return {"items": data.get("message", {}).get("items", [])}

    def _extract_items(self, raw_page: dict) -> list:
        return raw_page.get("items", [])

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one Crossref work object.

        CROSSREF WORK STRUCTURE:
          {
            "DOI":           "10.1038/s41586-024-07999-z",
            "title":         ["Paper title here"],      ← list!
            "abstract":      "<jats:p>Abstract text</jats:p>",
            "author": [
              {"given": "John", "family": "Smith", "affiliation": [...]}
            ],
            "published": {
              "date-parts": [[2024, 3, 15]]             ← nested list!
            },
            "container-title": ["Nature"],              ← list!
            "ISSN":            ["0028-0836"],            ← list!
            "volume":          "625",
            "issue":           "7994",
            "page":            "1-10",
            "type":            "journal-article",
            "is-referenced-by-count": 42,
            "references-count": 87,
            "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
            "funder": [{"name": "NIH", "award": ["R01HG000001"]}],
            "link": [{"URL": "https://...", "content-type": "application/pdf"}]
          }

        NOTE: Many fields in Crossref are arrays even when logically single-valued
        (title, container-title, ISSN). Always use [0] with a fallback.
        """
        try:
            # Title is a list in Crossref
            title_list = raw.get("title") or []
            title = title_list[0].strip() if title_list else ""
            if not title:
                return None

            doi = (raw.get("DOI") or "").strip() or None

            # ── Abstract ──────────────────────────────────────────────────────
            # Crossref abstracts are wrapped in JATS XML tags — strip them
            abstract_raw = raw.get("abstract") or ""
            abstract = self._strip_jats(abstract_raw) or None

            # ── Authors ───────────────────────────────────────────────────────
            authors = []
            for author in (raw.get("author") or []):
                given  = author.get("given", "").strip()
                family = author.get("family", "").strip()
                name   = author.get("name", "").strip()  # for org authors
                if family:
                    authors.append(f"{family}, {given}".strip(", "))
                elif name:
                    authors.append(name)

            # ── Journal ───────────────────────────────────────────────────────
            container = raw.get("container-title") or []
            journal   = container[0] if container else None
            issn_list = raw.get("ISSN") or []
            issn      = issn_list[0] if issn_list else None

            # ── Date ──────────────────────────────────────────────────────────
            pub_date_parts = (
                (raw.get("published") or {})
                .get("date-parts", [[]])[0]
            )
            pub_year = pub_date_parts[0] if pub_date_parts else None
            pub_date = None
            if pub_date_parts:
                parts = pub_date_parts
                year  = parts[0] if len(parts) > 0 else None
                month = str(parts[1]).zfill(2) if len(parts) > 1 else "01"
                day   = str(parts[2]).zfill(2) if len(parts) > 2 else "01"
                if year:
                    pub_date = f"{year}-{month}-{day}"

            # ── License / Open Access ─────────────────────────────────────────
            licenses = raw.get("license") or []
            is_oa = any(
                "creativecommons" in (lic.get("URL") or "").lower()
                for lic in licenses
            )

            # ── PDF URL ───────────────────────────────────────────────────────
            pdf_url = None
            for link in (raw.get("link") or []):
                if "pdf" in (link.get("content-type") or "").lower():
                    pdf_url = link.get("URL")
                    break

            # ── Funding ───────────────────────────────────────────────────────
            # Store funder names as keywords for now — Layer 3 will extract
            # these into proper funding graph nodes
            funders = raw.get("funder") or []
            funder_keywords = [
                f["name"] for f in funders if f.get("name")
            ]

            return PaperRecord(
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                keywords=funder_keywords,  # funders as searchable keywords
                journal=journal,
                issn=issn,
                publication_date=pub_date,
                publication_year=pub_year,
                volume=str(raw.get("volume") or "") or None,
                issue=str(raw.get("issue") or "") or None,
                pages=str(raw.get("page") or "") or None,
                article_types=[raw.get("type", "journal-article")],
                is_open_access=is_oa,
                pdf_url=pdf_url,
                citation_count=raw.get("is-referenced-by-count"),
                reference_count=raw.get("references-count"),
                is_preprint=False,
            )

        except Exception as e:
            logger.warning(f"[crossref] Failed to parse record: {e}")
            return None

    def _strip_jats(self, text: str) -> str:
        """
        Strips JATS XML tags from Crossref abstracts.

        Crossref wraps abstracts in JATS markup:
          <jats:p>The gut microbiome...</jats:p>
          <jats:sec><jats:title>Background</jats:title><jats:p>...</jats:p></jats:sec>

        We strip all tags and normalize whitespace to get plain text.
        """
        import re
        # Remove all XML/HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

"""
collectors/google_scholar_collector.py
----------------------------------------
Fetches papers from Google Scholar.

WHY GOOGLE SCHOLAR IS DIFFERENT FROM THE OTHER SOURCES:
  PubMed, Europe PMC, Semantic Scholar, and bioRxiv all have official REST
  APIs — they WANT you to query them programmatically. Google Scholar does NOT.
  It has no public API. Google actively detects and blocks scrapers because
  Scholar is a commercial product (tied to Google Ads and profile pages).

  This means we need a different strategy. We support TWO modes:

  MODE 1: scholarly (FREE)
    Uses the `scholarly` Python library which reverse-engineered Google
    Scholar's HTML. Works for small volumes (< 100 papers per session).
    Gets blocked after ~100 requests without a proxy. Best for development.

  MODE 2: SerpAPI (PAID, RELIABLE)
    SerpAPI is a commercial service that handles blocking, CAPTCHAs, and
    proxy rotation for you. Returns clean JSON. ~$50/month for 5000 searches.
    Sign up: https://serpapi.com/

WHY ADD GOOGLE SCHOLAR AT ALL?
  1. Indexes papers PubMed MISSES: conference papers, book chapters, theses.
  2. Has citation counts for ALL papers (not just indexed journals).
  3. Surfaces PDFs on university/ResearchGate pages for paywalled papers.
  4. Picks up interdisciplinary work from CS/ML venues.

ANTI-DETECTION STRATEGY (scholarly mode):
  Google detects bots via request frequency, user-agent, cookie patterns,
  and IP reputation. Our mitigations:
    - Random delays between requests (3-8 seconds, not fixed)
    - Optional ScraperAPI proxy (5000 free req/month at scraperapi.com)
    - Graceful fallback: if blocked, log and stop (don't crash)
"""

import os
import time
import random
import re
from typing import Optional, List
from loguru import logger

from models import PaperRecord
from collectors.base_collector import BaseCollector

_scholarly_module = None
_GoogleSearch = None


def _get_scholarly():
    global _scholarly_module
    if _scholarly_module is None:
        try:
            import scholarly as s
            _scholarly_module = s
        except ImportError:
            raise ImportError("Run: pip install scholarly")
    return _scholarly_module


def _get_serpapi():
    global _GoogleSearch
    if _GoogleSearch is None:
        try:
            from serpapi import GoogleSearch
            _GoogleSearch = GoogleSearch
        except ImportError:
            try:
                from serpapi.google_search import GoogleSearch
                _GoogleSearch = GoogleSearch
            except ImportError:
                raise ImportError("Run: pip install google-search-results")
    return _GoogleSearch


GOOGLE_SCHOLAR_MODE = os.getenv("GOOGLE_SCHOLAR_MODE", "scholarly").lower()
SERPAPI_KEY         = os.getenv("SERPAPI_KEY", "")
SCRAPER_API_KEY     = os.getenv("SCRAPER_API_KEY", "")


class GoogleScholarCollector(BaseCollector):
    """
    Collects papers from Google Scholar.
    Supports two modes: scholarly (free) and SerpAPI (paid, reliable).
    Select via GOOGLE_SCHOLAR_MODE env var: "scholarly" or "serpapi".
    """

    source_name = "google_scholar"

    # Google Scholar needs much longer gaps than real APIs to avoid blocking
    _DELAY_MIN = 3.0   # seconds
    _DELAY_MAX = 8.0

    def __init__(self):
        super().__init__()
        self.mode = GOOGLE_SCHOLAR_MODE

        if self.mode == "serpapi":
            if not SERPAPI_KEY:
                logger.warning(
                    "[google_scholar] SERPAPI_KEY not set — falling back to scholarly mode."
                )
                self.mode = "scholarly"
            else:
                logger.info("[google_scholar] Mode: SerpAPI (paid, reliable)")

        if self.mode == "scholarly":
            logger.info("[google_scholar] Mode: scholarly (free — may get blocked at ~100 req)")
            self._configure_scholarly()

    def _configure_scholarly(self):
        """
        Sets up scholarly with optional proxy support.

        Without a proxy: works for development, blocks after ~50-100 requests.
        With ScraperAPI: reliable for production (5000 free req/month).
          Set SCRAPER_API_KEY in .env — sign up free at scraperapi.com
        """
        try:
            s = _get_scholarly()
            if SCRAPER_API_KEY:
                proxy_url = f"http://scraperapi:{SCRAPER_API_KEY}@proxy-server.scraperapi.com:8001"
                pg = s.ProxyGenerator()
                pg.SingleProxy(http=proxy_url, https=proxy_url)
                s.scholarly.use_proxy(pg)
                logger.info("[google_scholar] ScraperAPI proxy configured")
            else:
                logger.warning(
                    "[google_scholar] No proxy set. Add SCRAPER_API_KEY to .env "
                    "for reliable collection (free at scraperapi.com)."
                )
        except Exception as e:
            logger.error(f"[google_scholar] scholarly setup error: {e}")

    # ─── Interface ─────────────────────────────────────────────────────────────

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Google Scholar query plan.

        We use multiple sub-queries because a single broad query misses papers
        that use specific terminology. Each sub-query targets a different angle
        of human microbiome research. The orchestrator deduplicates the results.

        GOOGLE SCHOLAR OPERATORS USED:
          as_ylo / as_yhi  →  year range filter (most reliable)
          No site: or filetype: needed — Scholar handles those internally
        """
        return {
            "sub_queries": [
                "human gut microbiome composition sequencing",
                "human microbiota 16S rRNA amplicon",
                "human metagenomics shotgun sequencing",
                "intestinal microbiome dysbiosis disease",
                "oral skin lung microbiome human",
                "microbiome host interaction immunity human",
            ],
            "year_from": int(date_from[:4]),
            "year_to":   int(date_to[:4]),
        }

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """Routes to the correct backend based on mode."""
        if self.mode == "serpapi":
            return self._fetch_serpapi(query_params, page, page_size)
        return self._fetch_scholarly(query_params, page, page_size)

    # ─── scholarly backend ────────────────────────────────────────────────────

    def _fetch_scholarly(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        Uses scholarly to search Google Scholar.

        HOW IT WORKS:
          scholarly.search_pubs() returns a Python generator. Each call to
          next() on the generator fetches ONE result, making an HTTP request
          to scholar.google.com in the background. We iterate up to page_size
          times with random delays in between.

          Each "page" in our abstraction targets a different sub-query.
          This spreads our requests across different search terms, which
          looks more natural to Google's bot detection.
        """
        s = _get_scholarly()
        sub_queries = query_params["sub_queries"]
        year_from   = query_params["year_from"]
        year_to     = query_params["year_to"]

        # Cycle through sub-queries by page number
        query = sub_queries[page % len(sub_queries)]
        logger.info(f"[google_scholar] scholarly: '{query}' | {year_from}–{year_to}")

        results = []
        try:
            gen = s.scholarly.search_pubs(
                query,
                year_low=year_from,
                year_high=year_to,
                patents=False,
                citations=False,
            )

            for i in range(page_size):
                # Random human-like delay between fetching each result
                if i > 0:
                    delay = random.uniform(self._DELAY_MIN, self._DELAY_MAX)
                    logger.debug(f"[google_scholar] Delay {delay:.1f}s")
                    time.sleep(delay)

                try:
                    pub = next(gen)
                    results.append({"_raw_scholarly": pub, "_query": query})
                except StopIteration:
                    logger.info(f"[google_scholar] End of results for '{query}'")
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if any(w in msg for w in ["captcha", "blocked", "429", "robot"]):
                        logger.error(
                            "[google_scholar] BLOCKED by Google. Options:\n"
                            "  1. Add SCRAPER_API_KEY=... to .env (free at scraperapi.com)\n"
                            "  2. Switch GOOGLE_SCHOLAR_MODE=serpapi (paid, reliable)\n"
                            "  3. Wait a few hours before retrying"
                        )
                        break
                    logger.warning(f"[google_scholar] Result {i} error: {e}")

        except Exception as e:
            logger.error(f"[google_scholar] scholarly search failed: {e}")

        self._save_raw(f"scholarly_p{page}", {"query": query, "n": len(results)})
        return {"records": results}

    # ─── SerpAPI backend ──────────────────────────────────────────────────────

    def _fetch_serpapi(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        Uses SerpAPI to fetch Google Scholar results.

        SERPAPI ADVANTAGE:
          SerpAPI executes searches using real browser automation on their
          infrastructure. You get clean JSON with no HTML parsing, no
          CAPTCHA issues, and no IP blocking. You only pay per search.

        SERPAPI GOOGLE SCHOLAR JSON:
          {
            "organic_results": [
              {
                "title": "...",
                "link": "https://doi.org/10.1038/...",
                "snippet": "150-char abstract snippet...",
                "publication_info": {
                  "summary": "J Smith, K Jones - Nature, 2024",
                  "authors": [{"name": "John Smith", "link": "..."}]
                },
                "inline_links": {
                  "cited_by": {"total": 42},
                  "related_pages_link": "..."
                },
                "resources": [
                  {"title": "PDF", "file_format": "PDF", "link": "https://..."}
                ]
              }
            ]
          }
        """
        GoogleSearch = _get_serpapi()
        sub_queries  = query_params["sub_queries"]
        year_from    = query_params["year_from"]
        year_to      = query_params["year_to"]

        # Which sub-query and which page within it
        q_idx  = page % len(sub_queries)
        p_idx  = page // len(sub_queries)
        query  = sub_queries[q_idx]
        start  = p_idx * page_size

        logger.info(f"[google_scholar] SerpAPI: '{query}' | {year_from}–{year_to} | start={start}")

        try:
            search = GoogleSearch({
                "engine":   "google_scholar",
                "q":        query,
                "as_ylo":   year_from,
                "as_yhi":   year_to,
                "num":      min(page_size, 20),   # Scholar max is 20 per page
                "start":    start,
                "api_key":  SERPAPI_KEY,
                "hl":       "en",
            })
            data    = search.get_dict()
            organic = data.get("organic_results", [])
            tagged  = [{"_raw_serpapi": r, "_query": query} for r in organic]

            self._save_raw(f"serpapi_p{page}", {"query": query, "n": len(tagged)})
            logger.info(f"[google_scholar] SerpAPI: {len(tagged)} results")
            return {"records": tagged}

        except Exception as e:
            logger.error(f"[google_scholar] SerpAPI failed: {e}")
            return {"records": []}

    def _extract_items(self, raw_page: dict) -> list:
        return raw_page.get("records", [])

    # ─── Parsing ──────────────────────────────────────────────────────────────

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """Dispatch to the right parser based on which backend fetched this record."""
        if "_raw_serpapi" in raw:
            return self._parse_serpapi(raw["_raw_serpapi"])
        if "_raw_scholarly" in raw:
            return self._parse_scholarly(raw["_raw_scholarly"])
        return None

    def _parse_scholarly(self, pub: dict) -> Optional[PaperRecord]:
        """
        Parses a scholarly result.

        SCHOLARLY RESULT STRUCTURE:
          {
            "bib": {
              "title": "...",
              "author": ["Smith J", "Jones K"],   ← list of name strings
              "pub_year": "2024",
              "venue": "Nature",
              "abstract": "..."
            },
            "num_citations": 42,
            "pub_url": "https://...",
            "eprint_url": "https://..."   ← PDF link if available
          }

        NOTE: scholarly does NOT give DOIs by default. Getting a DOI requires
        an extra fetch per paper (slow + more requests). We skip it here and
        let the orchestrator's cross-referencing with PubMed fill in DOIs.
        """
        try:
            bib = pub.get("bib", {})
            title = (bib.get("title") or "").strip()
            if not title:
                return None

            authors = bib.get("author", [])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(" and ")]

            year_raw = bib.get("pub_year")
            pub_year = int(year_raw) if year_raw and str(year_raw).isdigit() else None
            eprint   = pub.get("eprint_url")
            pub_url  = pub.get("pub_url", "")

            # Try to extract DOI from the pub_url
            doi = self._doi_from_url(pub_url)

            return PaperRecord(
                doi=doi,
                title=title,
                abstract=bib.get("abstract"),
                authors=authors,
                journal=bib.get("venue"),
                publication_year=pub_year,
                publication_date=f"{pub_year}-01-01" if pub_year else None,
                citation_count=pub.get("num_citations"),
                full_text_url=pub_url or None,
                pdf_url=eprint,
                is_open_access=bool(eprint),
                article_types=["Journal Article"],
            )
        except Exception as e:
            logger.warning(f"[google_scholar] scholarly parse error: {e}")
            return None

    def _parse_serpapi(self, item: dict) -> Optional[PaperRecord]:
        """
        Parses a SerpAPI Google Scholar result.

        SerpAPI's publication_info.summary string format:
          "J Smith, K Jones - Nature, 2024"
          "AB Johnson … - Cell Host & Microbe, 2025 - cell.com"

        We parse this to extract authors, journal, and year.
        """
        try:
            title = (item.get("title") or "").strip()
            if not title:
                return None

            # Parse the summary string
            summary = item.get("publication_info", {}).get("summary", "")
            authors, journal, pub_year = self._parse_summary(summary)

            # Fall back to structured author list if summary parsing failed
            if not authors:
                authors = [
                    a["name"]
                    for a in item.get("publication_info", {}).get("authors", [])
                    if a.get("name")
                ]

            # Citation count
            cited_by       = item.get("inline_links", {}).get("cited_by", {})
            citation_count = cited_by.get("total")

            # PDF link from resources array
            resources = item.get("resources", [])
            pdf_url   = next(
                (r["link"] for r in resources if r.get("file_format") == "PDF"),
                None
            )

            link = item.get("link", "")
            doi  = self._doi_from_url(link)

            return PaperRecord(
                doi=doi,
                title=title,
                abstract=item.get("snippet"),
                authors=authors,
                journal=journal,
                publication_year=pub_year,
                publication_date=f"{pub_year}-01-01" if pub_year else None,
                citation_count=citation_count,
                full_text_url=link or None,
                pdf_url=pdf_url,
                is_open_access=bool(pdf_url),
                article_types=["Journal Article"],
            )
        except Exception as e:
            logger.warning(f"[google_scholar] SerpAPI parse error: {e}")
            return None

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _parse_summary(self, summary: str):
        """
        Parses SerpAPI's publication summary string into components.
        Format: "Author1, Author2 - Journal, Year" or "Author1 ... - Journal, Year - source.com"
        Returns: (authors: list, journal: str|None, year: int|None)
        """
        if not summary:
            return [], None, None

        authors, journal, year = [], None, None

        if " - " in summary:
            parts = summary.split(" - ")
            author_part = parts[0]
            venue_part  = parts[1] if len(parts) > 1 else ""

            # Authors: comma-separated, may have "…" if truncated
            authors = [
                a.strip() for a in author_part.replace("…", "").split(",")
                if a.strip() and len(a.strip()) > 1
            ]

            # Venue: "Nature, 2024" or "Cell Host & Microbe, 2025"
            venue_parts = venue_part.rsplit(",", 1)
            if len(venue_parts) == 2 and venue_parts[1].strip().isdigit():
                journal = venue_parts[0].strip()
                year    = int(venue_parts[1].strip())
            else:
                journal = venue_part.strip()

        return authors, journal, year

    def _doi_from_url(self, url: str) -> Optional[str]:
        """Extracts a DOI from a URL if present."""
        if not url:
            return None
        m = re.search(r"(10\.\d{4,}/[^\s&?#\"']+)", url)
        return m.group(1) if m else None

    # ─── Cap max results ──────────────────────────────────────────────────────

    def collect(self, query, date_from, date_to, max_results=200, page_size=20):
        """
        Caps Google Scholar at 200 results per run.

        WHY THE CAP:
          Unlike real APIs that handle thousands of requests per hour,
          Google Scholar blocks you after ~100-200 requests in a session
          (even with a proxy). Pushing for 500 results gets you blocked
          AND wastes your proxy credits. 200 is the practical sweet spot:
          enough to get unique papers that other sources miss, not so many
          that you trigger blocking.
        """
        capped = min(max_results, 200)
        if max_results > 200:
            logger.warning(
                f"[google_scholar] Capping at 200 results "
                f"(requested {max_results}) to avoid rate-limiting."
            )
        return super().collect(
            query=query, date_from=date_from, date_to=date_to,
            max_results=capped, page_size=page_size,
        )

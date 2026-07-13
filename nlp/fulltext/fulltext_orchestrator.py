"""
nlp/fulltext/fulltext_orchestrator.py
---------------------------------------
Orchestrates full-text fetching with smart routing, persistent caching,
and no-retry tracking. Optimised for 500K+ papers without sacrificing quality.

DESIGN PRINCIPLES:
  1. Try the right strategy for each paper — not all 6 blindly.
     A paper with PMCID goes to EuropePMC/NCBI-PMC first (best quality).
     A paper with only DOI goes to Unpaywall first.
     A paper with only PMID goes to NCBI abstract.
     Papers with pdf_url or full_text_url always get those tried.

  2. Persistent fetch cache (data/fulltext/fetch_cache.json):
     If a paper was successfully fetched in a previous run, return the
     stored result immediately — no API call. If a paper was tried and
     ALL strategies failed, mark it as "exhausted" and skip in future runs.
     This makes re-runs of Layer 2 instant for already-processed papers.

     Exhausted entries carry a TTL (FULLTEXT_EXHAUSTED_TTL_DAYS, default 90
     days). OA embargoes lapse and Unpaywall's index grows over time, so a
     paper that had no available full text 3 months ago might have one now.
     Once an exhausted entry is older than the TTL, the next fetch() call
     retries all strategies exactly once. If it still fails, the entry is
     re-stamped with a fresh timestamp and skipped again until the next TTL
     window — this is never more than one extra attempt per TTL period, not
     unbounded retrying.

  3. No silent shortcuts: if full text exists somewhere, we find it.
     Only fall back to abstract when ALL applicable strategies fail.
     Abstract-only is logged explicitly so you can see the coverage rate.

  4. Strategy quality tiers:
     Tier 1 (Full structured text): EuropePMC XML, NCBI PMC
     Tier 2 (Full unstructured text): PDF, HTML scrape, Unpaywall, OpenAIRE
     Tier 3 (Structured abstract): NCBI PubMed abstract
     Tier 4 (Fallback): whatever abstract was in the PaperRecord already

  5. DOI → PMCID resolution (PMCIDResolver):
     Collectors like Crossref, OpenAlex, and CORE never populate `pmcid`
     even when the paper IS in PubMed Central. Before falling through to
     Tier 2/3 strategies, papers with a DOI but no PMCID get a free lookup
     via NCBI's ID Converter API. If a PMCID is found, the paper is
     promoted to Tier 1 (EuropePMC/NCBI PMC XML) instead of settling for
     PDF parsing or web scraping. Negative results are cached too, so a
     DOI with no PMCID is never looked up twice.

  6. OpenAIRE as an additional Tier 2 source (OpenAIREFetcher):
     Unpaywall's OA index doesn't cover everything — OpenAIRE aggregates
     hundreds of European institutional repositories, DOAJ, and BASE, with
     overlapping but non-identical coverage. Tried after Unpaywall fails,
     using the same DOI, before giving up on Tier 2 entirely.

FETCH CACHE SCHEMA (data/fulltext/fetch_cache.json):
  {
    "{content_hash}": {
      "status": "success" | "exhausted",
      "fetch_source": "ncbi_pmc" | "europepmc" | ... | "abstract_only",
      "fetch_tier": 1 | 2 | 3 | 4,
      "tried": ["europepmc", "ncbi_pmc", ...]   ← strategies that failed
      "exhausted_at": "2026-04-01T12:00:00"     ← only on exhausted entries
    }
  }
  The actual text is stored by FullTextStore (data/fulltext/{hash}.txt).
  The cache only stores status + metadata — not the text itself.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from loguru import logger

from config import FULLTEXT_EXHAUSTED_TTL_DAYS

from nlp.fulltext.europepmc_fulltext import EuropePMCFullText
from nlp.fulltext.ncbi_pmc_fetcher import NCBIPMCFetcher
from nlp.fulltext.pdf_parser import PDFParser
from nlp.fulltext.web_scraper import WebScraper
from nlp.fulltext.unpaywall_fetcher import UnpaywallFetcher
from nlp.fulltext.ncbi_abstract_fetcher import NCBIAbstractFetcher
from nlp.fulltext.pmcid_resolver import PMCIDResolver
from nlp.fulltext.openaire_fetcher import OpenAIREFetcher

_CACHE_PATH = (
    Path(__file__).parent.parent.parent / "data" / "fulltext" / "fetch_cache.json"
)
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        _CACHE_PATH.write_text(
            json.dumps(cache, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"[fulltext_cache] Save failed: {e}")


class FullTextOrchestrator:
    """
    Smart full-text fetcher with persistent caching and strategy routing.
    """

    def __init__(self):
        self.europepmc = EuropePMCFullText()
        self.ncbi_pmc  = NCBIPMCFetcher()
        self.pdf       = PDFParser()
        self.web       = WebScraper()
        self.unpaywall = UnpaywallFetcher()
        self.ncbi_abs  = NCBIAbstractFetcher()
        self.pmcid_resolver = PMCIDResolver()
        self.openaire  = OpenAIREFetcher()

        # Thread-safety lock — required when shared across Phase 1's 64 threads
        import threading
        self._lock = threading.Lock()

        # Load persistent cache — survives process restarts
        self._cache = _load_cache()
        self._cache_dirty = False

        logger.debug(
            f"[fulltext] Cache loaded: {len(self._cache)} entries "
            f"({sum(1 for v in self._cache.values() if v.get('status')=='success')} success, "
            f"{sum(1 for v in self._cache.values() if v.get('status')=='exhausted')} exhausted)"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch(self, paper) -> Optional[dict]:
        """
        Fetches the best available full text for a paper.

        Strategy routing:
          No PMCID but has DOI → try resolving PMCID via NCBI ID Converter
                                   (upgrades Crossref/OpenAlex/CORE papers)
          Has PMCID  → Tier 1: EuropePMC XML → NCBI PMC (highest quality)
          Has pdf_url → Tier 2a: PDF parser
          Has full_text_url → Tier 2b: HTML scraper
          Has DOI    → Tier 2c: Unpaywall → Tier 2d: OpenAIRE (legal OA PDF/HTML)
          Has PMID   → Tier 3: NCBI abstract (complete + structured)
          Nothing    → returns None (use existing abstract from collector)

        Every result is cached by content_hash so the same paper is
        never fetched twice. Exhausted papers (all strategies failed)
        are also cached to skip on future runs.

        Returns:
            dict with fetch_source, fetch_status, and text section keys,
            or None if no text could be retrieved beyond what the paper
            already has from Layer 1.
        """
        cache_key = getattr(paper, "content_hash", None)

        # ── Cache hit ─────────────────────────────────────────────────────────
        if cache_key:
            with self._lock:
                entry = self._cache.get(cache_key)
            if entry:
                if entry.get("status") == "success":
                    return {
                        "fetch_source":  entry.get("fetch_source", "cache"),
                        "fetch_status":  "cached",
                        "fetch_tier":    entry.get("fetch_tier", 0),
                        "_from_cache":   True,
                    }
                elif entry.get("status") == "exhausted":
                    if not self._exhausted_ttl_expired(entry):
                        logger.debug(
                            f"[fulltext] Skipping exhausted paper (TTL not expired): "
                            f"{getattr(paper, 'title', '')[:60]}"
                        )
                        return {
                            "fetch_source": "abstract_only",
                            "fetch_status": "exhausted",
                            "_from_cache":  True,
                        }
                    logger.debug(
                        f"[fulltext] Exhausted TTL expired ({FULLTEXT_EXHAUSTED_TTL_DAYS}d) — "
                        f"retrying: {getattr(paper, 'title', '')[:60]}"
                    )

        tried     = []
        pmcid     = getattr(paper, "pmcid",         None)
        pdf_url   = getattr(paper, "pdf_url",        None)
        ft_url    = getattr(paper, "full_text_url",  None)
        doi       = getattr(paper, "doi",            None)
        pmid      = getattr(paper, "pmid",           None)

        # ── DOI → PMCID resolution ──────────────────────────────────────────────
        # Crossref, OpenAlex, and CORE never populate `pmcid` even when the
        # paper is actually in PubMed Central. A free NCBI lookup can upgrade
        # these papers from Tier 2/3 strategies to Tier 1 structured XML.
        if not pmcid and doi:
            try:
                resolved = self.pmcid_resolver.resolve(doi)
            except Exception as e:
                logger.debug(f"[fulltext] pmcid_resolver failed for DOI {doi}: {e}")
                resolved = None
            if resolved:
                logger.debug(
                    f"[fulltext] Resolved PMCID {resolved} from DOI {doi} "
                    f"(collector didn't provide it) — upgrading to Tier 1"
                )
                pmcid = resolved

        # ── Tier 1: Full structured text from XML (best quality) ──────────────
        if pmcid:
            result = self._try("europepmc", lambda: self.europepmc.fetch(pmcid), tried)
            if result:
                result["fetch_tier"] = 1
                self._cache_success(cache_key, result)
                return result

            result = self._try("ncbi_pmc", lambda: self.ncbi_pmc.fetch(pmcid), tried)
            if result:
                result["fetch_tier"] = 1
                self._cache_success(cache_key, result)
                return result

        # ── Tier 2: Full unstructured text ────────────────────────────────────
        if pdf_url:
            result = self._try("pdf", lambda: self.pdf.fetch(pdf_url), tried)
            if result:
                result["fetch_tier"] = 2
                self._cache_success(cache_key, result)
                return result

        if ft_url:
            result = self._try("web", lambda: self.web.fetch(ft_url), tried)
            if result:
                result["fetch_tier"] = 2
                self._cache_success(cache_key, result)
                return result

        if doi:
            result = self._try("unpaywall", lambda: self.unpaywall.fetch(doi), tried)
            if result:
                result["fetch_tier"] = 2
                self._cache_success(cache_key, result)
                return result

            result = self._try("openaire", lambda: self.openaire.fetch(doi), tried)
            if result:
                result["fetch_tier"] = 2
                self._cache_success(cache_key, result)
                return result

        # ── Tier 3: Complete structured abstract from NCBI ────────────────────
        if pmid:
            result = self._try("ncbi_abstract", lambda: self.ncbi_abs.fetch(pmid), tried)
            if result:
                result["fetch_tier"] = 3
                self._cache_success(cache_key, result)
                return result

        # ── All strategies exhausted ──────────────────────────────────────────
        # Mark as exhausted so future runs skip this paper immediately.
        # The pipeline will use whatever abstract came from the collector.
        if cache_key:
            self._cache_exhausted(cache_key, tried)
            logger.debug(
                f"[fulltext] All strategies exhausted for "
                f"'{getattr(paper, 'title', '')[:60]}' "
                f"(tried: {tried}) — using collector abstract"
            )

        return None

    def flush_cache(self):
        """Explicitly flush the cache to disk."""
        if self._cache_dirty:
            _save_cache(self._cache)
            self._cache_dirty = False
        self.pmcid_resolver.flush_cache()

    def cache_stats(self) -> dict:
        """Returns a summary of cache contents."""
        success   = sum(1 for v in self._cache.values() if v.get("status") == "success")
        exhausted = sum(1 for v in self._cache.values() if v.get("status") == "exhausted")
        by_source = {}
        for v in self._cache.values():
            src = v.get("fetch_source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1
        return {
            "total":     len(self._cache),
            "success":   success,
            "exhausted": exhausted,
            "by_source": by_source,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _try(self, name: str, fn, tried: list) -> Optional[dict]:
        """
        Calls a fetcher strategy, catches all exceptions.
        Appends the strategy name to tried[] regardless of outcome.
        """
        tried.append(name)
        try:
            result = fn()
            if result and self._has_content(result):
                logger.debug(f"[fulltext] Strategy '{name}' succeeded")
                return result
        except Exception as e:
            logger.debug(f"[fulltext] Strategy '{name}' failed: {e}")
        return None

    def _has_content(self, result: dict) -> bool:
        """Returns True if the result contains meaningful text content."""
        text_keys = ("full_text", "abstract", "methods", "results", "discussion")
        return any(
            bool((result.get(k) or "").strip())
            for k in text_keys
        )

    def _cache_success(self, cache_key: Optional[str], result: dict):
        """Records a successful fetch in the cache."""
        if not cache_key:
            return
        with self._lock:
            self._cache[cache_key] = {
                "status":       "success",
                "fetch_source": result.get("fetch_source", "unknown"),
                "fetch_tier":   result.get("fetch_tier", 0),
            }
            self._cache_dirty = True
            # Save every 100 new entries to avoid data loss on crash
            if len(self._cache) % 100 == 0:
                _save_cache(self._cache)
                self._cache_dirty = False

    def _cache_exhausted(self, cache_key: Optional[str], tried: list):
        """
        Records that all strategies failed for this paper.
        Stamps `exhausted_at` so a future run can retry after the TTL
        (FULLTEXT_EXHAUSTED_TTL_DAYS) expires — OA embargoes lapse and
        Unpaywall's index grows, so "no full text today" isn't permanent.
        """
        if not cache_key:
            return
        with self._lock:
            self._cache[cache_key] = {
                "status":       "exhausted",
                "fetch_source": "abstract_only",
                "tried":        tried,
                "exhausted_at": datetime.utcnow().isoformat(),
            }
            self._cache_dirty = True
            if len(self._cache) % 100 == 0:
                _save_cache(self._cache)
                self._cache_dirty = False

    def _exhausted_ttl_expired(self, entry: dict) -> bool:
        """
        Returns True if an exhausted entry's TTL has expired and it's
        eligible for a retry. Entries with no `exhausted_at` timestamp
        (written before this feature existed) are treated as expired
        immediately, so old data gets one retry rather than being stuck
        forever without a timestamp.
        """
        exhausted_at_raw = entry.get("exhausted_at")
        if not exhausted_at_raw:
            return True
        try:
            exhausted_at = datetime.fromisoformat(exhausted_at_raw)
        except (ValueError, TypeError):
            return True
        return datetime.utcnow() - exhausted_at > timedelta(days=FULLTEXT_EXHAUSTED_TTL_DAYS)

    def __del__(self):
        """Flush cache to disk on garbage collection."""
        try:
            if self._cache_dirty:
                _save_cache(self._cache)
        except Exception:
            pass

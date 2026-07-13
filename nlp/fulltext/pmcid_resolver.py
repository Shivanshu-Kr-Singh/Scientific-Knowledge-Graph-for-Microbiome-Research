"""
nlp/fulltext/pmcid_resolver.py
--------------------------------
Resolves DOIs to PMCIDs using two independent, free, no-key APIs.

WHY THIS EXISTS:
  Collectors like Crossref, OpenAlex, and CORE never populate the `pmcid`
  field on a PaperRecord — they simply don't expose it in their APIs.
  But a meaningful fraction of those papers ARE deposited in PubMed
  Central under a PMCID that nobody looked up.

  Without this resolver, those papers can only reach Tier 2/3 full-text
  strategies (PDF parsing, web scraping, Unpaywall, abstract-only).
  With it, papers that turn out to have a PMCID get promoted to Tier 1
  (EuropePMC / NCBI PMC structured XML) — the highest quality full text
  available.

TWO INDEPENDENT SOURCES:
  1. NCBI ID Converter (primary) — tried first for every DOI.
  2. Europe PMC search-by-DOI (fallback) — tried only for DOIs the NCBI
     converter couldn't resolve. This is a genuinely separate index
     (EMBL-EBI, not NCBI) with its own ingestion pipeline, so it
     occasionally has a paper the NCBI converter hasn't caught up on yet
     (or vice versa — hence trying NCBI first since it's the canonical
     source). Confirmed against the live API to return `pmcid` directly
     in DOI search results, and to support batching via OR queries.
  Both are free, require no API key/registration, and are already used
  elsewhere in this codebase for full-text fetching (europepmc_fulltext.py,
  ncbi_pmc_fetcher.py) — this resolver just also uses them for ID lookup.

WHY BATCHING MATTERS:
  A single-DOI-per-request approach works fine at trial scale (tens of
  papers) but becomes the bottleneck at real scale.
    - NCBI ID Converter accepts up to 200 IDs per request (confirmed live).
    - Europe PMC accepts multi-DOI OR queries; chunked at 50 DOIs per
      request here to keep URLs short (confirmed live with 52 DOIs in
      one request, well under any practical URL length limit).
  At 10,000 papers, one-at-a-time resolution takes ~1 hour serialized
  before any full-text fetching even starts; batched, the same 10,000
  papers resolve in well under a minute across both sources combined.
  This is the same coverage, same free APIs — just not wasting round trips.

API DOCS:
  NCBI ID Converter:
    https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/
    Endpoint: https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/
    No API key. Max 200 IDs per request, all same type.

  Europe PMC Search:
    https://europepmc.org/RestfulWebService
    Endpoint: https://www.ebi.ac.uk/europepmc/webservices/rest/search
    No API key. Query by "DOI:{doi}", multiple DOIs joinable with " OR ".

CACHING:
  Results (including negative "not found" results, meaning neither source
  found a PMCID) are cached persistently in data/fulltext/pmcid_cache.json
  so the same DOI is never looked up twice against either source, whether
  resolved individually or as part of a batch.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

from config import DATA_DIR

# NCBI moved this endpoint (the old /pmc/utils/idconv/v1.0/ URL now 301s here).
# Calling the destination directly avoids an extra round trip per request.
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

_EMAIL = os.getenv("NCBI_EMAIL", "research@example.com")
_TOOL = "microbiome_miner"
_DELAY = 0.35              # seconds between NCBI requests — polite, no API key needed
_BATCH_SIZE = 200           # NCBI's documented max IDs per request
_EUROPEPMC_BATCH_SIZE = 50  # keeps OR-query URLs comfortably short; verified working at 52

_CACHE_PATH = DATA_DIR / "fulltext" / "pmcid_cache.json"
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
        _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[pmcid_resolver] Cache save failed: {e}")


class PMCIDResolver:
    """
    Resolves DOI → PMCID via NCBI's ID Converter (primary) and Europe PMC's
    search API (fallback), with a persistent cache so repeat lookups
    (including "not found" results) are free.
    """

    def __init__(self):
        self._cache = _load_cache()
        self._cache_dirty = False

    def resolve(self, doi: str) -> Optional[str]:
        """
        Looks up the PMCID for a given DOI, trying NCBI first and falling
        back to Europe PMC's search API if NCBI has no record of it.

        Args:
            doi: DOI string, e.g. "10.1038/s41586-024-07999-z"

        Returns:
            PMCID string (e.g. "PMC9876543") if found, else None.
        """
        if not doi or not doi.strip():
            return None

        doi_clean = doi.strip()

        # ── Cache hit (including cached negative results) ─────────────────────
        if doi_clean in self._cache:
            return self._cache[doi_clean] or None

        pmcid = self._query(doi_clean)
        if not pmcid:
            pmcid = self._query_europepmc(doi_clean)

        # Cache the result even if None — avoids re-querying DOIs with no PMCID
        self._cache[doi_clean] = pmcid
        self._cache_dirty = True
        if len(self._cache) % 50 == 0:
            _save_cache(self._cache)
            self._cache_dirty = False

        return pmcid

    def _query(self, doi: str) -> Optional[str]:
        try:
            resp = requests.get(
                IDCONV_URL,
                params={
                    "ids": doi,
                    "format": "json",
                    "email": _EMAIL,
                    "tool": _TOOL,
                },
                timeout=10,
            )
            time.sleep(_DELAY)

            if resp.status_code != 200:
                return None

            data = resp.json()
            records = data.get("records") or []
            if not records:
                return None

            record = records[0]

            # NCBI reports errors on the record itself (e.g. "status": "error")
            if record.get("status") == "error":
                return None

            pmcid = record.get("pmcid")
            if pmcid:
                logger.debug(f"[pmcid_resolver] Resolved DOI {doi} → {pmcid}")
            return pmcid or None

        except Exception as e:
            logger.debug(f"[pmcid_resolver] Lookup failed for DOI {doi}: {e}")
            return None

    def _query_europepmc(self, doi: str) -> Optional[str]:
        """
        Fallback lookup via Europe PMC's search API — a separate index
        from NCBI's ID Converter, occasionally covering a paper NCBI
        hasn't caught up on yet. Only called when NCBI returns nothing.
        """
        try:
            resp = requests.get(
                EUROPEPMC_SEARCH_URL,
                params={"query": f"DOI:{doi}", "format": "json", "pageSize": 1},
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            results = (data.get("resultList") or {}).get("result") or []
            if not results:
                return None

            pmcid = results[0].get("pmcid")
            if pmcid:
                logger.debug(
                    f"[pmcid_resolver] Resolved DOI {doi} → {pmcid} (via Europe PMC fallback)"
                )
            return pmcid or None

        except Exception as e:
            logger.debug(f"[pmcid_resolver] Europe PMC lookup failed for DOI {doi}: {e}")
            return None

    # ── Batch resolution ──────────────────────────────────────────────────────

    def resolve_batch(self, dois: list[str]) -> dict[str, Optional[str]]:
        """
        Resolves many DOIs to PMCIDs using as few HTTP requests as possible.

        This is the throughput fix for large runs: instead of one request
        per DOI (rate-limited to ~3/sec, so 10,000 DOIs = ~1 hour serialized),
        this chunks the list into groups of up to 200 and fires one request
        per chunk — 10,000 DOIs becomes ~50 requests, done in well under a
        minute. Coverage/accuracy is identical to calling resolve() in a loop;
        this only changes how many round trips it takes.

        Cache-aware: DOIs already resolved (positively or negatively) are
        skipped entirely and answered from cache without any network call.
        Only genuinely new DOIs are sent to NCBI, in batches.

        Args:
            dois: list of DOI strings. Duplicates and empty/None entries
                  are handled gracefully.

        Returns:
            dict mapping each input DOI (stripped) to its PMCID string,
            or None if no PMCID was found. DOIs that were empty/None are
            omitted from the result.
        """
        # Normalize, dedupe, drop empties — preserves nothing about order
        # since callers look results up by key, not position.
        unique_dois = {d.strip() for d in dois if d and d.strip()}
        if not unique_dois:
            return {}

        result: dict[str, Optional[str]] = {}
        to_fetch: list[str] = []

        # ── Serve everything already cached (including negative results) ──────
        for doi in unique_dois:
            if doi in self._cache:
                result[doi] = self._cache[doi] or None
            else:
                to_fetch.append(doi)

        if not to_fetch:
            logger.debug(
                f"[pmcid_resolver] Batch of {len(unique_dois)} DOIs — "
                f"all served from cache, 0 requests"
            )
            return result

        logger.info(
            f"[pmcid_resolver] Batch resolving {len(to_fetch)} new DOIs "
            f"({len(unique_dois) - len(to_fetch)} already cached) — "
            f"{-(-len(to_fetch) // _BATCH_SIZE)} request(s) of up to {_BATCH_SIZE} IDs each"
        )

        for i in range(0, len(to_fetch), _BATCH_SIZE):
            chunk = to_fetch[i : i + _BATCH_SIZE]
            chunk_results = self._query_batch(chunk)
            for doi in chunk:
                result[doi] = chunk_results.get(doi)
            time.sleep(_DELAY)

        # ── Europe PMC fallback pass — only for DOIs NCBI didn't resolve ───────
        # A separate index from NCBI's, so worth one more free, batched attempt
        # before giving up on a DOI entirely.
        still_missing = [doi for doi in to_fetch if not result.get(doi)]
        if still_missing:
            logger.info(
                f"[pmcid_resolver] {len(still_missing)} DOIs unresolved by NCBI — "
                f"trying Europe PMC fallback"
            )
            for i in range(0, len(still_missing), _EUROPEPMC_BATCH_SIZE):
                chunk = still_missing[i : i + _EUROPEPMC_BATCH_SIZE]
                chunk_results = self._query_europepmc_batch(chunk)
                for doi in chunk:
                    pmcid = chunk_results.get(doi)
                    if pmcid:
                        result[doi] = pmcid

        for doi in to_fetch:
            self._cache[doi] = result.get(doi)
        self._cache_dirty = True
        _save_cache(self._cache)
        self._cache_dirty = False

        resolved_count = sum(1 for v in result.values() if v)
        logger.info(
            f"[pmcid_resolver] Batch complete: {resolved_count}/{len(unique_dois)} "
            f"DOIs resolved to a PMCID"
        )
        return result

    def _query_batch(self, dois: list[str]) -> dict[str, Optional[str]]:
        """
        Fires one HTTP request for up to 200 DOIs and returns a dict
        mapping each requested DOI to its PMCID (or None if not found).
        """
        out: dict[str, Optional[str]] = {doi: None for doi in dois}
        try:
            resp = requests.get(
                IDCONV_URL,
                params={
                    "ids": ",".join(dois),
                    "format": "json",
                    "email": _EMAIL,
                    "tool": _TOOL,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(
                    f"[pmcid_resolver] Batch request failed with status "
                    f"{resp.status_code} for {len(dois)} DOIs"
                )
                return out

            data = resp.json()
            records = data.get("records") or []

            for record in records:
                requested = record.get("requested-id") or record.get("doi")
                if not requested:
                    continue
                if record.get("status") == "error":
                    out[requested] = None
                else:
                    out[requested] = record.get("pmcid") or None

            return out

        except Exception as e:
            logger.warning(f"[pmcid_resolver] Batch lookup failed: {e}")
            return out

    def _query_europepmc_batch(self, dois: list[str]) -> dict[str, Optional[str]]:
        """
        Fires one Europe PMC search request for up to _EUROPEPMC_BATCH_SIZE
        DOIs, joined via OR, and returns a dict mapping each requested DOI
        to its PMCID (or None if not found). Only called as a fallback for
        DOIs the NCBI ID Converter couldn't resolve.
        """
        out: dict[str, Optional[str]] = {doi: None for doi in dois}
        query = " OR ".join(f"DOI:{doi}" for doi in dois)
        try:
            resp = requests.get(
                EUROPEPMC_SEARCH_URL,
                params={"query": query, "format": "json", "pageSize": len(dois)},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(
                    f"[pmcid_resolver] Europe PMC batch request failed with "
                    f"status {resp.status_code} for {len(dois)} DOIs"
                )
                return out

            data = resp.json()
            results = (data.get("resultList") or {}).get("result") or []

            # Match results back to requested DOIs case-insensitively — Europe
            # PMC echoes the DOI as submitted by the publisher, which can
            # differ in case from how it was requested.
            by_doi_lower = {r.get("doi", "").lower(): r for r in results if r.get("doi")}
            for doi in dois:
                match = by_doi_lower.get(doi.lower())
                if match:
                    out[doi] = match.get("pmcid") or None

            return out

        except Exception as e:
            logger.warning(f"[pmcid_resolver] Europe PMC batch lookup failed: {e}")
            return out

    def flush_cache(self):
        """Explicitly flush the cache to disk."""
        if self._cache_dirty:
            _save_cache(self._cache)
            self._cache_dirty = False

    def __del__(self):
        try:
            if self._cache_dirty:
                _save_cache(self._cache)
        except Exception:
            pass

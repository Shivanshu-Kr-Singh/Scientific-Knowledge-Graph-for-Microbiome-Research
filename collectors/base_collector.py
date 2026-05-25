"""
Abstract base class for all data source collectors.
WHY INHERITANCE HERE?
Every collector (PubMed, Europe PMC, etc.) needs the same core behaviours:
- Rate limiting (respect API limits)
- Retry on failure (network glitches happen)
- Content hash checking (don't reprocess unchanged papers)
- Logging
- Saving raw responses to disk

Instead of copy-pasting this into every collector, we write it ONCE here.
Each specific collector then only needs to implement:
- build_query()→ how to form the search query for that source
- fetch_batch()→ how to fetch a page of results
- parse_record()→ how to convert raw JSON → PaperRecord
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import MAX_RETRIES, RETRY_BACKOFF_BASE, RATE_LIMITS, RAW_DIR
from models import PaperRecord

class BaseCollector(ABC):
    """All source collectors inherit from this. Subclasses must implement: source_name, build_query, fetch_page, parse_record."""
    # Each subclass sets this — used for rate limiting and file naming
    source_name: str = "base"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MicrobiomeMiner/1.0 (Academic research; contact@example.com)"
            # Always identify yourself to academic APIs. It's required by NCBI
            # and good practice everywhere. Anonymized scrapers get blocked faster.
        })
        self._last_request_time = 0.0
        self._rate_limit_seconds = RATE_LIMITS.get(self.source_name, 1.0)

        # Directory for caching raw API responses
        self._raw_dir = RAW_DIR / self.source_name
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{self.source_name}] Collector initialized")


    # ─── Rate Limiting ─────────────────────────────────────────────────────────
    def _wait_for_rate_limit(self):
        """
        Enforces a minimum gap between requests so we don't get rate-limited.
        HOW IT WORKS:
          We record when the last request was made. Before each new request,
          we check how much time has passed. If it's less than the minimum
          gap, we sleep for the remainder. This is 'token bucket' style
          rate limiting — simple but effective for single-threaded code.
        """
        elapsed = time.time() - self._last_request_time
        wait = self._rate_limit_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()


    # ─── HTTP with Retry ───────────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_BACKOFF_BASE, min=2, max=30),
        reraise=True
    )
    def _get(self, url: str, params: dict = None, **kwargs) -> requests.Response:
        """
        Makes an HTTP GET request with automatic rate limiting and retry.
        The @retry decorator from tenacity handles:
          - Retrying up to MAX_RETRIES times
          - Waiting 2s → 4s → 8s between retries (exponential backoff)
          - Raising the last exception if all retries are exhausted

        WHY EXPONENTIAL BACKOFF?
          If an API is temporarily overloaded, hammering it every second makes
          things worse. Waiting longer each time gives it time to recover.
        """
        self._wait_for_rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=30, **kwargs)

            # 429 = Too Many Requests. Respect the Retry-After header if present.
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"[{self.source_name}] Rate limited. Waiting {retry_after}s")
                time.sleep(retry_after)
                raise requests.exceptions.HTTPError("429 Rate Limited")

            response.raise_for_status()
            return response

        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{self.source_name}] Connection error: {e}")
            raise

    # ─── Content Hashing ──────────────────────────────────────────────────────

    def _compute_hash(self, paper: PaperRecord) -> str:
        """
        Creates an MD5 fingerprint of a paper's core content.

        WHY WE HASH:
          When the scheduler re-fetches papers it already has, this hash tells
          us whether anything changed. If the hash is the same as last time,
          we skip re-processing. If it changed (e.g. author corrected the
          abstract after publication), we re-run the NLP pipeline on it.

          We hash title + abstract + author list because those are the parts
          most likely to be corrected post-publication.
        """
        content = f"{paper.title}|{paper.abstract}|{','.join(paper.authors)}"
        return hashlib.md5(content.encode()).hexdigest()

    # ─── Raw Response Caching ─────────────────────────────────────────────────

    def _save_raw(self, identifier: str, data: dict):
        """
        Saves the raw API JSON response to disk before any processing.

        WHY CACHE RAW RESPONSES?
          1. If the NLP pipeline fails or changes, you can re-process without
             re-fetching from the API (faster, and avoids rate limits).
          2. You have an audit trail of exactly what each API returned.
          3. During development, you can test NLP changes on local data.
        """
        safe_id = identifier.replace("/", "_").replace(":", "_")
        path = self._raw_dir / f"{safe_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_raw(self, identifier: str) -> Optional[dict]:
        """Loads a previously cached raw response, or None if not cached."""
        safe_id = identifier.replace("/", "_").replace(":", "_")
        path = self._raw_dir / f"{safe_id}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # ─── Abstract Methods (subclasses must implement) ─────────────────────────

    @abstractmethod
    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        Returns the query parameters dict for this source's search API.
        Each source has a different query syntax, so each subclass handles its own.
        """
        ...

    @abstractmethod
    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """
        Fetches one page of search results from the API.
        Returns the raw JSON response as a dict.
        """
        ...

    @abstractmethod
    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Converts one raw API result dict into a unified PaperRecord.
        Returns None if the record should be skipped (e.g. wrong language, no title).
        """
        ...


    # ─── Main Collection Loop (same for all sources) ──────────────────────────
    def collect(
        self,
        query: str,
        date_from: str,
        date_to: str,
        max_results: int = 500,
        page_size: int = 50,
    ) -> List[PaperRecord]:
        """
        Runs the full collection loop for this source.

        WHAT HAPPENS HERE:
          1. Build the query for this source
          2. Fetch pages of results until we have max_results or run out
          3. Parse each raw result into a PaperRecord
          4. Compute and attach a content hash to each record
          5. Return the full list

        The caller (Orchestrator) handles deduplication across sources.
        """
        logger.info(f"[{self.source_name}] Starting collection | query='{query}' | {date_from} → {date_to}")

        query_params = self.build_query(query, date_from, date_to)
        papers: List[PaperRecord] = []
        page = 0
        total_fetched = 0

        while total_fetched < max_results:
            batch_size = min(page_size, max_results - total_fetched)

            try:
                raw_page = self.fetch_page(query_params, page=page, page_size=batch_size)
            except Exception as e:
                logger.error(f"[{self.source_name}] Failed to fetch page {page}: {e}")
                break   # Stop this source but let others continue

            # Parse each result in the page
            records_on_page = 0
            for raw_item in self._extract_items(raw_page):
                paper = self.parse_record(raw_item)
                if paper is None:
                    continue   # Filtered out (no title, wrong language, etc.)

                # Attach ingestion metadata
                paper.source = self.source_name
                paper.content_hash = self._compute_hash(paper)
                paper.fetched_at = datetime.utcnow().isoformat()

                papers.append(paper)
                records_on_page += 1

            total_fetched += records_on_page
            logger.info(f"[{self.source_name}] Page {page}: {records_on_page} records | Total: {total_fetched}")

            # Stop if this page had fewer results than requested — we've hit the end
            if records_on_page < batch_size:
                logger.info(f"[{self.source_name}] Reached end of results at page {page}")
                break

            page += 1

        logger.success(f"[{self.source_name}] Collection complete: {len(papers)} papers")
        return papers

    def _extract_items(self, raw_page: dict) -> list:
        """
        Extracts the list of individual paper records from a raw page response.
        Default implementation tries common patterns; subclasses can override.
        """
        # Try common key patterns used by different APIs
        for key in ["records", "resultList", "results", "hits", "items", "data"]:
            if key in raw_page:
                val = raw_page[key]
                # Some APIs nest further: {"resultList": {"result": [...]}}
                if isinstance(val, dict):
                    for subkey in ["result", "items", "records"]:
                        if subkey in val:
                            return val[subkey]
                if isinstance(val, list):
                    return val
        return []

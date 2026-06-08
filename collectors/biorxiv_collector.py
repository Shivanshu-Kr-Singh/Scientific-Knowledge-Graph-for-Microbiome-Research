"""
Fetches preprints from bioRxiv and medRxiv.
WHY PREPRINTS?
In fast-moving fields like microbiome research, important work appears
on bioRxiv months or years before journal publication. Tracking preprints lets you:-
- Catch cutting-edge work early
- Track which preprints eventually get published (and link them)
- See how conclusions change between preprint and final version

Returns JSON.
"""

from typing import List, Optional
from loguru import logger
from models import PaperRecord
from collectors.base_collector import BaseCollector

BIORXIV_BASE = "https://api.biorxiv.org"

# bioRxiv API returns at most 100 records per call
_PAGE_SIZE = 100


class BioRxivCollector(BaseCollector):
    """Fetches preprints from bioRxiv and medRxiv."""

    source_name = "biorxiv"

    def collect(
        self,
        query: str,
        date_from: str,
        date_to: str,
        max_results: int = 500,
        page_size: int = 50,
<<<<<<< HEAD
    ) -> List[PaperRecord]:
        """
        Override to emit a bioRxiv-appropriate start log.
        The bioRxiv API has no keyword search — it filters by date range only,
        so logging the query string would be misleading.
        All actual pagination logic lives in fetch_page().
        """
        from datetime import datetime
        logger.info(
            f"[{self.source_name}] Starting collection | "
            f"{date_from} → {date_to} | "
            f"keyword search: not available in API — client-side filtering applied"
        )

        query_params = self.build_query(query, date_from, date_to)
        papers: List[PaperRecord] = []
        page = 0
        total_fetched = 0

        while total_fetched < max_results:
            batch_size = min(page_size, max_results - total_fetched)
=======
        start_offset: int = 0,
    ) -> List[PaperRecord]:
        """
        Override collect() for bioRxiv.

        fetch_page() now returns already-parsed, already-filtered PaperRecord
        objects (via inline RelevanceFilter). This loop just accumulates them —
        no re-parsing or re-stamping needed.
        """
        logger.info(
            f"[{self.source_name}] Starting collection | "
            f"{date_from} → {date_to} | "
            f"keyword search: not available in API — per-batch relevance filtering applied"
        )
        if start_offset > 0:
            logger.info(f"[{self.source_name}] Resuming from offset {start_offset}")

        query_params        = self.build_query(query, date_from, date_to)
        papers: List[PaperRecord] = []
        effective_page_size = max_results
        page                = start_offset // max(effective_page_size, 1)
        total_fetched       = 0

        while total_fetched < max_results:
            batch_size = min(effective_page_size, max_results - total_fetched)
>>>>>>> 74ebc70b97b04bd4abe564892ac2a6c5b4ce7932

            try:
                raw_page = self.fetch_page(query_params, page=page, page_size=batch_size)
            except Exception as e:
                logger.error(f"[{self.source_name}] Failed to fetch page {page}: {e}")
                break

<<<<<<< HEAD
            records_on_page = 0
            for raw_item in self._extract_items(raw_page):
                paper = self.parse_record(raw_item)
                if paper is None:
                    continue

                paper.source = self.source_name
                paper.content_hash = self._compute_hash(paper)
                paper.fetched_at = datetime.utcnow().isoformat()

                papers.append(paper)
                records_on_page += 1

            total_fetched += records_on_page
            logger.info(
                f"[{self.source_name}] Page {page}: {records_on_page} records | Total: {total_fetched}"
            )

            if records_on_page < batch_size:
                logger.info(f"[{self.source_name}] Reached end of results at page {page}")
=======
            # fetch_page returns pre-parsed, pre-filtered PaperRecord objects
            records_on_page = 0
            for item in self._extract_items(raw_page):
                if isinstance(item, PaperRecord):
                    # Already parsed and filtered — add directly
                    papers.append(item)
                    records_on_page += 1
                # (non-PaperRecord items are ignored — shouldn't happen)

            total_fetched += records_on_page
            logger.info(
                f"[{self.source_name}] Page {page}: {records_on_page} relevant records | "
                f"Total: {total_fetched}"
            )

            if records_on_page < batch_size:
                logger.info(
                    f"[{self.source_name}] Reached end of results at page {page} "
                    f"(fewer relevant papers than target)"
                )
>>>>>>> 74ebc70b97b04bd4abe564892ac2a6c5b4ce7932
                break

            page += 1

        logger.success(f"[{self.source_name}] Collection complete: {len(papers)} papers")
        return papers
<<<<<<< HEAD
=======
        return papers
>>>>>>> 74ebc70b97b04bd4abe564892ac2a6c5b4ce7932

    def build_query(self, query: str, date_from: str, date_to: str) -> dict:
        """
        bioRxiv API filters by date range only — no keyword search.
        We do keyword filtering client-side after fetching.

        Date format for bioRxiv API: YYYY-MM-DD
        """
        date_from_clean = date_from.replace("/", "-")[:10]
        date_to_clean   = date_to.replace("/", "-")[:10]

        return {
            "date_from": date_from_clean,
            "date_to":   date_to_clean,
            "servers":   ["biorxiv", "medrxiv"],
        }

    def fetch_page(self, query_params: dict, page: int, page_size: int) -> dict:
        """
<<<<<<< HEAD
        Fetches ALL papers in the date range from both bioRxiv and medRxiv,
        then applies keyword filtering to keep only microbiome-relevant ones.

        WHY EXHAUST ALL PAGES:
          The bioRxiv API has no keyword search — it returns everything
          published in the date window regardless of topic. We must scan
          all available pages before filtering, otherwise we'd miss relevant
          papers that happen to fall on later pages.

        bioRxiv API URL format:
          /details/{server}/{date_from}/{date_to}/{offset}/json

          offset — 0-based cursor, increments by 100 each call.
          Each call returns up to 100 records.
          An empty "collection" array means we've reached the end.

        page_size is the desired number of *filtered* results to collect
        per server. We stop paginating once we've found enough.
        """
        all_results = []

        for server in query_params["servers"]:
            offset = 0
            server_kept = 0
            total_available = None  # populated from first API response
=======
        Fetches batches of 30 from bioRxiv/medRxiv, running RelevanceFilter
        after EACH batch, accumulating only relevant papers until we have
        page_size relevant papers per server (or exhaust all available papers).

        WHY PER-BATCH FILTERING:
          The bioRxiv API returns papers by date range only — no topic filter.
          Any given batch of 30 can contain 0 relevant papers. We must keep
          fetching and filtering until we accumulate enough relevant ones.

        CURSOR BEHAVIOUR:
          The offset advances by the actual API batch size (30) regardless of
          how many papers passed the filter. This ensures we correctly skip
          already-scanned papers on the next run even if they all filtered out.

        bioRxiv API URL format:
          /details/{server}/{date_from}/{date_to}/{offset}/json
          Returns up to 30 results per call (API-fixed, not configurable).
        """
        from collectors.relevance_filter import RelevanceFilter
        from datetime import datetime as _dt

        all_relevant = []

        for server in query_params["servers"]:
            initial_offset  = page * _PAGE_SIZE
            offset          = initial_offset
            server_relevant = 0
            total_available = None

            # One RelevanceFilter instance per server — reuse across batches
            rel_filter = RelevanceFilter()
>>>>>>> 74ebc70b97b04bd4abe564892ac2a6c5b4ce7932

            while True:
                url = (
                    f"{BIORXIV_BASE}/details/{server}/"
                    f"{query_params['date_from']}/"
                    f"{query_params['date_to']}/"
                    f"{offset}/json"
                )

                try:
                    response   = self._get(url)
                    data       = response.json()
                    collection = data.get("collection", [])
                except Exception as e:
                    logger.error(f"[biorxiv] Failed {server} offset {offset}: {e}")
                    break
<<<<<<< HEAD

                # Log total available once, on the first page — mirrors PubMed/EuropePMC
                if offset == 0:
                    raw_total = data.get("messages", [{}])[0].get("total", None)
                    total_available = int(raw_total) if raw_total is not None else None
                    if total_available is not None:
                        logger.info(
                            f"[biorxiv] Total results available on {server}: "
                            f"{total_available} | collecting up to {page_size}"
                        )

                if not collection:
                    # Empty page → exhausted all results for this server
                    logger.info(f"[biorxiv] {server}: no more results at offset {offset}")
                    break

                for item in collection:
                    item["_server"] = server

                all_results.extend(collection)
                server_kept += len(collection)

                logger.info(
                    f"[biorxiv] {server} offset {offset}: "
                    f"{len(collection)} papers fetched | total so far: {server_kept}"
                )

                # Stop early if we already have enough for this server
                if server_kept >= page_size:
                    logger.info(
                        f"[biorxiv] {server}: reached target of {page_size} "
                        f"papers — stopping early "
                        f"({total_available - server_kept if total_available else '?'} remaining uncollected)"
                    )
                    break

                # Fewer results than a full page → this was the last page
                if len(collection) < _PAGE_SIZE:
                    break

                offset += _PAGE_SIZE
=======
>>>>>>> 74ebc70b97b04bd4abe564892ac2a6c5b4ce7932

                # Log total available on the first call for this server
                if offset == initial_offset:
                    raw_total = data.get("messages", [{}])[0].get("total", None)
                    total_available = int(raw_total) if raw_total is not None else None
                    if total_available is not None:
                        logger.info(
                            f"[biorxiv] Total results available on {server}: "
                            f"{total_available} | target: {page_size} relevant papers"
                        )

<<<<<<< HEAD
=======
                if not collection:
                    logger.info(f"[biorxiv] {server}: no more results at offset {offset}")
                    break

                # Tag each item with its server
                for item in collection:
                    item["_server"] = server

                # ── Parse raw dicts → PaperRecord for filtering ───────────────
                batch_records = []
                for raw in collection:
                    paper = self.parse_record(raw)
                    if paper:
                        paper.source       = self.source_name
                        paper.content_hash = self._compute_hash(paper)
                        paper.fetched_at   = _dt.utcnow().isoformat()
                        batch_records.append(paper)

                # ── Run RelevanceFilter on this batch of 30 ───────────────────
                kept, _, _ = rel_filter.filter(batch_records)

                logger.info(
                    f"[biorxiv] {server} offset {offset}: "
                    f"{len(collection)} raw → {len(kept)} relevant "
                    f"(accumulated: {server_relevant + len(kept)}/{page_size})"
                )

                all_relevant.extend(kept)
                server_relevant += len(kept)

                # Cursor always advances by the actual API batch size
                # regardless of how many passed the filter
                offset += len(collection)

                # Stop once we have enough relevant papers for this server
                if server_relevant >= page_size:
                    remaining = (total_available - offset) if total_available else "?"
                    logger.info(
                        f"[biorxiv] {server}: reached target of {page_size} "
                        f"relevant papers — stopping "
                        f"({remaining} raw papers remaining uncollected)"
                    )
                    break

                # Fewer than expected results → last available page
                if len(collection) < _PAGE_SIZE:
                    logger.info(
                        f"[biorxiv] {server}: exhausted all available papers "
                        f"at offset {offset} — found {server_relevant} relevant"
                    )
                    break

        self._save_raw(f"page_{page}", {"relevant_count": len(all_relevant)})

        # Return already-parsed PaperRecord objects — _extract_items passes them through
        return {"records": all_relevant, "_pre_parsed": True}

>>>>>>> 74ebc70b97b04bd4abe564892ac2a6c5b4ce7932
    def _extract_items(self, raw_page: dict) -> list:
        return raw_page.get("records", [])

    def parse_record(self, raw: dict) -> Optional[PaperRecord]:
        """
        Parses one bioRxiv/medRxiv JSON record.

        BIORXIV JSON STRUCTURE:
          {
            "doi": "10.1101/2024.01.15.575821",
            "title": "...",
            "authors": "Smith J; Jones K; Lee M",  ← semicolon-separated string!
            "author_corresponding": "Smith J",
            "author_corresponding_institution": "MIT",
            "date": "2024-01-15",
            "version": "1",                          ← preprint version number
            "type": "new results",
            "category": "microbiology",
            "jatsxml": "https://...",               ← full XML if available
            "abstract": "...",
            "published": "10.1038/...",              ← DOI of final published version, if known
            "server": "biorxiv"
          }
        """
        try:
            title = (raw.get("title") or "").strip()
            if not title:
                return None

            doi = raw.get("doi")

            # bioRxiv authors come as a semicolon-separated string, not a list
            authors_raw = raw.get("authors", "")
            if isinstance(authors_raw, str):
                authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
            else:
                authors = authors_raw

            # If there's a published DOI, the preprint became a paper
            published_doi = raw.get("published")
            is_published  = bool(published_doi and published_doi != "NA")

            return PaperRecord(
                doi=doi,
                title=title,
                abstract=raw.get("abstract"),
                authors=authors,
                publication_date=raw.get("date"),
                publication_year=int(raw.get("date", "2024")[:4]) if raw.get("date") else None,
                journal=f"{raw.get('_server', 'biorxiv').capitalize()} (preprint)",
                article_types=[raw.get("type", "preprint"), raw.get("category", "")],
                is_open_access=True,   # All bioRxiv preprints are open access
                is_preprint=not is_published,
            )

        except Exception as e:
            logger.warning(f"[biorxiv] Failed to parse record: {e}")
            return None

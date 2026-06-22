"""
Runs all collectors in sequence, merges results, and deduplicates.
WHY AN ORCHESTRATOR?
Without this, you'd have to manually run each collector and figure out
how to merge them. The orchestrator is the single entry point for Layer 1:
call collect_all() and get back one clean, deduplicated list of PaperRecord
objects — ready to feed into the NLP pipeline.

        DEDUPLICATION STRATEGY:
  The same paper often appears in multiple sources:
    - A paper published in Nature → in PubMed AND Europe PMC AND Semantic Scholar

  We deduplicate by DOI first (most reliable), then by PMID, then by
  normalized title (fuzzy fallback). When a paper appears in multiple sources,
  we MERGE the records: take the best available value for each field
  (e.g. citation count from Semantic Scholar, MeSH terms from PubMed).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from tqdm import tqdm

from config import (
    SEARCH_QUERY, DATE_FROM, DATE_TO,
    MAX_RESULTS_PER_SOURCE, PROC_DIR
)
from models import PaperRecord
from collectors.pubmed_collector import PubMedCollector
from collectors.relevance_filter import RelevanceFilter
from collectors.europepmc_collector import EuropePMCCollector
from collectors.semantic_scholar_collector import SemanticScholarCollector
from collectors.openalex_collector import OpenAlexCollector
from collectors.crossref_collector import CrossrefCollector
from collectors.core_collector import CoreCollector
from collectors.pmc_enricher import PMCEnricher

# Path to the file that persists per-source fetch cursors across runs
CURSOR_FILE = PROC_DIR / "collector_cursors.json"


class CollectionOrchestrator:
    """
    Coordinates all collectors and produces a single deduplicated dataset.
    """

    def __init__(self):
        self.collectors = [
            PubMedCollector(),
            EuropePMCCollector(),
            SemanticScholarCollector(),
            OpenAlexCollector(),
            CrossrefCollector(),
            CoreCollector(),
        ]
        logger.info(f"Orchestrator ready with {len(self.collectors)} collectors")

    def collect_all(
        self,
        query:      str = SEARCH_QUERY,
        date_from:  str = DATE_FROM,
        date_to:    str = DATE_TO,
        max_per_source: int = MAX_RESULTS_PER_SOURCE,
    ) -> List[PaperRecord]:
        """
        Runs all collectors, merges results, deduplicates, and returns final list.

        STEPS:
          1. Run each collector → get raw lists of PaperRecords
          2. Pool all records from all sources
          3. Deduplicate: group records with the same DOI/PMID
          4. For each duplicate group, merge into one best record
          5. Save the final merged list to disk
          6. Return the list for Layer 2 to process
        """
        logger.info("=" * 60)
        logger.info(f"Starting full collection run")
        logger.info(f"Query: '{query}' | {date_from} → {date_to}")
        logger.info("=" * 60)

        all_records: List[PaperRecord] = []

        # ── Load cursors (resume from where last run left off) ─────────────────
        cursors = self._load_cursors()
        updated_cursors = dict(cursors)  # will be updated after each collector

        # ── Step 1: Run each collector ─────────────────────────────────────────
        for collector in self.collectors:
            source = collector.source_name
            start_offset = cursors.get(source, 0)

            # Semantic Scholar uses token-based pagination — inject the saved token
            if source == "semantic_scholar":
                saved_token = cursors.get("semantic_scholar_token")
                collector._resume_token = saved_token
                if saved_token:
                    logger.info(f"[semantic_scholar] Resuming from saved continuation token")
            # OpenAlex uses opaque cursor strings — inject the saved cursor
            elif source == "openalex":
                saved_cursor = cursors.get("openalex_cursor")
                collector._resume_cursor = saved_cursor
                if saved_cursor and start_offset > 0:
                    logger.info(f"[openalex] Resuming from saved cursor string")
            elif start_offset > 0:
                logger.info(
                    f"[{source}] Resuming from offset {start_offset} "
                    f"(fetched {start_offset} papers in previous runs)"
                )

            try:
                records = collector.collect(
                    query=query,
                    date_from=date_from,
                    date_to=date_to,
                    max_results=max_per_source,
                    start_offset=start_offset,
                )

                # Advance cursor by ACTUAL records collected, not by max_per_source.
                # This ensures if collection stops early (network drop, API limit,
                # source exhausted), the cursor reflects reality.
                # Next run with MAX_PER_SOURCE=5000 will collect 5000 MORE from
                # where this run actually stopped — not restart from 0.
                actual_collected = len(records)

                # For PubMed: use the last retstart the collector reached,
                # since it tracks its own offset internally via WebHistory.
                # For all others: start_offset + actual_collected is correct.
                # For PubMed: use the last retstart the collector reached,
                # since it tracks its own offset internally via WebHistory.
                # For OpenAlex: save the opaque cursor string for cross-run resume.
                # For all others: start_offset + actual_collected is correct.
                if source == "pubmed":
                    last_offset = getattr(collector, "_last_retstart", None)
                    if last_offset is not None:
                        updated_cursors[source] = last_offset
                    else:
                        updated_cursors[source] = start_offset + actual_collected

                elif source == "openalex":
                    # Save numeric offset for resume detection
                    updated_cursors[source] = start_offset + actual_collected
                    # Save opaque cursor string for actual pagination resume
                    last_cursor = getattr(collector, "_last_cursor", None)
                    if last_cursor:
                        updated_cursors["openalex_cursor"] = last_cursor
                        # Inject into next collector instance at run start
                    else:
                        # Cursor exhausted — reset both
                        updated_cursors.pop("openalex_cursor", None)
                        logger.info("[openalex] All results consumed — cursor reset")
                else:
                    updated_cursors[source] = start_offset + actual_collected

                # For S2: save the continuation token for next run
                if source == "semantic_scholar":
                    last_token = getattr(collector, "_last_token", None)
                    if last_token:
                        updated_cursors["semantic_scholar_token"] = last_token
                    else:
                        # Token exhausted — S2 results fully consumed, reset
                        updated_cursors.pop("semantic_scholar_token", None)
                        updated_cursors[source] = 0
                        logger.info("[semantic_scholar] All results consumed — cursor reset")

                all_records.extend(records)
                logger.info(
                    f"[{source}] Added {actual_collected} records | "
                    f"cursor → {updated_cursors[source]}"
                )

            except Exception as e:
                logger.error(f"[{source}] COLLECTOR FAILED: {e}")
                logger.exception(e)
                # Don't advance cursor if collector failed — retry same offset next run
                updated_cursors[source] = start_offset

        # ── Save updated cursors ───────────────────────────────────────────────
        self._save_cursors(updated_cursors)

        logger.info(f"Total raw records before dedup: {len(all_records)}")

        # ── Step 2 & 3: Deduplicate ────────────────────────────────────────────
        merged = self._deduplicate_and_merge(all_records)

        logger.success(f"After deduplication: {len(merged)} unique papers")

        # ── Step 3: Post-collection relevance filter ──────────────────────
        # Stage 1: MeSH metadata filter (PubMed papers)
        # Stage 2: Weighted rule scorer (all sources, from stage2_rules.yaml)
        # Stage 3: ML classifier (if trained model exists)
        # + Metagenomics gate (project-specific requirement)
        if merged:
            rel_filter = RelevanceFilter()
            merged, removed, review_queue = rel_filter.filter(merged)
        else:
            removed, review_queue = [], []
        logger.info(
            f"Relevance filter: kept {len(merged)}, "
            f"removed {len(removed)}, "
            f"flagged for review: {len(review_queue)}"
        )

        # ── Step 4: PMC full-text enrichment ──────────────────────────────────
        # For any paper that has a PMCID, fetch its full XML from PMC and
        # attach structured full text (Methods, Results, Discussion, etc.).
        # This upgrades existing papers rather than finding new ones.
        pmc_candidates = sum(1 for p in merged if p.pmcid and not p.full_text)
        if pmc_candidates > 0:
            logger.info(
                f"[pmc_enricher] {pmc_candidates} papers have PMCID — "
                f"fetching full text from PMC"
            )
            enricher = PMCEnricher()
            merged = enricher.enrich(merged, max_enrichments=pmc_candidates)
        else:
            logger.info("[pmc_enricher] No papers with PMCID — skipping enrichment")

        # ── Step 5: Save to disk ───────────────────────────────────────────────
        output_path = self._save_merged(merged)
        logger.success(f"Saved merged dataset → {output_path}")

        # ── Step 6: Print summary ──────────────────────────────────────────────
        self._print_summary(merged)

        return merged

    # ─── Deduplication Logic ──────────────────────────────────────────────────

    def _deduplicate_and_merge(self, records: List[PaperRecord]) -> List[PaperRecord]:
        """
        Groups records by their dedup key and merges each group into one record.

        MERGE PRIORITY:
          Different sources are better at different fields. We use this priority:
            - PubMed:           best for MeSH terms, article types, dates
            - Europe PMC:       best for full text availability, PMC IDs
            - Semantic Scholar: best for citation counts, reference lists
            - OpenAlex:         best for open metadata, funder info
            - Crossref:         best for DOI/publisher metadata
            - CORE:             best for open-access full text

          For each field, we take the FIRST non-null value found in priority order.
          For list fields (authors, mesh_terms), we union and deduplicate.
        """
        # Source priority — lower index = higher priority for metadata
        SOURCE_PRIORITY = ["pubmed", "europepmc", "semantic_scholar", "openalex", "crossref", "core"]

        # Group records by their dedup key
        groups: Dict[str, List[PaperRecord]] = {}

        for record in records:
            key = record.get_dedup_key()
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        merged_records = []
        duplicate_count = 0

        for key, group in tqdm(groups.items(), desc="Merging records"):
            if len(group) == 1:
                merged_records.append(group[0])
                continue

            duplicate_count += len(group) - 1

            # Sort group by source priority
            def sort_key(r: PaperRecord):
                try:
                    return SOURCE_PRIORITY.index(r.source)
                except ValueError:
                    return len(SOURCE_PRIORITY)

            group.sort(key=sort_key)

            # Merge: start with highest-priority record, fill in from others
            merged = group[0].model_copy()

            for other in group[1:]:
                # For scalar fields: take the first non-null value
                for field in ["doi", "pmid", "pmcid", "abstract", "journal",
                              "issn", "publication_date", "publication_year",
                              "volume", "issue", "pages", "citation_count",
                              "reference_count", "full_text_url", "pdf_url"]:
                    if getattr(merged, field) is None and getattr(other, field) is not None:
                        setattr(merged, field, getattr(other, field))

                # For boolean fields: OR them (true if any source says true)
                merged.is_open_access = merged.is_open_access or other.is_open_access

                # For list fields: union (deduplicated)
                merged.authors    = self._merge_lists(merged.authors, other.authors)
                merged.keywords   = self._merge_lists(merged.keywords, other.keywords)
                merged.mesh_terms = self._merge_lists(merged.mesh_terms, other.mesh_terms)
                merged.article_types = self._merge_lists(merged.article_types, other.article_types)

            # Mark the record as having been merged from multiple sources
            merged.source = f"merged:{'+'.join(r.source for r in group)}"

            merged_records.append(merged)

        logger.info(f"Removed {duplicate_count} duplicate records")
        return merged_records

    def _merge_lists(self, list1: list, list2: list) -> list:
        """Returns union of two lists, preserving order, deduplicating by lowercase."""
        seen = set()
        result = []
        for item in (list1 or []) + (list2 or []):
            key = item.lower() if isinstance(item, str) else str(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def _load_cursors(self) -> Dict[str, int]:
        """
        Loads per-source fetch cursors from disk.
        Returns a dict like {"pubmed": 40, "europepmc": 20, ...}.

        AUTO-RESET LOGIC:
          If the cursor file says we've already fetched papers, but there are
          no collected_*.json files in the processed folder, the data was
          deleted while cursors were not. In that case we reset all cursors
          to 0 so the next run re-fetches from the beginning.
        """
        if not CURSOR_FILE.exists():
            return {}

        with open(CURSOR_FILE) as f:
            cursors = json.load(f)

        # Check if any non-zero cursor exists but the collected data is gone
        has_nonzero_cursor = any(v > 0 for v in cursors.values())
        collected_files = list(PROC_DIR.glob("collected_*.json"))

        if has_nonzero_cursor and not collected_files:
            logger.warning(
                "[cursors] Collected data files not found but cursors are non-zero — "
                "data may have been deleted. Resetting all cursors to 0."
            )
            self.reset_cursors()
            return {}

        return cursors

    def _save_cursors(self, cursors: Dict[str, int]):
        """Persists the updated cursors to disk after a successful run."""
        with open(CURSOR_FILE, "w") as f:
            json.dump(cursors, f, indent=2)
        logger.info(f"[cursors] Saved fetch cursors → {CURSOR_FILE}")

    def reset_cursors(self):
        """
        Resets all cursors back to 0 so the next run starts from the beginning.
        Call this when you want to re-collect from scratch (e.g. new date range).
        """
        if CURSOR_FILE.exists():
            CURSOR_FILE.unlink()
        logger.info("[cursors] All fetch cursors reset to 0")
        """Returns union of two lists, preserving order, deduplicating by lowercase."""
        seen = set()
        result = []
        for item in (list1 or []) + (list2 or []):
            key = item.lower() if isinstance(item, str) else str(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _save_merged(self, records: List[PaperRecord]) -> Path:
        """
        Saves the merged record list to a timestamped JSON file.
        This is the handoff point between Layer 1 and Layer 2.
        Layer 2 will load this file and run NLP on each record.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = PROC_DIR / f"collected_{timestamp}.json"

        data = [r.model_dump() for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return path

    def load_latest(self) -> List[PaperRecord]:
        """
        Loads the most recently saved collection from disk.
        Useful for re-running Layer 2 without re-fetching from APIs.
        """
        files = sorted(PROC_DIR.glob("collected_*.json"), reverse=True)
        if not files:
            raise FileNotFoundError("No collected data found. Run collect_all() first.")

        path = files[0]
        logger.info(f"Loading latest collection: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [PaperRecord(**item) for item in data]

    def load_all(self) -> List[PaperRecord]:
        """
        Loads and merges ALL collected_*.json files from disk, deduplicating
        across runs by DOI → PMID → title fallback.

        WHY THIS EXISTS:
          load_latest() only sees the most recent batch. For ML training
          at scale (60k papers across many runs), you need ALL collected
          papers merged into one deduplicated list — otherwise the model
          trains on a tiny subset and misses patterns from earlier runs.

        Returns a single deduplicated list sorted newest-first.
        """
        files = sorted(PROC_DIR.glob("collected_*.json"), reverse=True)
        if not files:
            raise FileNotFoundError("No collected data found. Run collect_all() first.")

        logger.info(f"[load_all] Found {len(files)} collected file(s) — merging...")

        seen_keys: set = set()
        all_papers: List[PaperRecord] = []
        total_raw = 0

        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"[load_all] Skipping {path.name}: {e}")
                continue

            file_added = 0
            for item in data:
                try:
                    paper = PaperRecord(**item)
                    key = paper.get_dedup_key()
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_papers.append(paper)
                        file_added += 1
                except Exception:
                    pass
            total_raw += len(data)
            logger.info(f"[load_all]   {path.name}: {len(data)} records, "
                        f"{file_added} new after dedup")

        logger.info(f"[load_all] Total: {total_raw} raw → {len(all_papers)} unique papers")
        return all_papers

    def load_all_rejected(self) -> List[PaperRecord]:
        """
        Loads and merges ALL rejected_*.json files, deduplicating across runs.

        Used by train_ml_model() to build the negative training set from
        every rejection across all collection runs — not just the latest one.
        """
        files = sorted(PROC_DIR.glob("rejected_*.json"))
        if not files:
            logger.info("[load_all_rejected] No rejection files found.")
            return []

        logger.info(f"[load_all_rejected] Found {len(files)} rejected file(s) — merging...")

        seen_keys: set = set()
        all_rejected: List[PaperRecord] = []
        total_raw = 0

        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"[load_all_rejected] Skipping {path.name}: {e}")
                continue

            file_added = 0
            for item in data:
                try:
                    paper = PaperRecord(**item)
                    key = paper.get_dedup_key()
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_rejected.append(paper)
                        file_added += 1
                except Exception:
                    pass
            total_raw += len(data)
            logger.info(f"[load_all_rejected]   {path.name}: {len(data)} raw, "
                        f"{file_added} new after dedup")

        logger.info(f"[load_all_rejected] Total: {total_raw} raw → "
                    f"{len(all_rejected)} unique rejected papers")
        return all_rejected

    # ─── Reporting ────────────────────────────────────────────────────────────

    def _print_summary(self, records: List[PaperRecord]):
        """Prints a human-readable summary of the collection run."""
        from collections import Counter

        years = Counter(r.publication_year for r in records if r.publication_year)
        oa_count = sum(1 for r in records if r.is_open_access)
        preprint_count = sum(1 for r in records if r.is_preprint)

        logger.info("─" * 40)
        logger.info(f"COLLECTION SUMMARY")
        logger.info(f"  Total unique papers:    {len(records)}")
        logger.info(f"  Open access:            {oa_count} ({100*oa_count//max(len(records),1)}%)")
        logger.info(f"  Preprints:              {preprint_count}")
        logger.info(f"  Papers by year:         {dict(sorted(years.items()))}")
        logger.info("─" * 40)

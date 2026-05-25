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
    - A preprint that got published → in bioRxiv AND PubMed

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
from collectors.biorxiv_collector import BioRxivCollector


class CollectionOrchestrator:
    """
    Coordinates all collectors and produces a single deduplicated dataset.
    """

    def __init__(self):
        self.collectors = [
            PubMedCollector(),
            EuropePMCCollector(),
            SemanticScholarCollector(),
            BioRxivCollector(),
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

        # ── Step 1: Run each collector ─────────────────────────────────────────
        for collector in self.collectors:
            try:
                records = collector.collect(
                    query=query,
                    date_from=date_from,
                    date_to=date_to,
                    max_results=max_per_source,
                )
                all_records.extend(records)
                logger.info(f"[{collector.source_name}] Added {len(records)} records")

            except Exception as e:
                # If one source fails entirely, log and continue with the others.
                # We NEVER let one broken source kill the whole job.
                logger.error(f"[{collector.source_name}] COLLECTOR FAILED: {e}")
                logger.exception(e)

        logger.info(f"Total raw records before dedup: {len(all_records)}")

        # ── Step 2 & 3: Deduplicate ────────────────────────────────────────────
        merged = self._deduplicate_and_merge(all_records)

        logger.success(f"After deduplication: {len(merged)} unique papers")

        # ── Step 3: Post-collection relevance filter (3-stage) ───────────────
        # Stage 1: MeSH metadata filter (PubMed papers)
        # Stage 2: Weighted rule scorer (all sources, from organisms.yaml)
        # Stage 3: ML classifier (if trained model exists)
        # + Metagenomics gate (project-specific requirement)
        rel_filter = RelevanceFilter()
        merged, removed, review_queue = rel_filter.filter(merged)
        logger.info(
            f"Relevance filter: kept {len(merged)}, "
            f"removed {len(removed)}, "
            f"flagged for review: {len(review_queue)}"
        )

        # ── Step 4: Save to disk ───────────────────────────────────────────────
        output_path = self._save_merged(merged)
        logger.success(f"Saved merged dataset → {output_path}")

        # ── Step 5: Print summary ──────────────────────────────────────────────
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
            - bioRxiv:          best for preprint version info

          For each field, we take the FIRST non-null value found in priority order.
          For list fields (authors, mesh_terms), we union and deduplicate.
        """
        # Source priority — lower index = higher priority for metadata
        SOURCE_PRIORITY = ["pubmed", "europepmc", "semantic_scholar", "biorxiv"]

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

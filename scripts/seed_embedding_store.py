"""
scripts/seed_embedding_store.py

One-time backfill script that seeds the Embedding Store from historical audit data.
Reads data/audit/{kept,rejected,llm_verified}.json, encodes papers with non-empty
abstracts, and places them into the appropriate Embedding Store partition.

Idempotent: deduplicates by DOI/PMID before inserting.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from loguru import logger

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from collectors.embedding_model import EmbeddingModel
from collectors.embedding_store import EmbeddingStore, EmbeddingMetadata


AUDIT_DIR = _PROJECT_ROOT / "data" / "audit"


class BackfillSeeder:
    """
    Reads data/audit/{kept,rejected,llm_verified}.json
    Encodes papers with non-empty abstracts using EmbeddingModel.
    Places into positive/negative partitions.
    Idempotent: deduplicates by DOI/PMID before inserting.
    """

    def __init__(self):
        self._model = EmbeddingModel()
        self._store = EmbeddingStore()

    def run(self) -> dict:
        """
        Execute the backfill seeding process.

        Returns:
            dict with keys: positive_added, negative_added, skipped_no_abstract, skipped_duplicate
        """
        stats = {
            "positive_added": 0,
            "negative_added": 0,
            "skipped_no_abstract": 0,
            "skipped_duplicate": 0,
        }

        # Collect records with their target partitions
        records = []
        records.extend(self._load_file(AUDIT_DIR / "kept.json", "positive"))
        records.extend(self._load_file(AUDIT_DIR / "rejected.json", "negative"))
        records.extend(self._load_llm_verified(AUDIT_DIR / "llm_verified.json"))

        logger.info(f"Loaded {len(records)} total audit records")

        # Check if any records have abstracts
        has_abstract = any(
            self._has_abstract(record) for record, _ in records
        )
        if not has_abstract:
            sys.exit(
                "No audit records with abstract fields found. "
                "Please run the enhanced Audit Logger first to populate abstracts."
            )

        for record, partition in records:
            # Skip records without abstract
            if not self._has_abstract(record):
                stats["skipped_no_abstract"] += 1
                continue

            # Deduplicate by DOI/PMID
            doi = record.get("doi")
            pmid = record.get("pmid")
            if self._store.contains(doi=doi, pmid=pmid):
                stats["skipped_duplicate"] += 1
                continue

            # Encode and store
            title = record.get("title", "")
            abstract = record.get("abstract", "")
            vector = self._model.encode_paper(title, abstract)

            metadata = EmbeddingMetadata(
                doi=doi,
                pmid=pmid,
                title=title,
                partition=partition,
                added_at=datetime.now(timezone.utc).isoformat(),
            )

            self._store.append(vector, metadata)

            if partition == "positive":
                stats["positive_added"] += 1
            else:
                stats["negative_added"] += 1

        logger.info(
            f"Backfill complete: "
            f"positive_added={stats['positive_added']}, "
            f"negative_added={stats['negative_added']}, "
            f"skipped_no_abstract={stats['skipped_no_abstract']}, "
            f"skipped_duplicate={stats['skipped_duplicate']}"
        )

        return stats

    def _load_file(self, path: Path, partition: str) -> list[tuple[dict, str]]:
        """Load a JSON audit file and tag each record with the given partition."""
        if not path.exists():
            logger.warning(f"Audit file not found: {path}")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} records from {path.name}")
            return [(record, partition) for record in data]
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read {path}: {e}")
            return []

    def _load_llm_verified(self, path: Path) -> list[tuple[dict, str]]:
        """
        Load llm_verified.json with partition logic:
        - keep=True → positive
        - keep=False → negative
        """
        if not path.exists():
            logger.warning(f"Audit file not found: {path}")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} records from {path.name}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read {path}: {e}")
            return []

        results = []
        for record in data:
            if record.get("keep", False):
                results.append((record, "positive"))
            else:
                results.append((record, "negative"))
        return results

    @staticmethod
    def _has_abstract(record: dict) -> bool:
        """Check if a record has a non-empty abstract field."""
        abstract = record.get("abstract")
        return bool(abstract and abstract.strip())


if __name__ == "__main__":
    BackfillSeeder().run()

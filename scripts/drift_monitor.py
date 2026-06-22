"""
scripts/drift_monitor.py

Monthly drift monitoring script that samples automated (non-LLM-verified) decisions
from the past month for manual review.

Samples 1% of automated decisions with a minimum of 10 papers.
Writes sampled papers to data/audit/drift_review_YYYYMM.json.
Logs sample size and keep/reject distribution.
"""

import json
import random
import sys
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

from loguru import logger

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATA_DIR, DRIFT_SAMPLE_RATE, DRIFT_MIN_SAMPLE


AUDIT_DIR = DATA_DIR / "audit"


class DriftMonitor:
    """
    Samples 1% of automated (non-LLM-verified) decisions from the past month.
    Minimum 10 papers in sample. Writes to data/audit/drift_review_YYYYMM.json.
    """

    MIN_SAMPLE_SIZE = DRIFT_MIN_SAMPLE
    SAMPLE_RATE = DRIFT_SAMPLE_RATE

    def __init__(self):
        self._audit_dir = AUDIT_DIR

    def run(self) -> dict:
        """
        Execute drift monitoring: sample automated decisions from past month,
        write review file, log stats.

        Returns:
            {"sample_size": int, "keep_count": int, "reject_count": int}
        """
        # Load automated decisions from audit data
        automated_decisions = self._load_automated_decisions()
        population_size = len(automated_decisions)

        logger.info(f"Drift monitor: found {population_size} automated decisions from past month")

        if population_size == 0:
            logger.warning("No automated decisions found for drift review")
            return {"sample_size": 0, "keep_count": 0, "reject_count": 0}

        # Compute sample size: max(ceil(population * rate), min_sample)
        # But if population < min_sample, sample all of them
        if population_size < self.MIN_SAMPLE_SIZE:
            sample_size = population_size
        else:
            sample_size = max(ceil(population_size * self.SAMPLE_RATE), self.MIN_SAMPLE_SIZE)

        # Clamp to population (can't sample more than available)
        sample_size = min(sample_size, population_size)

        # Random sample without replacement
        sampled = random.sample(automated_decisions, sample_size)

        # Compute keep/reject distribution
        keep_count = sum(1 for p in sampled if p.get("decision") == "keep")
        reject_count = sample_size - keep_count

        # Write to drift review file
        now = datetime.now(timezone.utc)
        review_filename = f"drift_review_{now.strftime('%Y%m')}.json"
        review_path = self._audit_dir / review_filename

        self._audit_dir.mkdir(parents=True, exist_ok=True)

        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(sampled, f, indent=2)

        logger.info(
            f"Drift review written to {review_path.name}: "
            f"sample_size={sample_size}, keep={keep_count}, reject={reject_count}"
        )

        return {
            "sample_size": sample_size,
            "keep_count": keep_count,
            "reject_count": reject_count,
        }

    def _load_automated_decisions(self) -> list[dict]:
        """
        Load audit records that were resolved automatically (not by LLM/Stage 4).

        Automated decisions are those resolved at stages 1, 2, 3, 3.5, or gate —
        i.e., the `stage` field does NOT contain 'stage4' or 'llm'.
        """
        automated = []

        for filepath in [self._audit_dir / "kept.json", self._audit_dir / "rejected.json"]:
            records = self._load_json_file(filepath)
            for record in records:
                if self._is_automated(record):
                    automated.append(record)

        return automated

    def _load_json_file(self, path: Path) -> list[dict]:
        """Load a JSON audit file, returning empty list on failure."""
        if not path.exists():
            logger.debug(f"Audit file not found: {path}")
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning(f"Unexpected format in {path.name}, expected list")
                return []
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read {path}: {e}")
            return []

    @staticmethod
    def _is_automated(record: dict) -> bool:
        """
        Check if a record was an automated decision (not LLM-verified).

        Automated means resolved at stages 1, 2, 3, 3.5, or gate —
        NOT at stage4/LLM.
        """
        stage = str(record.get("stage", "")).lower()
        # If the stage contains 'stage4' or 'llm', it's LLM-verified
        if "stage4" in stage or "llm" in stage:
            return False
        return True


if __name__ == "__main__":
    result = DriftMonitor().run()
    print(f"Drift monitor result: {result}")

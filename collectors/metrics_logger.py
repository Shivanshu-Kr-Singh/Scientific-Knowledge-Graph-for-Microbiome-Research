"""
collectors/metrics_logger.py

Pipeline metrics logging — appends one JSON record per pipeline run
to a JSONL file for observability and stage resolution trend tracking.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
"""

import json
import dataclasses
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config import DATA_DIR

METRICS_PATH = DATA_DIR / "metrics" / "pipeline_runs.jsonl"


@dataclass
class PipelineMetrics:
    """One pipeline run's metrics — appended as JSONL."""

    timestamp: str
    total_papers: int
    stage1_resolved: int
    stage2_resolved: int
    gate_resolved: int
    stage3_resolved: int
    stage3_5_resolved: int
    stage4_resolved: int
    llm_calls: int
    semantic_cache_hits: int
    batch_count: int
    batch_retries: int
    embedding_store_positive: int
    embedding_store_negative: int
    avg_embedding_latency_ms: float
    p95_embedding_latency_ms: float


class MetricsLogger:
    """Appends pipeline run metrics to a JSONL file."""

    def __init__(self, path: Path = METRICS_PATH) -> None:
        self._path = path

    def record(self, metrics: PipelineMetrics) -> None:
        """Append one JSON record to the JSONL file.

        Creates parent directories if they don't exist.
        Handles write errors gracefully — logs and continues without
        crashing the pipeline.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            record = dataclasses.asdict(metrics)
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug("Pipeline metrics recorded to {}", self._path)
        except (OSError, IOError) as exc:
            logger.error(
                "Failed to write pipeline metrics to {}: {}",
                self._path,
                exc,
            )

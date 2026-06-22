"""
Property 19: Pipeline Metrics Record Completeness

**Validates: Requirements 15.1, 15.2, 15.3, 15.4**

For any pipeline run processing N papers, the appended JSONL record SHALL
contain: timestamp, total_papers equal to N, per-stage resolution counts that
sum to N, llm_calls count, semantic_cache_hits count, batch stats, and
embedding store sizes.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from collectors.metrics_logger import MetricsLogger, PipelineMetrics


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Total papers in a pipeline run (1–10000)
_total_papers_st = st.integers(min_value=1, max_value=10000)


@st.composite
def _stage_partition_st(draw: st.DrawFn, total: int) -> tuple[int, int, int, int, int, int]:
    """Partition `total` into 6 non-negative integers that sum to total.

    Represents: stage1, stage2, gate, stage3, stage3_5, stage4 resolutions.
    """
    # Draw 5 split points in [0, total], sort them, then compute differences
    cuts = sorted(draw(st.lists(
        st.integers(min_value=0, max_value=total),
        min_size=5,
        max_size=5,
    )))
    parts = (
        cuts[0],
        cuts[1] - cuts[0],
        cuts[2] - cuts[1],
        cuts[3] - cuts[2],
        cuts[4] - cuts[3],
        total - cuts[4],
    )
    return parts


# Non-negative integers for counter fields
_counter_st = st.integers(min_value=0, max_value=100000)

# Non-negative floats for latency fields
_latency_st = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 19: Pipeline Metrics Record Completeness
# **Validates: Requirements 15.1, 15.2, 15.3, 15.4**
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    total_papers=_total_papers_st,
    data=st.data(),
    llm_calls=_counter_st,
    semantic_cache_hits=_counter_st,
    batch_count=_counter_st,
    batch_retries=_counter_st,
    embedding_store_positive=_counter_st,
    embedding_store_negative=_counter_st,
    avg_latency=_latency_st,
    p95_latency=_latency_st,
)
def test_property_pipeline_metrics_record_completeness(
    total_papers: int,
    data: st.DataObject,
    llm_calls: int,
    semantic_cache_hits: int,
    batch_count: int,
    batch_retries: int,
    embedding_store_positive: int,
    embedding_store_negative: int,
    avg_latency: float,
    p95_latency: float,
) -> None:
    """
    **Property 19: Pipeline Metrics Record Completeness**

    **Validates: Requirements 15.1, 15.2, 15.3, 15.4**

    For any pipeline run processing N papers, the appended JSONL record SHALL
    contain: timestamp, total_papers equal to N, per-stage resolution counts
    that sum to N, llm_calls count, semantic_cache_hits count, batch stats,
    and embedding store sizes.
    """
    # Generate per-stage partition that sums to total_papers
    stage_counts = data.draw(_stage_partition_st(total_papers))
    (
        stage1_resolved,
        stage2_resolved,
        gate_resolved,
        stage3_resolved,
        stage3_5_resolved,
        stage4_resolved,
    ) = stage_counts

    # Build PipelineMetrics with a valid timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    metrics = PipelineMetrics(
        timestamp=timestamp,
        total_papers=total_papers,
        stage1_resolved=stage1_resolved,
        stage2_resolved=stage2_resolved,
        gate_resolved=gate_resolved,
        stage3_resolved=stage3_resolved,
        stage3_5_resolved=stage3_5_resolved,
        stage4_resolved=stage4_resolved,
        llm_calls=llm_calls,
        semantic_cache_hits=semantic_cache_hits,
        batch_count=batch_count,
        batch_retries=batch_retries,
        embedding_store_positive=embedding_store_positive,
        embedding_store_negative=embedding_store_negative,
        avg_embedding_latency_ms=avg_latency,
        p95_embedding_latency_ms=p95_latency,
    )

    # Write to a temporary JSONL file
    tmp_dir = Path(tempfile.mkdtemp())
    jsonl_path = tmp_dir / "pipeline_runs.jsonl"
    logger = MetricsLogger(path=jsonl_path)
    logger.record(metrics)

    # Read back the last line
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) >= 1, "No record written to JSONL file"
    record = json.loads(lines[-1])

    # ── Required fields are present ───────────────────────────────────────
    required_fields = {
        "timestamp",
        "total_papers",
        "stage1_resolved",
        "stage2_resolved",
        "gate_resolved",
        "stage3_resolved",
        "stage3_5_resolved",
        "stage4_resolved",
        "llm_calls",
        "semantic_cache_hits",
        "batch_count",
        "batch_retries",
        "embedding_store_positive",
        "embedding_store_negative",
        "avg_embedding_latency_ms",
        "p95_embedding_latency_ms",
    }
    missing = required_fields - set(record.keys())
    assert not missing, f"Missing required fields: {missing}"

    # ── timestamp is present and non-empty ────────────────────────────────
    assert record["timestamp"], "timestamp is empty"
    assert isinstance(record["timestamp"], str), "timestamp is not a string"

    # ── total_papers matches N ────────────────────────────────────────────
    assert record["total_papers"] == total_papers, (
        f"total_papers mismatch: expected {total_papers}, got {record['total_papers']}"
    )

    # ── Per-stage counts sum to total_papers ──────────────────────────────
    stage_sum = (
        record["stage1_resolved"]
        + record["stage2_resolved"]
        + record["gate_resolved"]
        + record["stage3_resolved"]
        + record["stage3_5_resolved"]
        + record["stage4_resolved"]
    )
    assert stage_sum == total_papers, (
        f"Stage resolution sum {stage_sum} != total_papers {total_papers}.\n"
        f"  stage1={record['stage1_resolved']}, stage2={record['stage2_resolved']}, "
        f"gate={record['gate_resolved']}, stage3={record['stage3_resolved']}, "
        f"stage3_5={record['stage3_5_resolved']}, stage4={record['stage4_resolved']}"
    )

    # ── All values match what was passed in ─────────────────────────────────
    assert record["timestamp"] == timestamp
    assert record["stage1_resolved"] == stage1_resolved
    assert record["stage2_resolved"] == stage2_resolved
    assert record["gate_resolved"] == gate_resolved
    assert record["stage3_resolved"] == stage3_resolved
    assert record["stage3_5_resolved"] == stage3_5_resolved
    assert record["stage4_resolved"] == stage4_resolved
    assert record["llm_calls"] == llm_calls
    assert record["semantic_cache_hits"] == semantic_cache_hits
    assert record["batch_count"] == batch_count
    assert record["batch_retries"] == batch_retries
    assert record["embedding_store_positive"] == embedding_store_positive
    assert record["embedding_store_negative"] == embedding_store_negative
    assert record["avg_embedding_latency_ms"] == pytest.approx(avg_latency)
    assert record["p95_embedding_latency_ms"] == pytest.approx(p95_latency)

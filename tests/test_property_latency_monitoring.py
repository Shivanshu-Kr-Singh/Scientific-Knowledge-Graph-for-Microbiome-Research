"""
Property Tests for Latency Monitoring

**Validates: Requirements 13.1, 13.2**

Property 17: Embedding Query Latency Recording
  Each query records a positive float duration (ms); rolling average > 200ms
  triggers a warning log.

  For any N queries executed against the EmbeddingStore:
  1. The internal `_query_latencies` list SHALL have exactly N entries
  2. All recorded latencies SHALL be positive floats (> 0)
  3. WHEN the rolling average of recorded latencies exceeds 200ms,
     a warning log SHALL be emitted
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from loguru import logger

from collectors.embedding_store import EmbeddingStore, EmbeddingMetadata


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Number of queries to execute (1-20)
_num_queries_st = st.integers(min_value=1, max_value=20)

# Float32 vector of dimension 384 with values in [-1, 1], no NaN/Inf.
_vector_st = arrays(
    dtype=np.float32,
    shape=(384,),
    elements=st.floats(
        min_value=-1.0, max_value=1.0,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False, width=32,
    ),
)

# Simulated high latency values (> 200ms) for warning trigger tests
_high_latency_st = st.floats(min_value=201.0, max_value=5000.0, allow_nan=False, allow_infinity=False)

# List of high latencies (to inject for warning trigger testing)
_high_latencies_list_st = st.lists(
    _high_latency_st,
    min_size=1,
    max_size=50,
)


# ---------------------------------------------------------------------------
# Property 17: Embedding Query Latency Recording
# **Validates: Requirements 13.1, 13.2**
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=15000)
@given(num_queries=_num_queries_st, query_vector=_vector_st)
def test_property_latency_recording_count_and_positivity(
    tmp_path_factory,
    num_queries: int,
    query_vector: np.ndarray,
) -> None:
    """
    **Property 17: Embedding Query Latency Recording (Part 1 - Count & Positivity)**

    **Validates: Requirements 13.1**

    For any N queries executed, `_query_latencies` has exactly N entries,
    and all entries are positive floats.
    """
    assume(np.linalg.norm(query_vector) > 1e-7)

    store_dir = tmp_path_factory.mktemp("store_latency_count")
    store = EmbeddingStore(store_dir=store_dir)

    # Add a few embeddings to ensure queries do real work
    for i in range(3):
        vec = np.random.default_rng(seed=i).standard_normal(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        meta = EmbeddingMetadata(
            doi=f"10.1000/latency-test-{i}",
            pmid=str(30000000 + i),
            title=f"Latency Test Paper {i}",
            partition="positive",
            added_at="2024-01-01T00:00:00Z",
        )
        store.append(vec, meta)

    # Execute N similarity queries
    for _ in range(num_queries):
        store.query_similar(query_vector, partition="positive", top_k=3)

    # Verify: exactly N latency entries recorded
    assert len(store._query_latencies) == num_queries, (
        f"Expected {num_queries} latency entries, got {len(store._query_latencies)}"
    )

    # Verify: all latency values are positive floats
    for idx, latency in enumerate(store._query_latencies):
        assert isinstance(latency, float), (
            f"Latency entry {idx} is not a float: {type(latency)}"
        )
        assert latency > 0.0, (
            f"Latency entry {idx} is not positive: {latency}"
        )


@settings(max_examples=30, deadline=15000)
@given(high_latencies=_high_latencies_list_st)
def test_property_latency_warning_on_high_rolling_avg(
    tmp_path_factory,
    high_latencies: list[float],
) -> None:
    """
    **Property 17: Embedding Query Latency Recording (Part 2 - Warning Emission)**

    **Validates: Requirements 13.2**

    WHEN rolling average latency exceeds 200ms, a warning log SHALL be emitted.
    We inject high latency values directly via `_record_latency` to simulate
    slow queries and verify the warning is triggered.
    """
    store_dir = tmp_path_factory.mktemp("store_latency_warn")
    store = EmbeddingStore(store_dir=store_dir)

    # Capture loguru warnings using a list sink
    captured_warnings: list[str] = []

    def warning_sink(message):
        if message.record["level"].name == "WARNING":
            captured_warnings.append(str(message))

    sink_id = logger.add(warning_sink, level="WARNING", format="{message}")

    try:
        # Inject high latencies (all > 200ms) to guarantee rolling avg > 200ms
        for latency_ms in high_latencies:
            store._record_latency(latency_ms)

        # After injecting all high latencies, the rolling avg must exceed 200ms
        avg = store._rolling_avg_latency_ms()
        assert avg > 200.0, (
            f"Rolling average should exceed 200ms with injected latencies, got {avg:.2f}ms"
        )

        # Verify warning was emitted (at least once)
        latency_warnings = [
            msg for msg in captured_warnings if "Latency warning" in msg
        ]
        assert len(latency_warnings) > 0, (
            f"Expected at least one latency warning log when rolling avg={avg:.2f}ms > 200ms, "
            f"but found none. Captured warnings: {captured_warnings}"
        )
    finally:
        logger.remove(sink_id)


@settings(max_examples=20, deadline=15000)
@given(num_queries=st.integers(min_value=1, max_value=10), query_vector=_vector_st)
def test_property_latency_no_warning_on_low_latency(
    tmp_path_factory,
    num_queries: int,
    query_vector: np.ndarray,
) -> None:
    """
    **Property 17: Embedding Query Latency Recording (Part 3 - No Spurious Warning)**

    **Validates: Requirements 13.1, 13.2**

    WHEN rolling average latency is well below 200ms (normal fast queries on
    a small in-memory store), no warning log SHALL be emitted.
    """
    assume(np.linalg.norm(query_vector) > 1e-7)

    store_dir = tmp_path_factory.mktemp("store_latency_no_warn")
    store = EmbeddingStore(store_dir=store_dir)

    # Add a few embeddings - small store ensures fast queries (< 200ms)
    for i in range(3):
        vec = np.random.default_rng(seed=i + 100).standard_normal(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        meta = EmbeddingMetadata(
            doi=f"10.1000/no-warn-{i}",
            pmid=str(40000000 + i),
            title=f"Fast Query Paper {i}",
            partition="positive",
            added_at="2024-01-01T00:00:00Z",
        )
        store.append(vec, meta)

    # Capture loguru warnings using a list sink
    captured_warnings: list[str] = []

    def warning_sink(message):
        if message.record["level"].name == "WARNING":
            captured_warnings.append(str(message))

    sink_id = logger.add(warning_sink, level="WARNING", format="{message}")

    try:
        # Execute queries on a tiny in-memory store (should be sub-millisecond)
        for _ in range(num_queries):
            store.query_similar(query_vector, partition="positive", top_k=3)

        # Verify rolling avg is well below threshold
        avg = store._rolling_avg_latency_ms()
        assert avg < 200.0, (
            f"Rolling average on a tiny store should be < 200ms, got {avg:.2f}ms"
        )

        # Verify NO latency warning was emitted
        latency_warnings = [
            msg for msg in captured_warnings if "Latency warning" in msg
        ]
        assert len(latency_warnings) == 0, (
            f"No latency warning should be emitted when avg={avg:.2f}ms < 200ms, "
            f"but found: {latency_warnings}"
        )
    finally:
        logger.remove(sink_id)

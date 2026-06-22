"""
Property Tests for Semantic Cache

**Validates: Requirements 8.1, 8.2, 8.4**

Property 12: Semantic Cache Threshold Correctness
  For any stored embedding, a query vector with cosine similarity > 0.97
  SHALL return a cache hit, and similarity ≤ 0.97 SHALL return a miss.

Property 13: Semantic Cache Growth
  After storing a verdict, cache size increases by exactly one.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from collectors.llm_verifier import SemanticCache, LLMVerdict


# ---------------------------------------------------------------------------
# Fake paper object for testing
# ---------------------------------------------------------------------------

@dataclass
class FakePaper:
    """Minimal paper object with doi, pmid, title attributes."""
    doi: str = "10.1000/test-paper"
    pmid: str = "99999999"
    title: str = "Test Paper on Gut Microbiome"


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Embedding dimension — use a smaller dimension for fast tests
_EMBED_DIM = 128

# Strategy for generating non-zero embedding vectors
_embedding_st = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=_EMBED_DIM,
    max_size=_EMBED_DIM,
).map(lambda xs: np.array(xs, dtype=np.float32))

# Strategy for similarity values clearly above threshold (hit)
_sim_above_threshold_st = st.floats(
    min_value=0.971, max_value=0.9999, allow_nan=False, allow_infinity=False
)

# Strategy for similarity values at or below threshold (miss)
_sim_at_or_below_threshold_st = st.floats(
    min_value=0.0, max_value=0.97, allow_nan=False, allow_infinity=False
)

# Strategy for generating verdicts
_verdict_st = st.builds(
    LLMVerdict,
    keep=st.booleans(),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    reason=st.text(min_size=1, max_size=30),
)


# ---------------------------------------------------------------------------
# Helper: Create a query vector with known cosine similarity to a reference
# ---------------------------------------------------------------------------

def _make_query_with_similarity(reference: np.ndarray, target_similarity: float) -> np.ndarray:
    """
    Given a reference vector and a target cosine similarity in (0, 1),
    construct a query vector whose cosine similarity to reference equals
    target_similarity (within floating point tolerance).

    Strategy: query = alpha * reference + beta * orthogonal
    where alpha and beta are chosen to achieve the target cosine similarity.
    """
    ref = reference.astype(np.float64)
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        return reference.copy()

    # Create an orthogonal vector using Gram-Schmidt on a random vector
    rng = np.random.default_rng(42)
    random_vec = rng.standard_normal(len(ref))
    # Subtract projection onto reference
    proj = np.dot(random_vec, ref) / (ref_norm ** 2) * ref
    orthogonal = random_vec - proj
    orth_norm = np.linalg.norm(orthogonal)
    if orth_norm == 0:
        # Degenerate case: try another random vector
        random_vec = rng.standard_normal(len(ref))
        proj = np.dot(random_vec, ref) / (ref_norm ** 2) * ref
        orthogonal = random_vec - proj
        orth_norm = np.linalg.norm(orthogonal)

    # Normalize both
    ref_unit = ref / ref_norm
    orth_unit = orthogonal / orth_norm

    # query = cos(theta) * ref_unit + sin(theta) * orth_unit
    # cosine_similarity(query, ref_unit) = cos(theta) = target_similarity
    cos_theta = target_similarity
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta ** 2))

    query = cos_theta * ref_unit + sin_theta * orth_unit
    return query.astype(np.float32)


# ---------------------------------------------------------------------------
# Property 12: Semantic Cache Threshold Correctness
# **Validates: Requirements 8.1, 8.2**
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    base_embedding=_embedding_st,
    target_sim_hit=_sim_above_threshold_st,
    verdict=_verdict_st,
)
def test_property_semantic_cache_threshold_hit(
    base_embedding: np.ndarray,
    target_sim_hit: float,
    verdict: LLMVerdict,
) -> None:
    """
    **Property 12 (hit case): Semantic Cache Threshold Correctness**

    **Validates: Requirements 8.1, 8.2**

    When a query vector has cosine similarity > 0.97 with a stored embedding,
    the cache SHALL return a hit (non-None verdict).
    """
    # Skip zero-norm embeddings (would break cosine similarity)
    assume(np.linalg.norm(base_embedding) > 1e-6)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(store_dir=Path(tmpdir))

        # Store the base embedding with a verdict
        paper = FakePaper()
        cache.store_verdict(base_embedding, verdict, paper)

        # Create a query vector with similarity > 0.97
        query = _make_query_with_similarity(base_embedding, target_sim_hit)

        # Verify the query has the expected similarity
        actual_sim = float(
            np.dot(base_embedding, query)
            / (np.linalg.norm(base_embedding) * np.linalg.norm(query))
        )

        # Only assert if our constructed vector actually achieved > 0.97
        # (floating point may introduce small drift)
        assume(actual_sim > 0.97)

        result = cache.lookup(query)
        assert result is not None, (
            f"Expected cache hit for similarity={actual_sim:.6f} > 0.97, "
            f"but got None"
        )


@settings(max_examples=50)
@given(
    base_embedding=_embedding_st,
    target_sim_miss=_sim_at_or_below_threshold_st,
    verdict=_verdict_st,
)
def test_property_semantic_cache_threshold_miss(
    base_embedding: np.ndarray,
    target_sim_miss: float,
    verdict: LLMVerdict,
) -> None:
    """
    **Property 12 (miss case): Semantic Cache Threshold Correctness**

    **Validates: Requirements 8.1, 8.2**

    When a query vector has cosine similarity ≤ 0.97 with all stored embeddings,
    the cache SHALL return a miss (None).
    """
    # Skip zero-norm embeddings (would break cosine similarity)
    assume(np.linalg.norm(base_embedding) > 1e-6)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(store_dir=Path(tmpdir))

        # Store the base embedding with a verdict
        paper = FakePaper()
        cache.store_verdict(base_embedding, verdict, paper)

        # Create a query vector with similarity ≤ 0.97
        query = _make_query_with_similarity(base_embedding, target_sim_miss)

        # Verify the query has the expected similarity
        actual_sim = float(
            np.dot(base_embedding, query)
            / (np.linalg.norm(base_embedding) * np.linalg.norm(query))
        )

        # Only assert if our constructed vector actually achieved ≤ 0.97
        assume(actual_sim <= 0.97)

        result = cache.lookup(query)
        assert result is None, (
            f"Expected cache miss for similarity={actual_sim:.6f} ≤ 0.97, "
            f"but got a verdict: keep={result.keep}"
        )


# ---------------------------------------------------------------------------
# Property 13: Semantic Cache Growth
# **Validates: Requirements 8.4**
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    embedding=_embedding_st,
    verdict=_verdict_st,
)
def test_property_semantic_cache_growth(
    embedding: np.ndarray,
    verdict: LLMVerdict,
) -> None:
    """
    **Property 13: Semantic Cache Growth**

    **Validates: Requirements 8.4**

    After storing a verdict, the cache size SHALL increase by exactly one.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = SemanticCache(store_dir=Path(tmpdir))

        size_before = cache.size
        assert size_before == 0, f"Fresh cache should have size 0, got {size_before}"

        paper = FakePaper()
        cache.store_verdict(embedding, verdict, paper)

        size_after = cache.size
        assert size_after == size_before + 1, (
            f"After store_verdict, cache size should be {size_before + 1}, "
            f"got {size_after}"
        )

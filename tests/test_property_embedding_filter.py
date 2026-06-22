"""
Property Tests for Embedding Filter (Stage 3.5 Classification)

**Validates: Requirements 5.3, 5.4, 5.5**

Property 9: Stage 3.5 Classification Threshold Invariant
  For any pair of similarity scores (pos_similarity, neg_similarity) where both
  partitions have ≥50 embeddings:
  - If pos_similarity ≥ 0.85 AND neg_similarity < 0.60 → decision SHALL be KEEP
  - If neg_similarity ≥ 0.85 AND pos_similarity < 0.60 → decision SHALL be REJECT
  - Otherwise → decision SHALL be BORDERLINE
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from collectors.embedding_filter import EmbeddingFilter, EmbeddingVerdict
from collectors.embedding_store import SimilarityResult, EmbeddingMetadata


# ---------------------------------------------------------------------------
# Fake paper object for testing
# ---------------------------------------------------------------------------

@dataclass
class FakePaper:
    """Minimal paper object with title and abstract attributes."""
    title: str = "Test Paper on Gut Microbiome"
    abstract: str = "This paper studies the human gut microbiome in IBD patients."


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Similarity scores in [0.0, 1.0] range
_sim_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 9: Stage 3.5 Classification Threshold Invariant
# **Validates: Requirements 5.3, 5.4, 5.5**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(pos_sim=_sim_score_st, neg_sim=_sim_score_st)
def test_property_stage3_5_classification_threshold_invariant(
    pos_sim: float,
    neg_sim: float,
) -> None:
    """
    **Property 9: Stage 3.5 Classification Threshold Invariant**

    **Validates: Requirements 5.3, 5.4, 5.5**

    For any (pos_sim, neg_sim) pair where both partitions have ≥50 embeddings,
    the decision matches the threshold rules exactly:
      - pos_sim >= 0.85 and neg_sim < 0.60 → "KEEP"
      - neg_sim >= 0.85 and pos_sim < 0.60 → "REJECT"
      - otherwise → "BORDERLINE"
    """
    # Mock the EmbeddingModel
    mock_model = MagicMock()
    mock_model.encode_paper.return_value = np.zeros(384, dtype=np.float32)

    # Mock the EmbeddingStore
    mock_store = MagicMock()
    # Both partitions have ≥50 embeddings (above MIN_PARTITION_SIZE)
    type(mock_store).positive_count = PropertyMock(return_value=100)
    type(mock_store).negative_count = PropertyMock(return_value=100)

    # Configure query_similar to return the hypothesis-generated similarity scores
    pos_result = SimilarityResult(
        score=pos_sim,
        metadata=EmbeddingMetadata(
            doi="10.1000/pos-example",
            pmid="12345678",
            title="Positive reference paper",
            partition="positive",
            added_at="2024-01-01T00:00:00Z",
        ),
    )
    neg_result = SimilarityResult(
        score=neg_sim,
        metadata=EmbeddingMetadata(
            doi="10.1000/neg-example",
            pmid="87654321",
            title="Negative reference paper",
            partition="negative",
            added_at="2024-01-01T00:00:00Z",
        ),
    )

    def mock_query_similar(vector, partition, top_k=1):
        if partition == "positive":
            return [pos_result]
        elif partition == "negative":
            return [neg_result]
        return []

    mock_store.query_similar.side_effect = mock_query_similar

    # Create the EmbeddingFilter with mocked dependencies
    embedding_filter = EmbeddingFilter(mock_model, mock_store)

    # Evaluate the fake paper
    paper = FakePaper()
    verdict = embedding_filter.evaluate(paper)

    # Determine expected decision based on threshold logic
    if pos_sim >= 0.85 and neg_sim < 0.60:
        expected_decision = "KEEP"
    elif neg_sim >= 0.85 and pos_sim < 0.60:
        expected_decision = "REJECT"
    else:
        expected_decision = "BORDERLINE"

    # Assert the decision matches the threshold rules exactly
    assert verdict.decision == expected_decision, (
        f"Decision mismatch for pos_sim={pos_sim:.4f}, neg_sim={neg_sim:.4f}: "
        f"expected '{expected_decision}', got '{verdict.decision}'"
    )

    # Also verify the similarity scores are reported correctly
    assert verdict.pos_similarity == pos_sim, (
        f"pos_similarity mismatch: expected {pos_sim}, got {verdict.pos_similarity}"
    )
    assert verdict.neg_similarity == neg_sim, (
        f"neg_similarity mismatch: expected {neg_sim}, got {verdict.neg_similarity}"
    )

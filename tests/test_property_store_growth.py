"""
Property Tests for Embedding Store Growth (Active Store Growth Logic)

**Validates: Requirements 9.1, 9.2, 9.3, 9.4**

Property 14: Embedding Store Growth Threshold Logic
  For any relevance score in [0.0, 1.0] and any duplicate status (True/False):
  - score >= 0.80 AND NOT duplicate → embedding appended to positive partition
  - score <= 0.20 AND NOT duplicate → embedding appended to negative partition
  - 0.20 < score < 0.80 → nothing appended (skip)
  - duplicate (any score) → nothing appended (skip)
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Fake paper object for testing
# ---------------------------------------------------------------------------

@dataclass
class FakePaper:
    """Minimal paper object with title, abstract, doi, and pmid attributes."""
    title: str = "Test Paper on Gut Microbiome"
    abstract: str = "This paper studies the human gut microbiome in IBD patients."
    doi: str = "10.1000/test-doi-12345"
    pmid: str = "99999999"


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Relevance scores in [0.0, 1.0] range
_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Whether the paper already exists in the store (duplicate)
_is_duplicate_st = st.booleans()


# ---------------------------------------------------------------------------
# Property 14: Embedding Store Growth Threshold Logic
# **Validates: Requirements 9.1, 9.2, 9.3, 9.4**
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(score=_score_st, is_duplicate=_is_duplicate_st)
def test_property_embedding_store_growth_threshold_logic(
    score: float,
    is_duplicate: bool,
) -> None:
    """
    **Property 14: Embedding Store Growth Threshold Logic**

    **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

    For any relevance score in [0.0, 1.0] and any duplicate status:
      - score >= 0.80 and NOT duplicate → embedding appended to positive partition
      - score <= 0.20 and NOT duplicate → embedding appended to negative partition
      - 0.20 < score < 0.80 → nothing appended (skip regardless of duplicate)
      - duplicate → nothing appended regardless of score
    """
    # Create fake paper
    paper = FakePaper()

    # Mock the embedding store
    mock_store = MagicMock()
    mock_store.contains.return_value = is_duplicate

    # Mock the embedding model
    mock_model = MagicMock()
    fake_embedding = np.random.rand(384).astype(np.float32)
    mock_model.encode_paper.return_value = fake_embedding

    # Patch the lazy singleton getters
    with patch(
        "collectors.relevance_filter._get_embedding_store", return_value=mock_store
    ), patch(
        "collectors.relevance_filter._get_embedding_model", return_value=mock_model
    ):
        from collectors.relevance_filter import RelevanceFilter

        # Create a RelevanceFilter instance and call the growth method
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf._embedding_store_growth(paper, score)

    # Determine expected behavior
    is_borderline = 0.20 < score < 0.80

    if is_borderline:
        # Borderline scores → no store interaction beyond the early return
        mock_store.contains.assert_not_called()
        mock_store.append.assert_not_called()
        mock_model.encode_paper.assert_not_called()
    elif is_duplicate:
        # Duplicate paper → store.contains called, but no append
        mock_store.contains.assert_called_once_with(doi=paper.doi, pmid=paper.pmid)
        mock_store.append.assert_not_called()
        mock_model.encode_paper.assert_not_called()
    else:
        # Not borderline and not duplicate → should encode and append
        mock_store.contains.assert_called_once_with(doi=paper.doi, pmid=paper.pmid)
        mock_model.encode_paper.assert_called_once_with(paper.title, paper.abstract)
        mock_store.append.assert_called_once()

        # Verify correct partition
        call_kwargs = mock_store.append.call_args[1]
        metadata = call_kwargs["metadata"]

        if score >= 0.80:
            assert metadata.partition == "positive", (
                f"Expected positive partition for score={score:.4f}, "
                f"got '{metadata.partition}'"
            )
        elif score <= 0.20:
            assert metadata.partition == "negative", (
                f"Expected negative partition for score={score:.4f}, "
                f"got '{metadata.partition}'"
            )

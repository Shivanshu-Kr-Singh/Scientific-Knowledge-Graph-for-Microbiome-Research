"""
Property Tests for Batched LLM Verifier — Batch Size Invariant

**Validates: Requirements 7.1**

Property 11: Batch Size Invariant
  For any list of N papers requiring LLM verification, the Batched Verifier
  SHALL partition them into ceil(N/16) batches where each batch contains at
  most 16 papers, and the union of all batches equals the original set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Fake paper object for testing
# ---------------------------------------------------------------------------

@dataclass
class FakePaper:
    """Minimal paper object with a title attribute for batch splitting tests."""
    title: str


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Generate lists of N papers where N ∈ [0, 200]
_paper_list_st = st.integers(min_value=0, max_value=200).flatmap(
    lambda n: st.just([FakePaper(title=f"Paper_{i}") for i in range(n)])
)


# ---------------------------------------------------------------------------
# Property 11: Batch Size Invariant
# **Validates: Requirements 7.1**
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(papers=_paper_list_st)
def test_property_batch_size_invariant(papers: list) -> None:
    """
    **Property 11: Batch Size Invariant**

    **Validates: Requirements 7.1**

    For any N papers, partitioned into ceil(N/16) batches of ≤16,
    the union of all batches equals the original set:
      1. Number of batches == ceil(N / 16)
      2. No batch exceeds 16 papers
      3. The union of all batches equals the original set (all papers
         accounted for, no duplicates, order preserved)
    """
    from collectors.llm_verifier import BatchedVerifier

    # Bypass __init__ to avoid config/environment dependencies
    with patch.object(BatchedVerifier, "__init__", lambda self: None):
        verifier = BatchedVerifier()
        verifier._max_batch_size = 16

        # Execute the method under test
        batches = verifier._split_into_batches(papers)

        n = len(papers)

        # Property 1: Number of batches equals ceil(N / 16)
        expected_batch_count = math.ceil(n / 16) if n > 0 else 0
        assert len(batches) == expected_batch_count, (
            f"Expected {expected_batch_count} batches for {n} papers, "
            f"got {len(batches)}"
        )

        # Property 2: No batch exceeds 16 papers
        for i, batch in enumerate(batches):
            assert len(batch) <= 16, (
                f"Batch {i} has {len(batch)} papers, exceeding max of 16"
            )

        # Property 3: Union of all batches equals the original set
        # (all papers accounted for, no duplicates, order preserved)
        reconstructed = []
        for batch in batches:
            reconstructed.extend(batch)

        assert len(reconstructed) == n, (
            f"Total papers in batches ({len(reconstructed)}) != "
            f"original count ({n})"
        )

        # Verify identity equality — same objects, same order
        for idx in range(n):
            assert reconstructed[idx] is papers[idx], (
                f"Paper at index {idx} differs: "
                f"expected '{papers[idx].title}', "
                f"got '{reconstructed[idx].title}'"
            )

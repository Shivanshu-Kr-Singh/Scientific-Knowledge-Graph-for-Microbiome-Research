"""
Property Tests for Batched LLM Verifier (Batch Size Invariant)

**Validates: Requirements 7.1**

Property 11: Batch Size Invariant
  For any N papers (1 to 200), the BatchedVerifier._split_into_batches() method:
  - Produces ceil(N / 16) batches
  - Each batch has at most 16 papers
  - The union of all batches equals the original set (no papers lost, no duplicates)
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
    """Minimal paper object with unique identifier for set membership checks."""
    title: str
    abstract: str = "A study on microbiome diversity."


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Generate a list of 1–200 unique papers
_paper_list_st = st.integers(min_value=1, max_value=200).flatmap(
    lambda n: st.just(
        [FakePaper(title=f"Paper {i}") for i in range(n)]
    )
)


# ---------------------------------------------------------------------------
# Property 11: Batch Size Invariant
# **Validates: Requirements 7.1**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(papers=_paper_list_st)
def test_property_batch_size_invariant(papers: list) -> None:
    """
    **Property 11: Batch Size Invariant**

    **Validates: Requirements 7.1**

    For any list of N papers (1 to 200), the BatchedVerifier._split_into_batches()
    method produces ceil(N/16) batches where:
    - Each batch has at most 16 papers
    - The union of all batches equals the original set (no papers lost, no duplicates)
    """
    from collectors.llm_verifier import BatchedVerifier

    # Patch config imports to avoid needing actual config/env setup
    with patch("collectors.llm_verifier.OLLAMA_BASE_URL", "http://localhost:11434"), \
         patch("collectors.llm_verifier.OLLAMA_MODEL", "llama3"):
        verifier = BatchedVerifier()

    N = len(papers)
    MAX_BATCH_SIZE = 16

    # Call the splitting method directly
    batches = verifier._split_into_batches(papers)

    # 1) Number of batches must equal ceil(N / 16)
    expected_batch_count = math.ceil(N / MAX_BATCH_SIZE)
    assert len(batches) == expected_batch_count, (
        f"Expected {expected_batch_count} batches for {N} papers, got {len(batches)}"
    )

    # 2) Each batch has at most 16 papers
    for i, batch in enumerate(batches):
        assert len(batch) <= MAX_BATCH_SIZE, (
            f"Batch {i} has {len(batch)} papers, exceeds max of {MAX_BATCH_SIZE}"
        )
        # Also verify each batch is non-empty
        assert len(batch) > 0, f"Batch {i} is empty"

    # 3) Union of all batches equals original set (no papers lost, no duplicates)
    # Flatten all batches
    all_papers_from_batches = []
    for batch in batches:
        all_papers_from_batches.extend(batch)

    # Same total count (no papers lost or duplicated)
    assert len(all_papers_from_batches) == N, (
        f"Total papers in batches ({len(all_papers_from_batches)}) != original count ({N})"
    )

    # Verify by identity — each paper object appears exactly once
    original_ids = [id(p) for p in papers]
    batch_ids = [id(p) for p in all_papers_from_batches]
    assert sorted(original_ids) == sorted(batch_ids), (
        "Papers in batches do not match original set by identity"
    )

    # Verify order is preserved (batches are sequential slices)
    for i, paper in enumerate(all_papers_from_batches):
        assert paper is papers[i], (
            f"Order mismatch at position {i}: batch paper is not the same as original"
        )

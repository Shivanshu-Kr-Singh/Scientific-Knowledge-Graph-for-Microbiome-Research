"""
Tests for FuzzyMatcher — Levenshtein edit-distance matching.

Includes:
  - Unit tests: basic confidence formula, short-form skip, match results.
  - Property 7: Fuzzy Match Confidence Formula
    **Validates: Requirements 12.3**

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.fuzzy_matcher import FuzzyMatcher


# ---------------------------------------------------------------------------
# Unit tests — compute_confidence()
# ---------------------------------------------------------------------------


def test_compute_confidence_zero_distance() -> None:
    """edit_distance=0 always yields confidence=1.0."""
    assert FuzzyMatcher.compute_confidence(0, 10, 10) == pytest.approx(1.0)
    assert FuzzyMatcher.compute_confidence(0, 4, 8) == pytest.approx(1.0)


def test_compute_confidence_distance_one() -> None:
    """edit_distance=1 with equal lengths of 4 yields 1.0 - (1/4)*0.5 = 0.875."""
    result = FuzzyMatcher.compute_confidence(1, 4, 4)
    expected = 1.0 - (1 / 4) * 0.5
    assert result == pytest.approx(expected)


def test_compute_confidence_distance_two() -> None:
    """edit_distance=2 with max length 10 yields 1.0 - (2/10)*0.5 = 0.9."""
    result = FuzzyMatcher.compute_confidence(2, 10, 8)
    expected = 1.0 - (2 / 10) * 0.5
    assert result == pytest.approx(expected)


def test_compute_confidence_uses_max_length() -> None:
    """The formula uses max(len_surface, len_candidate), not min."""
    # max(4, 10) = 10
    result = FuzzyMatcher.compute_confidence(1, 4, 10)
    expected = 1.0 - (1 / 10) * 0.5
    assert result == pytest.approx(expected)

    # max(10, 4) = 10 — symmetric
    result2 = FuzzyMatcher.compute_confidence(1, 10, 4)
    assert result2 == pytest.approx(expected)


def test_compute_confidence_result_in_unit_interval() -> None:
    """Result is always in [0.0, 1.0]."""
    for d in range(3):
        for l in range(4, 20):
            result = FuzzyMatcher.compute_confidence(d, l, l)
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Property 7: Fuzzy Match Confidence Formula
# **Validates: Requirements 12.3**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    edit_distance=st.integers(min_value=0, max_value=2),
    len_surface=st.integers(min_value=4),
    len_candidate=st.integers(min_value=4),
)
def test_property_fuzzy_match_confidence_formula(
    edit_distance: int,
    len_surface: int,
    len_candidate: int,
) -> None:
    """
    **Property 7: Fuzzy Match Confidence Formula**

    **Validates: Requirements 12.3**

    For any (edit_distance ∈ {0, 1, 2}, len_surface ≥ 4, len_candidate ≥ 4):
    - compute_confidence() equals 1.0 - (d / max(len_s, len_c)) * 0.5
      within floating-point tolerance.
    - The result is in [0.0, 1.0].
    """
    result = FuzzyMatcher.compute_confidence(edit_distance, len_surface, len_candidate)

    # Compute expected value using the formula from Requirement 12.3
    expected = 1.0 - (edit_distance / max(len_surface, len_candidate)) * 0.5

    # Assert formula correctness within floating-point tolerance
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"compute_confidence({edit_distance}, {len_surface}, {len_candidate}) = {result}, "
        f"expected {expected}"
    )

    # Assert result is in [0.0, 1.0]
    assert 0.0 <= result <= 1.0, (
        f"compute_confidence({edit_distance}, {len_surface}, {len_candidate}) = {result} "
        f"is outside [0.0, 1.0]"
    )

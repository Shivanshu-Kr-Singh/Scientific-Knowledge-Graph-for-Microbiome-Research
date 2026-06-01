"""
Tests for AbbreviationExpander — curated abbreviation table + genus-initial
pattern matching.

Includes:
  - Unit tests: exact match, genus-initial pattern, add_mapping hot-reload,
    empty input, no-match returns empty list.
  - Property 12: Abbreviation Confidence Proportionality
    **Validates: Requirements 11.2, 11.4**

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

from __future__ import annotations

import sqlite3

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.abbreviation_expander import AbbreviationExpander
from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_canonical_registry_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def expander_conn() -> sqlite3.Connection:
    """In-memory SQLite connection with the canonical_registry schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_canonical_registry_schema())
    yield conn
    conn.close()


@pytest.fixture
def expander(expander_conn: sqlite3.Connection) -> AbbreviationExpander:
    """Fresh AbbreviationExpander backed by an in-memory database."""
    return AbbreviationExpander(conn=expander_conn)


def _fresh_expander() -> AbbreviationExpander:
    """Create an AbbreviationExpander backed by a fresh in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_canonical_registry_schema())
    return AbbreviationExpander(conn=conn)


# ---------------------------------------------------------------------------
# Unit tests — basic expand() behaviour
# ---------------------------------------------------------------------------


def test_expand_empty_string_returns_empty(expander: AbbreviationExpander) -> None:
    """expand('') returns an empty list without raising."""
    result = expander.expand("")
    assert result == []


def test_expand_no_match_returns_empty(expander: AbbreviationExpander) -> None:
    """expand() returns [] when no mapping exists for the surface form."""
    result = expander.expand("E. coli")
    assert result == []


def test_expand_exact_match_single_candidate(expander: AbbreviationExpander) -> None:
    """Exact match with one full form returns confidence=1.0."""
    expander.add_mapping("SCFA", "short-chain fatty acid", added_by="test")
    candidates = expander.expand("SCFA")
    assert len(candidates) == 1
    assert candidates[0].expanded_form == "short-chain fatty acid"
    assert candidates[0].confidence == pytest.approx(1.0)


def test_expand_exact_match_two_candidates(expander: AbbreviationExpander) -> None:
    """Exact match with two full forms returns confidence=0.5 each."""
    expander.add_mapping("IBD", "inflammatory bowel disease", added_by="test")
    expander.add_mapping("IBD", "irritable bowel disorder", added_by="test")
    candidates = expander.expand("IBD")
    assert len(candidates) == 2
    for c in candidates:
        assert c.confidence == pytest.approx(0.5)


def test_expand_genus_initial_single_genus(expander: AbbreviationExpander) -> None:
    """Genus-initial pattern with one matching genus returns confidence=1.0."""
    expander.add_mapping("E. coli", "Escherichia coli", added_by="test")
    candidates = expander.expand("E. test")
    assert len(candidates) == 1
    assert candidates[0].expanded_form == "Escherichia test"
    assert candidates[0].confidence == pytest.approx(1.0)


def test_expand_genus_initial_two_genera(expander: AbbreviationExpander) -> None:
    """Genus-initial pattern with two matching genera returns confidence=0.5 each."""
    expander.add_mapping("E. coli", "Escherichia coli", added_by="test")
    expander.add_mapping("E. faecalis", "Enterococcus faecalis", added_by="test")
    candidates = expander.expand("E. test")
    assert len(candidates) == 2
    for c in candidates:
        assert c.confidence == pytest.approx(0.5)


def test_expand_genus_initial_candidates_sorted_lexicographically(
    expander: AbbreviationExpander,
) -> None:
    """Candidates are returned in lexicographic order by expanded_form."""
    expander.add_mapping("E. coli", "Escherichia coli", added_by="test")
    expander.add_mapping("E. faecalis", "Enterococcus faecalis", added_by="test")
    candidates = expander.expand("E. test")
    forms = [c.expanded_form for c in candidates]
    assert forms == sorted(forms)


def test_expand_genus_initial_no_match_for_letter(expander: AbbreviationExpander) -> None:
    """Genus-initial pattern returns [] when no genus starts with the given letter."""
    expander.add_mapping("E. coli", "Escherichia coli", added_by="test")
    # No genus starting with "B" in the table
    candidates = expander.expand("B. subtilis")
    assert candidates == []


def test_expand_exact_match_takes_priority_over_genus_initial(
    expander: AbbreviationExpander,
) -> None:
    """
    When the surface form has an exact match in the curated table, the exact
    match is returned (not the genus-initial expansion).
    """
    expander.add_mapping("E. coli", "Escherichia coli", added_by="test")
    expander.add_mapping("E. faecalis", "Enterococcus faecalis", added_by="test")
    # "E. coli" has an exact match — should return that, not genus-initial expansion
    candidates = expander.expand("E. coli")
    assert len(candidates) == 1
    assert candidates[0].expanded_form == "Escherichia coli"
    assert candidates[0].confidence == pytest.approx(1.0)


def test_add_mapping_hot_reload(expander: AbbreviationExpander) -> None:
    """add_mapping() takes effect immediately without requiring a restart."""
    assert expander.expand("SCFA") == []
    expander.add_mapping("SCFA", "short-chain fatty acid", added_by="test")
    candidates = expander.expand("SCFA")
    assert len(candidates) == 1
    assert candidates[0].expanded_form == "short-chain fatty acid"


def test_add_mapping_idempotent(expander: AbbreviationExpander) -> None:
    """Adding the same mapping twice does not create duplicate candidates."""
    expander.add_mapping("SCFA", "short-chain fatty acid", added_by="test")
    expander.add_mapping("SCFA", "short-chain fatty acid", added_by="test")
    candidates = expander.expand("SCFA")
    assert len(candidates) == 1


def test_expand_genus_initial_deduplicates_same_genus(
    expander: AbbreviationExpander,
) -> None:
    """
    When multiple entries share the same genus, the genus is counted only once
    in the genus-initial expansion.
    """
    # Two entries with the same genus "Escherichia"
    expander.add_mapping("E. coli", "Escherichia coli", added_by="test")
    expander.add_mapping("E. fergusonii", "Escherichia fergusonii", added_by="test")
    # Only one unique genus "Escherichia" starts with "E"
    candidates = expander.expand("E. test")
    assert len(candidates) == 1
    assert candidates[0].expanded_form == "Escherichia test"
    assert candidates[0].confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Property 12: Abbreviation Confidence Proportionality
# **Validates: Requirements 11.2, 11.4**
# ---------------------------------------------------------------------------

# Pre-computed list of 10 distinct genera all starting with "A" for use in
# the property test. We use a fixed list to avoid generating random strings
# that might collide or have unexpected first characters.
_GENERA_STARTING_WITH_A = [
    "Aardvark",
    "Abacus",
    "Acacia",
    "Acinetobacter",
    "Actinomyces",
    "Aerococcus",
    "Agrobacterium",
    "Akkermansia",
    "Alcaligenes",
    "Alistipes",
]


@settings(max_examples=100)
@given(n=st.integers(min_value=1, max_value=10))
def test_property_abbreviation_confidence_proportionality(n: int) -> None:
    """
    **Property 12: Abbreviation Confidence Proportionality**

    **Validates: Requirements 11.2, 11.4**

    For N genera matching a genus-initial abbreviation:
    - Exactly N candidates are returned.
    - Each candidate has confidence == 1.0 / N.
    - When N=1, confidence == 1.0.
    """
    expander = _fresh_expander()

    # Select the first N genera from our pre-computed list
    genera = _GENERA_STARTING_WITH_A[:n]

    # Add a mapping for each genus so the genus-initial lookup can find them.
    # We map "A. <species>" -> "<Genus> <species>" for each genus.
    for genus in genera:
        species = genus.lower() + "ensis"  # e.g. "Aardvarkensis"
        abbreviated = f"A. {species}"
        full_form = f"{genus} {species}"
        expander.add_mapping(abbreviated, full_form, added_by="test")

    # Call expand with a novel species epithet that has no exact match,
    # triggering the genus-initial lookup path.
    candidates = expander.expand("A. test")

    # Assert exactly N candidates are returned
    assert len(candidates) == n, (
        f"Expected {n} candidates for {n} genera starting with 'A', "
        f"got {len(candidates)}: {[c.expanded_form for c in candidates]}"
    )

    # Assert each candidate has confidence == 1.0 / N
    expected_confidence = 1.0 / n
    for candidate in candidates:
        assert candidate.confidence == pytest.approx(expected_confidence), (
            f"Expected confidence {expected_confidence} (1.0/{n}), "
            f"got {candidate.confidence} for '{candidate.expanded_form}'"
        )

    # Explicit check: when N=1, confidence must be exactly 1.0
    if n == 1:
        assert candidates[0].confidence == pytest.approx(1.0), (
            f"When N=1, confidence must be 1.0, got {candidates[0].confidence}"
        )

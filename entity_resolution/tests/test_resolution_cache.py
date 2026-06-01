"""
Tests for ResolutionCache — two-tier LRU + SQLite cache with version-based
invalidation.

Includes:
  - Unit tests: get/put round-trip, LRU eviction, version mismatch, invalidate_version,
    SQLite persistence after clearing in-memory cache.
  - Property 11: Cache Version Invalidation
    **Validates: Requirements 2.5, 8.5**

Requirements: 2.5, 8.2, 8.3, 8.4, 8.5, 8.6
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_resolution_cache_schema,
)
from entity_resolution.models import CandidateScore, ResolutionResult
from entity_resolution.resolution_cache import ResolutionCache
from entity_resolution.utils import normalize_surface_form


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_STRATEGIES = [
    "manual_override",
    "exact",
    "normalized",
    "abbreviation",
    "synonym",
    "fuzzy",
    "ontology",
    "none",
]

_ENTITY_TYPES = ["taxon", "disease", "method"]


def _st_nonempty_text(max_size: int = 50) -> st.SearchStrategy[str]:
    """Non-empty printable text (letters, digits, spaces)."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
        min_size=1,
        max_size=max_size,
    ).map(str.strip).filter(lambda s: len(s) > 0)


def _st_surface_form() -> st.SearchStrategy[str]:
    """Generate a non-empty surface form string."""
    return _st_nonempty_text(80)


def _st_canonical_id() -> st.SearchStrategy[str]:
    """Generate a non-empty canonical ID string."""
    return _st_nonempty_text(30)


def _st_resolution_result(
    surface_form: Optional[str] = None,
    entity_type: Optional[str] = None,
) -> st.SearchStrategy[ResolutionResult]:
    """Generate an arbitrary ResolutionResult."""
    sf_strategy = st.just(surface_form) if surface_form else _st_surface_form()
    et_strategy = st.just(entity_type) if entity_type else st.sampled_from(_ENTITY_TYPES)
    return st.builds(
        ResolutionResult,
        surface_form=sf_strategy,
        entity_type=et_strategy,
        canonical_id=st.one_of(st.none(), _st_canonical_id()),
        grounded=st.booleans(),
        winning_strategy=st.sampled_from(_STRATEGIES),
        grounding_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        conflict_set=st.just([]),
        paper_id=_st_nonempty_text(40),
        timestamp=st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2099, 12, 31),
            timezones=st.just(timezone.utc),
        ),
        high_conflict=st.booleans(),
        hierarchy_match=st.booleans(),
        hierarchy_level=st.one_of(st.none(), st.integers(min_value=1, max_value=3)),
    )


def _st_registry_version() -> st.SearchStrategy[int]:
    """Generate a positive registry version integer."""
    return st.integers(min_value=1, max_value=1_000_000)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_conn() -> sqlite3.Connection:
    """In-memory SQLite connection with the resolution_cache schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_resolution_cache_schema())
    yield conn
    conn.close()


@pytest.fixture
def cache(cache_conn: sqlite3.Connection) -> ResolutionCache:
    """Fresh ResolutionCache backed by an in-memory database."""
    return ResolutionCache(capacity=10, conn=cache_conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    surface_form: str = "Escherichia coli",
    entity_type: str = "taxon",
    canonical_id: Optional[str] = "562",
    grounded: bool = True,
    winning_strategy: str = "exact",
    grounding_confidence: float = 0.95,
    paper_id: str = "paper-001",
) -> ResolutionResult:
    """Build a minimal ResolutionResult for testing."""
    return ResolutionResult(
        surface_form=surface_form,
        entity_type=entity_type,
        canonical_id=canonical_id,
        grounded=grounded,
        winning_strategy=winning_strategy,
        grounding_confidence=grounding_confidence,
        conflict_set=[],
        paper_id=paper_id,
        timestamp=datetime.now(timezone.utc),
        high_conflict=False,
        hierarchy_match=False,
        hierarchy_level=None,
    )


def _fresh_cache(capacity: int = 100) -> ResolutionCache:
    """Create a ResolutionCache backed by a fresh in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_resolution_cache_schema())
    return ResolutionCache(capacity=capacity, conn=conn)


# ---------------------------------------------------------------------------
# Unit tests — basic get/put round-trip
# ---------------------------------------------------------------------------


def test_get_returns_none_on_empty_cache(cache: ResolutionCache) -> None:
    """get() returns None when the cache is empty."""
    result = cache.get("Escherichia coli", current_registry_version=1)
    assert result is None


def test_put_then_get_returns_result(cache: ResolutionCache) -> None:
    """put() followed by get() with the same version returns the stored result."""
    result = _make_result()
    cache.put("Escherichia coli", result, registry_version=1)

    retrieved = cache.get("Escherichia coli", current_registry_version=1)
    assert retrieved is not None
    assert retrieved.canonical_id == result.canonical_id
    assert retrieved.winning_strategy == result.winning_strategy
    assert retrieved.grounding_confidence == result.grounding_confidence


def test_get_is_case_insensitive_via_normalisation(cache: ResolutionCache) -> None:
    """
    Cache key is the normalised surface form, so lookups are case-insensitive
    (both 'E. coli' and 'e. coli' map to the same normalised key).
    """
    result = _make_result(surface_form="E. coli")
    cache.put("E. coli", result, registry_version=1)

    # Lookup with different casing — normalisation should match
    retrieved = cache.get("e. coli", current_registry_version=1)
    assert retrieved is not None
    assert retrieved.canonical_id == result.canonical_id


def test_get_returns_none_on_version_mismatch(cache: ResolutionCache) -> None:
    """
    get() returns None when the cached entry's registry_version does not match
    the current_registry_version.

    Requirements: 8.3
    """
    result = _make_result()
    cache.put("Escherichia coli", result, registry_version=1)

    # Request with a different (newer) version — should be a miss
    retrieved = cache.get("Escherichia coli", current_registry_version=2)
    assert retrieved is None


def test_put_overwrites_existing_entry(cache: ResolutionCache) -> None:
    """put() with the same surface form overwrites the previous entry."""
    result_v1 = _make_result(canonical_id="562")
    result_v2 = _make_result(canonical_id="1301")

    cache.put("Escherichia coli", result_v1, registry_version=1)
    cache.put("Escherichia coli", result_v2, registry_version=1)

    retrieved = cache.get("Escherichia coli", current_registry_version=1)
    assert retrieved is not None
    assert retrieved.canonical_id == "1301"


# ---------------------------------------------------------------------------
# Unit tests — LRU eviction
# ---------------------------------------------------------------------------


def test_lru_eviction_when_capacity_exceeded() -> None:
    """
    When capacity is exceeded, the least-recently-used entry is evicted from
    the in-memory tier.

    Requirements: 8.2
    """
    cache = _fresh_cache(capacity=3)

    # Fill to capacity
    for i in range(3):
        cache.put(f"form_{i}", _make_result(surface_form=f"form_{i}"), registry_version=1)

    # Access form_0 to make it most-recently-used
    cache.get("form_0", current_registry_version=1)

    # Add a 4th entry — should evict form_1 (LRU)
    cache.put("form_3", _make_result(surface_form="form_3"), registry_version=1)

    # form_0 and form_2 and form_3 should still be in memory
    assert cache.get("form_0", current_registry_version=1) is not None
    assert cache.get("form_2", current_registry_version=1) is not None
    assert cache.get("form_3", current_registry_version=1) is not None

    # form_1 was evicted from memory, but it should still be in SQLite
    # (SQLite is not subject to LRU eviction)
    assert cache.get("form_1", current_registry_version=1) is not None


def test_lru_eviction_memory_only_miss_after_eviction() -> None:
    """
    After LRU eviction from memory, the entry is still retrievable from SQLite.
    """
    cache = _fresh_cache(capacity=2)

    cache.put("alpha", _make_result(surface_form="alpha", canonical_id="A"), registry_version=1)
    cache.put("beta", _make_result(surface_form="beta", canonical_id="B"), registry_version=1)

    # Access beta to make it MRU; alpha becomes LRU
    cache.get("beta", current_registry_version=1)

    # Add gamma — evicts alpha from memory
    cache.put("gamma", _make_result(surface_form="gamma", canonical_id="C"), registry_version=1)

    # alpha should still be retrievable (from SQLite)
    result = cache.get("alpha", current_registry_version=1)
    assert result is not None
    assert result.canonical_id == "A"


def test_lru_capacity_one() -> None:
    """Edge case: capacity=1 always evicts the previous entry from memory."""
    cache = _fresh_cache(capacity=1)

    cache.put("first", _make_result(surface_form="first", canonical_id="1"), registry_version=1)
    cache.put("second", _make_result(surface_form="second", canonical_id="2"), registry_version=1)

    # Both should be retrievable (second from memory, first from SQLite)
    assert cache.get("second", current_registry_version=1) is not None
    assert cache.get("first", current_registry_version=1) is not None


# ---------------------------------------------------------------------------
# Unit tests — invalidate_version
# ---------------------------------------------------------------------------


def test_invalidate_version_removes_entries_from_both_tiers() -> None:
    """
    invalidate_version() removes all entries with the given version from both
    the in-memory and SQLite tiers.

    Requirements: 8.5
    """
    cache = _fresh_cache(capacity=100)

    # Put entries under version 1
    for i in range(5):
        cache.put(f"form_{i}", _make_result(surface_form=f"form_{i}"), registry_version=1)

    # Put entries under version 2
    for i in range(5, 8):
        cache.put(f"form_{i}", _make_result(surface_form=f"form_{i}"), registry_version=2)

    # Invalidate version 1
    count = cache.invalidate_version(1)
    assert count == 5

    # Version-1 entries should be gone
    for i in range(5):
        assert cache.get(f"form_{i}", current_registry_version=1) is None
        assert cache.get(f"form_{i}", current_registry_version=2) is None

    # Version-2 entries should still be present
    for i in range(5, 8):
        assert cache.get(f"form_{i}", current_registry_version=2) is not None


def test_invalidate_version_returns_zero_when_no_entries() -> None:
    """invalidate_version() returns 0 when no entries match the given version."""
    cache = _fresh_cache()
    count = cache.invalidate_version(99)
    assert count == 0


def test_invalidate_version_does_not_affect_other_versions() -> None:
    """invalidate_version(V) does not remove entries with version != V."""
    cache = _fresh_cache(capacity=100)

    cache.put("form_a", _make_result(surface_form="form_a"), registry_version=1)
    cache.put("form_b", _make_result(surface_form="form_b"), registry_version=2)
    cache.put("form_c", _make_result(surface_form="form_c"), registry_version=3)

    cache.invalidate_version(2)

    assert cache.get("form_a", current_registry_version=1) is not None
    assert cache.get("form_b", current_registry_version=2) is None
    assert cache.get("form_c", current_registry_version=3) is not None


def test_invalidate_version_returns_correct_count_with_memory_and_sqlite() -> None:
    """
    invalidate_version() counts entries correctly even when some are only in
    SQLite (evicted from memory by LRU).
    """
    cache = _fresh_cache(capacity=2)

    # Put 3 entries under version 1 — the first will be evicted from memory
    cache.put("form_0", _make_result(surface_form="form_0"), registry_version=1)
    cache.put("form_1", _make_result(surface_form="form_1"), registry_version=1)
    cache.put("form_2", _make_result(surface_form="form_2"), registry_version=1)

    # form_0 is evicted from memory but still in SQLite
    count = cache.invalidate_version(1)
    assert count == 3  # all 3 should be counted (from SQLite)


# ---------------------------------------------------------------------------
# Unit tests — SQLite persistence
# ---------------------------------------------------------------------------


def test_sqlite_persistence_after_clearing_memory_cache() -> None:
    """
    After clearing the in-memory cache, entries are still retrievable from
    the SQLite tier.

    Requirements: 8.2, 8.3
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_resolution_cache_schema())

    cache = ResolutionCache(capacity=100, conn=conn)
    result = _make_result(canonical_id="562")
    cache.put("Escherichia coli", result, registry_version=1)

    # Simulate clearing the in-memory tier
    cache._lru.clear()

    # Should still be retrievable from SQLite
    retrieved = cache.get("Escherichia coli", current_registry_version=1)
    assert retrieved is not None
    assert retrieved.canonical_id == "562"


def test_sqlite_persistence_version_mismatch_after_clearing_memory() -> None:
    """
    After clearing the in-memory cache, a version mismatch in SQLite still
    returns None.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_resolution_cache_schema())

    cache = ResolutionCache(capacity=100, conn=conn)
    result = _make_result()
    cache.put("Escherichia coli", result, registry_version=1)

    # Clear memory tier
    cache._lru.clear()

    # Request with newer version — should be a miss
    retrieved = cache.get("Escherichia coli", current_registry_version=2)
    assert retrieved is None


# ---------------------------------------------------------------------------
# Property 11: Cache Version Invalidation
# **Validates: Requirements 2.5, 8.5**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    surface_form=_st_surface_form(),
    version_v=_st_registry_version(),
    result_v=_st_resolution_result(),
    result_v1=_st_resolution_result(),
)
def test_property_cache_version_invalidation(
    surface_form: str,
    version_v: int,
    result_v: ResolutionResult,
    result_v1: ResolutionResult,
) -> None:
    """
    **Property 11: Cache Version Invalidation**

    **Validates: Requirements 2.5, 8.5**

    Steps:
    1. Cache a result under version V.
    2. Advance registry to V+1 (simulate by calling invalidate_version(V)).
    3. Assert that get() with version V+1 returns None (cache miss — forces
       re-execution).
    4. Assert that after put() with version V+1, get() with version V+1
       returns the result.
    """
    version_v1 = version_v + 1
    cache = _fresh_cache(capacity=1000)

    # Step 1: Cache a result under version V
    cache.put(surface_form, result_v, registry_version=version_v)

    # Verify it is retrievable under version V
    retrieved_v = cache.get(surface_form, current_registry_version=version_v)
    assert retrieved_v is not None, (
        f"Expected cached result under version {version_v} to be retrievable"
    )

    # Step 2: Advance registry to V+1 (invalidate old version)
    invalidated = cache.invalidate_version(version_v)
    assert invalidated >= 1, (
        f"Expected at least 1 entry to be invalidated, got {invalidated}"
    )

    # Step 3: get() with version V+1 must return None (cache miss)
    retrieved_after_invalidation = cache.get(
        surface_form, current_registry_version=version_v1
    )
    assert retrieved_after_invalidation is None, (
        f"Expected None after invalidating version {version_v}, "
        f"but got a result with version V+1={version_v1}"
    )

    # Step 4: put() with version V+1, then get() with version V+1 returns result
    cache.put(surface_form, result_v1, registry_version=version_v1)
    retrieved_v1 = cache.get(surface_form, current_registry_version=version_v1)
    assert retrieved_v1 is not None, (
        f"Expected cached result under version {version_v1} to be retrievable "
        f"after put()"
    )
    assert retrieved_v1.canonical_id == result_v1.canonical_id, (
        f"canonical_id mismatch: expected {result_v1.canonical_id!r}, "
        f"got {retrieved_v1.canonical_id!r}"
    )
    assert retrieved_v1.winning_strategy == result_v1.winning_strategy, (
        f"winning_strategy mismatch: expected {result_v1.winning_strategy!r}, "
        f"got {retrieved_v1.winning_strategy!r}"
    )
    assert retrieved_v1.grounding_confidence == result_v1.grounding_confidence, (
        f"grounding_confidence mismatch: expected {result_v1.grounding_confidence}, "
        f"got {retrieved_v1.grounding_confidence}"
    )

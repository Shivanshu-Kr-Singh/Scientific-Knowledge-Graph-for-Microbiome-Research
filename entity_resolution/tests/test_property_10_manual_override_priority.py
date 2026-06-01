"""
Property 10: Manual Override Priority

**Validates: Requirements 9.1, 9.2, 9.4**

For any surface form with a ``ManualOverride`` set, assert ``resolve()``
returns the override's ``canonical_id`` with ``grounding_confidence=1.0``
and ``winning_strategy="manual_override"`` regardless of automated strategy
results.

The test registers the same surface form in the ``CanonicalRegistry`` with a
*different* ``canonical_id`` so that automated strategies (exact match,
normalized match, synonym lookup) would return a different result if the
manual override were not applied first.

Requirements: 9.1, 9.2, 9.4
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.canonical_registry import CanonicalRegistry
from entity_resolution.manual_override_manager import ManualOverrideManager
from entity_resolution.models import (
    CanonicalEntityRecord,
    EntityType,
    SynonymProvenance,
    SynonymRecord,
)
from entity_resolution.resolution_pipeline import ResolutionPipeline

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-empty printable text for surface forms (avoid empty/whitespace-only)
_surface_form_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
        whitelist_characters="-_.",
    ),
    min_size=3,
    max_size=40,
).map(str.strip).filter(lambda s: len(s) >= 2)

# Valid taxon canonical IDs: positive integer strings
_taxon_id_st = st.integers(min_value=1, max_value=999_999).map(str)

# Generate two *distinct* taxon IDs for override vs. registry
_two_distinct_taxon_ids_st = st.tuples(_taxon_id_st, _taxon_id_st).filter(
    lambda pair: pair[0] != pair[1]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_in_memory_registry_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the canonical_registry schema."""
    from entity_resolution.conftest import _CANONICAL_REGISTRY_DDL, _apply_ddl

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _apply_ddl(conn, _CANONICAL_REGISTRY_DDL)
    return conn


def _make_pipeline(
    registry_conn: sqlite3.Connection,
) -> tuple[ResolutionPipeline, CanonicalRegistry, ManualOverrideManager]:
    """
    Build a ResolutionPipeline wired to in-memory components.

    Returns (pipeline, registry, override_manager).
    """
    registry = CanonicalRegistry(conn=registry_conn)
    override_manager = ManualOverrideManager(conn=registry_conn)
    pipeline = ResolutionPipeline(
        registry=registry,
        override_manager=override_manager,
        # No audit store, cache, or optional strategies needed for this property
    )
    return pipeline, registry, override_manager


def _register_entity(
    registry: CanonicalRegistry,
    canonical_id: str,
    surface_form: str,
) -> None:
    """Register a taxon entity with the given surface form in the registry."""
    now = datetime.now(timezone.utc)
    record = CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=surface_form,
        entity_type=EntityType.TAXON,
        ontology_source="ncbi_taxonomy",
        synonyms=[],
        created_at=now,
        updated_at=now,
    )
    success, error = registry.register(record)
    assert success, f"Failed to register entity {canonical_id!r}: {error}"


# ---------------------------------------------------------------------------
# Property 10: Manual Override Priority
# **Validates: Requirements 9.1, 9.2, 9.4**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    surface_form=_surface_form_st,
    ids=_two_distinct_taxon_ids_st,
)
def test_property_manual_override_priority(
    surface_form: str,
    ids: tuple[str, str],
) -> None:
    """
    **Property 10: Manual Override Priority**

    **Validates: Requirements 9.1, 9.2, 9.4**

    For any surface form with a ``ManualOverride`` set:
    1. ``resolve()`` returns the override's ``canonical_id``
    2. ``grounding_confidence == 1.0``
    3. ``winning_strategy == "manual_override"``

    This holds regardless of automated strategy results: the same surface form
    is registered in the ``CanonicalRegistry`` with a *different* canonical_id
    so that exact-match would return a different result if the override were
    not applied first.
    """
    override_id, registry_id = ids  # override_id != registry_id (guaranteed by filter)

    # Use a fresh in-memory database for each example to avoid cross-contamination
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, override_manager = _make_pipeline(conn)

        # Register the surface form in the registry with registry_id
        # (automated strategies would return registry_id if override is ignored)
        _register_entity(registry, registry_id, surface_form)

        # Set a manual override pointing to override_id (different from registry_id)
        success, error = override_manager.set_override(
            surface_form=surface_form,
            canonical_id=override_id,
            entity_type="taxon",
            curator_id="test_curator",
            justification="Property 10 test override",
        )
        assert success, (
            f"set_override() failed for surface_form={surface_form!r}, "
            f"canonical_id={override_id!r}: {error}"
        )

        # Resolve the surface form
        result = pipeline.resolve(
            surface_form=surface_form,
            entity_type="taxon",
            paper_id="test_paper_001",
        )

        # Assertion 1: canonical_id must be the override's canonical_id
        assert result.canonical_id == override_id, (
            f"Expected canonical_id={override_id!r} (from manual override), "
            f"got {result.canonical_id!r} (registry_id={registry_id!r}). "
            f"surface_form={surface_form!r}"
        )

        # Assertion 2: grounding_confidence must be 1.0 (Requirement 9.4)
        assert result.grounding_confidence == 1.0, (
            f"Expected grounding_confidence=1.0 for manual override, "
            f"got {result.grounding_confidence}. "
            f"surface_form={surface_form!r}, canonical_id={override_id!r}"
        )

        # Assertion 3: winning_strategy must be "manual_override"
        assert result.winning_strategy == "manual_override", (
            f"Expected winning_strategy='manual_override', "
            f"got {result.winning_strategy!r}. "
            f"surface_form={surface_form!r}, canonical_id={override_id!r}"
        )

        # Bonus: result must be grounded
        assert result.grounded is True, (
            f"Expected grounded=True for manual override result, "
            f"got grounded={result.grounded}. "
            f"surface_form={surface_form!r}"
        )

    finally:
        conn.close()


@settings(max_examples=100)
@given(
    surface_form=_surface_form_st,
    ids=_two_distinct_taxon_ids_st,
)
def test_property_manual_override_bypasses_automated_strategies(
    surface_form: str,
    ids: tuple[str, str],
) -> None:
    """
    **Property 10: Manual Override Priority — automated strategies bypassed**

    **Validates: Requirements 9.1, 9.2**

    When a manual override exists, the pipeline must NOT return the canonical_id
    that automated strategies (exact match) would have returned.

    This verifies that the override is checked *before* any automated strategy
    and that the automated result is discarded.
    """
    override_id, registry_id = ids  # override_id != registry_id

    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, override_manager = _make_pipeline(conn)

        # Register the surface form in the registry with registry_id
        _register_entity(registry, registry_id, surface_form)

        # Set a manual override pointing to override_id
        success, _ = override_manager.set_override(
            surface_form=surface_form,
            canonical_id=override_id,
            entity_type="taxon",
            curator_id="test_curator",
        )
        assert success

        result = pipeline.resolve(
            surface_form=surface_form,
            entity_type="taxon",
            paper_id="test_paper_002",
        )

        # The automated strategy (exact match) would return registry_id,
        # but the override must take precedence and return override_id.
        assert result.canonical_id != registry_id, (
            f"Manual override was ignored: resolve() returned registry_id={registry_id!r} "
            f"instead of override_id={override_id!r}. "
            f"surface_form={surface_form!r}"
        )
        assert result.canonical_id == override_id, (
            f"Expected override_id={override_id!r}, got {result.canonical_id!r}. "
            f"surface_form={surface_form!r}"
        )

    finally:
        conn.close()


@settings(max_examples=100)
@given(
    surface_form=_surface_form_st,
    override_id=_taxon_id_st,
)
def test_property_manual_override_without_registry_entry(
    surface_form: str,
    override_id: str,
) -> None:
    """
    **Property 10: Manual Override Priority — no registry entry needed**

    **Validates: Requirements 9.1, 9.2, 9.4**

    A manual override must work even when the surface form is NOT registered
    in the CanonicalRegistry (i.e., automated strategies would all fail).

    The override must still return canonical_id with grounding_confidence=1.0
    and winning_strategy="manual_override".
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, override_manager = _make_pipeline(conn)

        # Do NOT register the surface form in the registry
        # (automated strategies will all fail)

        # Set a manual override
        success, error = override_manager.set_override(
            surface_form=surface_form,
            canonical_id=override_id,
            entity_type="taxon",
            curator_id="test_curator",
        )
        assert success, (
            f"set_override() failed: {error}. "
            f"surface_form={surface_form!r}, canonical_id={override_id!r}"
        )

        result = pipeline.resolve(
            surface_form=surface_form,
            entity_type="taxon",
            paper_id="test_paper_003",
        )

        assert result.canonical_id == override_id, (
            f"Expected canonical_id={override_id!r}, got {result.canonical_id!r}. "
            f"surface_form={surface_form!r}"
        )
        assert result.grounding_confidence == 1.0, (
            f"Expected grounding_confidence=1.0, got {result.grounding_confidence}. "
            f"surface_form={surface_form!r}"
        )
        assert result.winning_strategy == "manual_override", (
            f"Expected winning_strategy='manual_override', got {result.winning_strategy!r}. "
            f"surface_form={surface_form!r}"
        )
        assert result.grounded is True

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests — explicit examples
# ---------------------------------------------------------------------------


def test_manual_override_takes_priority_over_exact_match() -> None:
    """
    Explicit example: manual override beats exact match.

    Registry has "Escherichia coli" -> "562".
    Override sets "Escherichia coli" -> "99999".
    resolve() must return "99999" with confidence=1.0 and strategy="manual_override".
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, override_manager = _make_pipeline(conn)

        # Register in registry with canonical_id "562"
        _register_entity(registry, "562", "Escherichia coli")

        # Set override to "99999"
        success, error = override_manager.set_override(
            surface_form="Escherichia coli",
            canonical_id="99999",
            entity_type="taxon",
            curator_id="curator_001",
            justification="Test override",
        )
        assert success, f"set_override() failed: {error}"

        result = pipeline.resolve("Escherichia coli", "taxon", "paper_001")

        assert result.canonical_id == "99999"
        assert result.grounding_confidence == 1.0
        assert result.winning_strategy == "manual_override"
        assert result.grounded is True

    finally:
        conn.close()


def test_no_override_falls_through_to_exact_match() -> None:
    """
    Without a manual override, the pipeline falls through to exact match.

    Registry has "Bacteroides fragilis" -> "817".
    No override set.
    resolve() must return "817" via exact match.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, override_manager = _make_pipeline(conn)

        _register_entity(registry, "817", "Bacteroides fragilis")

        result = pipeline.resolve("Bacteroides fragilis", "taxon", "paper_002")

        assert result.canonical_id == "817"
        assert result.winning_strategy == "exact"
        assert result.grounded is True

    finally:
        conn.close()


def test_override_removed_falls_through_to_automated() -> None:
    """
    After removing a manual override, the pipeline uses automated strategies.

    Registry has "Lactobacillus acidophilus" -> "1579".
    Override sets it to "88888", then override is removed.
    After removal, resolve() must return "1579" via exact match.

    Requirements: 9.5
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, override_manager = _make_pipeline(conn)

        _register_entity(registry, "1579", "Lactobacillus acidophilus")

        # Set override
        success, _ = override_manager.set_override(
            surface_form="Lactobacillus acidophilus",
            canonical_id="88888",
            entity_type="taxon",
            curator_id="curator_002",
        )
        assert success

        # Verify override is active
        result_with_override = pipeline.resolve(
            "Lactobacillus acidophilus", "taxon", "paper_003"
        )
        assert result_with_override.canonical_id == "88888"
        assert result_with_override.winning_strategy == "manual_override"

        # Remove override
        removed = override_manager.remove_override("Lactobacillus acidophilus")
        assert removed is True

        # Now resolve should fall through to exact match
        result_without_override = pipeline.resolve(
            "Lactobacillus acidophilus", "taxon", "paper_004"
        )
        assert result_without_override.canonical_id == "1579"
        assert result_without_override.winning_strategy == "exact"

    finally:
        conn.close()


def test_override_confidence_is_always_1_0() -> None:
    """
    Manual override grounding_confidence is always 1.0 regardless of the
    canonical_id or surface form.

    Requirements: 9.4
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, override_manager = _make_pipeline(conn)

        override_manager.set_override(
            surface_form="some entity",
            canonical_id="42",
            entity_type="taxon",
            curator_id="curator_003",
        )

        result = pipeline.resolve("some entity", "taxon", "paper_005")

        assert result.grounding_confidence == 1.0, (
            f"Manual override confidence must always be 1.0, got {result.grounding_confidence}"
        )

    finally:
        conn.close()

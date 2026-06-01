"""
Property 1: Determinism and Idempotency

**Validates: Requirements 2.1, 2.4, 15.1**

For any surface form, assert two sequential ``resolve()`` calls return
identical ``canonical_id``, ``winning_strategy``, ``grounding_confidence``,
and ``conflict_set``; also assert ``resolve(canonical_id)`` returns
``canonical_id`` and ``grounded=True``.

Requirements: 2.1, 2.4, 15.1
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.canonical_registry import CanonicalRegistry
from entity_resolution.models import (
    CandidateScore,
    CanonicalEntityRecord,
    EntityType,
    ResolutionResult,
    SynonymProvenance,
    SynonymRecord,
)
from entity_resolution.resolution_pipeline import ResolutionPipeline
from entity_resolution.synonym_index import SynonymIndex

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

# Entity types supported by the pipeline
_entity_type_st = st.sampled_from(["taxon", "disease", "method"])


# ---------------------------------------------------------------------------
# Helpers: canonical ID generation per entity type
# ---------------------------------------------------------------------------


def _make_canonical_id(entity_type: str, seed: int) -> str:
    """Return a valid canonical ID for the given entity type."""
    if entity_type == "taxon":
        return str(abs(seed) % 999_999 + 1)
    elif entity_type == "disease":
        # Pattern: one uppercase letter + one or more digits, e.g. "D006262"
        letter = chr(ord("A") + (abs(seed) % 26))
        digits = str(abs(seed) % 99999 + 1)
        return f"{letter}{digits}"
    else:  # method
        alphanum = str(abs(seed) % 99999 + 1)
        return f"METHOD-{alphanum}"


# ---------------------------------------------------------------------------
# Helpers: in-memory pipeline setup
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
) -> tuple[ResolutionPipeline, CanonicalRegistry, SynonymIndex]:
    """
    Build a ResolutionPipeline wired to in-memory components.

    Returns (pipeline, registry, synonym_index).
    """
    registry = CanonicalRegistry(conn=registry_conn)
    synonym_index = SynonymIndex(conn=registry_conn)
    registry.set_synonym_index(synonym_index)

    pipeline = ResolutionPipeline(
        registry=registry,
        synonym_index=synonym_index,
        # No audit store, cache, or optional strategies needed for this property
    )
    return pipeline, registry, synonym_index


def _register_entity(
    registry: CanonicalRegistry,
    canonical_id: str,
    primary_name: str,
    entity_type: str,
    synonym_surface_form: str,
) -> None:
    """Register a canonical entity with the given surface form as a synonym."""
    now = datetime.now(timezone.utc)
    record = CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=primary_name,
        entity_type=EntityType(entity_type),
        ontology_source="ncbi_taxonomy" if entity_type == "taxon" else (
            "mesh" if entity_type == "disease" else "internal"
        ),
        synonyms=[
            SynonymRecord(
                surface_form=synonym_surface_form,
                provenance=SynonymProvenance.CURATOR,
                added_by="test",
                added_at=now,
            )
        ] if synonym_surface_form != primary_name else [],
        created_at=now,
        updated_at=now,
    )
    success, error = registry.register(record)
    assert success, f"Failed to register entity {canonical_id!r}: {error}"


# ---------------------------------------------------------------------------
# Conflict set normalisation helper
# ---------------------------------------------------------------------------


def _normalize_conflict_set(conflict_set: List[CandidateScore]) -> List[dict]:
    """
    Normalise a conflict_set for comparison, excluding timestamps.

    Returns a sorted list of dicts with canonical_id, strategy, and
    grounding_confidence (rounded to 9 decimal places to avoid float noise).
    """
    normalized = [
        {
            "canonical_id": c.canonical_id,
            "strategy": c.strategy,
            "grounding_confidence": round(c.grounding_confidence, 9),
        }
        for c in conflict_set
    ]
    # Sort deterministically by (canonical_id, strategy) for comparison
    return sorted(normalized, key=lambda d: (d["canonical_id"], d["strategy"]))


# ---------------------------------------------------------------------------
# Property 1: Determinism and Idempotency
# **Validates: Requirements 2.1, 2.4, 15.1**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    surface_form=_surface_form_st,
    entity_type=_entity_type_st,
)
def test_property_determinism_two_sequential_calls(
    surface_form: str,
    entity_type: str,
) -> None:
    """
    **Property 1: Determinism and Idempotency**

    **Validates: Requirements 2.1, 2.4, 15.1**

    For any surface form, two sequential ``resolve()`` calls must return
    identical ``canonical_id``, ``winning_strategy``, ``grounding_confidence``,
    and ``conflict_set``.

    A canonical entity is registered with the generated surface form as a
    synonym so that the pipeline has something to resolve.
    """
    # Derive a stable canonical_id from the surface form to avoid collisions
    seed = hash(surface_form) & 0x7FFFFFFF
    canonical_id = _make_canonical_id(entity_type, seed)

    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        # Register a canonical entity with the surface form as a synonym
        _register_entity(
            registry=registry,
            canonical_id=canonical_id,
            primary_name=f"Primary {canonical_id}",
            entity_type=entity_type,
            synonym_surface_form=surface_form,
        )

        paper_id = "test_paper_determinism"

        # First resolve call
        result1 = pipeline.resolve(surface_form, entity_type, paper_id)

        # Second resolve call — must be identical
        result2 = pipeline.resolve(surface_form, entity_type, paper_id)

        # Assertion 1: canonical_id must be identical
        assert result1.canonical_id == result2.canonical_id, (
            f"Determinism violation: canonical_id differs between calls.\n"
            f"  Call 1: {result1.canonical_id!r}\n"
            f"  Call 2: {result2.canonical_id!r}\n"
            f"  surface_form={surface_form!r}, entity_type={entity_type!r}"
        )

        # Assertion 2: winning_strategy must be identical
        assert result1.winning_strategy == result2.winning_strategy, (
            f"Determinism violation: winning_strategy differs between calls.\n"
            f"  Call 1: {result1.winning_strategy!r}\n"
            f"  Call 2: {result2.winning_strategy!r}\n"
            f"  surface_form={surface_form!r}, entity_type={entity_type!r}"
        )

        # Assertion 3: grounding_confidence must be identical
        assert result1.grounding_confidence == result2.grounding_confidence, (
            f"Determinism violation: grounding_confidence differs between calls.\n"
            f"  Call 1: {result1.grounding_confidence}\n"
            f"  Call 2: {result2.grounding_confidence}\n"
            f"  surface_form={surface_form!r}, entity_type={entity_type!r}"
        )

        # Assertion 4: conflict_set must be identical (normalised, no timestamps)
        cs1 = _normalize_conflict_set(result1.conflict_set)
        cs2 = _normalize_conflict_set(result2.conflict_set)
        assert cs1 == cs2, (
            f"Determinism violation: conflict_set differs between calls.\n"
            f"  Call 1: {cs1}\n"
            f"  Call 2: {cs2}\n"
            f"  surface_form={surface_form!r}, entity_type={entity_type!r}"
        )

    finally:
        conn.close()


@settings(max_examples=100)
@given(
    surface_form=_surface_form_st,
    entity_type=_entity_type_st,
)
def test_property_idempotency_canonical_id_resolves_to_itself(
    surface_form: str,
    entity_type: str,
) -> None:
    """
    **Property 1: Determinism and Idempotency — canonical ID self-resolution**

    **Validates: Requirements 2.1, 2.4, 15.1**

    After resolving a surface form to a canonical_id, resolving the
    canonical_id itself must return the same canonical_id with grounded=True.

    This verifies idempotency: ``resolve(canonical_id)`` == ``canonical_id``.
    """
    seed = hash(surface_form) & 0x7FFFFFFF
    canonical_id = _make_canonical_id(entity_type, seed)

    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        # Register the canonical entity; the primary_name is also registered
        # as a synonym, so resolving the canonical_id directly will hit exact match.
        _register_entity(
            registry=registry,
            canonical_id=canonical_id,
            primary_name=canonical_id,  # primary_name == canonical_id for idempotency
            entity_type=entity_type,
            synonym_surface_form=surface_form,
        )

        paper_id = "test_paper_idempotency"

        # First: resolve the surface form to get the canonical_id
        result_surface = pipeline.resolve(surface_form, entity_type, paper_id)

        # The surface form should resolve to our registered canonical_id
        # (either via exact match, normalized match, or synonym lookup)
        assert result_surface.canonical_id == canonical_id, (
            f"Surface form did not resolve to expected canonical_id.\n"
            f"  Expected: {canonical_id!r}\n"
            f"  Got: {result_surface.canonical_id!r}\n"
            f"  surface_form={surface_form!r}, entity_type={entity_type!r}"
        )

        # Second: resolve the canonical_id itself — must return canonical_id and grounded=True
        result_canonical = pipeline.resolve(canonical_id, entity_type, paper_id)

        assert result_canonical.canonical_id == canonical_id, (
            f"Idempotency violation: resolve(canonical_id) returned different canonical_id.\n"
            f"  Input canonical_id: {canonical_id!r}\n"
            f"  Returned canonical_id: {result_canonical.canonical_id!r}\n"
            f"  entity_type={entity_type!r}"
        )

        assert result_canonical.grounded is True, (
            f"Idempotency violation: resolve(canonical_id) returned grounded=False.\n"
            f"  canonical_id={canonical_id!r}, entity_type={entity_type!r}"
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests — explicit examples
# ---------------------------------------------------------------------------


def test_determinism_taxon_explicit() -> None:
    """
    Explicit example: two sequential resolve() calls for a taxon return
    identical results.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(
            registry=registry,
            canonical_id="562",
            primary_name="Escherichia coli",
            entity_type="taxon",
            synonym_surface_form="E. coli",
        )

        result1 = pipeline.resolve("E. coli", "taxon", "paper_001")
        result2 = pipeline.resolve("E. coli", "taxon", "paper_001")

        assert result1.canonical_id == result2.canonical_id == "562"
        assert result1.winning_strategy == result2.winning_strategy
        assert result1.grounding_confidence == result2.grounding_confidence
        assert _normalize_conflict_set(result1.conflict_set) == _normalize_conflict_set(result2.conflict_set)

    finally:
        conn.close()


def test_idempotency_taxon_canonical_id_resolves_to_itself() -> None:
    """
    Explicit example: resolve("562", "taxon") returns canonical_id="562"
    and grounded=True.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        # Register with primary_name == canonical_id so it resolves via exact match
        _register_entity(
            registry=registry,
            canonical_id="562",
            primary_name="562",
            entity_type="taxon",
            synonym_surface_form="Escherichia coli",
        )

        result = pipeline.resolve("562", "taxon", "paper_002")

        assert result.canonical_id == "562"
        assert result.grounded is True

    finally:
        conn.close()


def test_determinism_disease_explicit() -> None:
    """
    Explicit example: two sequential resolve() calls for a disease return
    identical results.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(
            registry=registry,
            canonical_id="D006262",
            primary_name="Health",
            entity_type="disease",
            synonym_surface_form="good health",
        )

        result1 = pipeline.resolve("good health", "disease", "paper_003")
        result2 = pipeline.resolve("good health", "disease", "paper_003")

        assert result1.canonical_id == result2.canonical_id
        assert result1.winning_strategy == result2.winning_strategy
        assert result1.grounding_confidence == result2.grounding_confidence
        assert _normalize_conflict_set(result1.conflict_set) == _normalize_conflict_set(result2.conflict_set)

    finally:
        conn.close()


def test_determinism_method_explicit() -> None:
    """
    Explicit example: two sequential resolve() calls for a method return
    identical results.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(
            registry=registry,
            canonical_id="METHOD-16S",
            primary_name="16S rRNA sequencing",
            entity_type="method",
            synonym_surface_form="16S sequencing",
        )

        result1 = pipeline.resolve("16S sequencing", "method", "paper_004")
        result2 = pipeline.resolve("16S sequencing", "method", "paper_004")

        assert result1.canonical_id == result2.canonical_id
        assert result1.winning_strategy == result2.winning_strategy
        assert result1.grounding_confidence == result2.grounding_confidence
        assert _normalize_conflict_set(result1.conflict_set) == _normalize_conflict_set(result2.conflict_set)

    finally:
        conn.close()


def test_determinism_unresolved_form_is_also_deterministic() -> None:
    """
    Even when a surface form cannot be resolved, two calls must return
    identical results (both unresolved, same winning_strategy="none").
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        # Do NOT register anything — surface form will be unresolved
        result1 = pipeline.resolve("completely unknown entity xyz", "taxon", "paper_005")
        result2 = pipeline.resolve("completely unknown entity xyz", "taxon", "paper_005")

        assert result1.canonical_id == result2.canonical_id  # both None
        assert result1.winning_strategy == result2.winning_strategy  # both "none"
        assert result1.grounding_confidence == result2.grounding_confidence
        assert result1.grounded == result2.grounded  # both False

    finally:
        conn.close()

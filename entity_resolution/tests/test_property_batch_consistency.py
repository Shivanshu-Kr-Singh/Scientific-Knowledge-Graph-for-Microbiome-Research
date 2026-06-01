"""
Property 6: Batch Consistency

**Validates: Requirements 8.1, 15.6**

For any list of 1–1000 surface forms, assert ``batch_resolve([F₁…Fₙ])``
equals ``[resolve(Fᵢ) for Fᵢ in forms]`` element-wise on all fields:
``canonical_id``, ``winning_strategy``, ``grounding_confidence``, ``grounded``.

Timestamps are excluded from the comparison because they are generated at
call time and may differ between the batch call and the individual calls.

Requirements: 8.1, 15.6
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.canonical_registry import CanonicalRegistry
from entity_resolution.conftest import _CANONICAL_REGISTRY_DDL, _apply_ddl
from entity_resolution.models import (
    CanonicalEntityRecord,
    EntityType,
    ResolutionResult,
)
from entity_resolution.resolution_pipeline import ResolutionPipeline
from entity_resolution.synonym_index import SynonymIndex

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Valid entity types
_ENTITY_TYPES = ["taxon", "disease", "method"]

_entity_type_st = st.sampled_from(_ENTITY_TYPES)

# Non-empty printable text for surface forms (avoid empty/whitespace-only)
_surface_form_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
        whitelist_characters="-_.",
    ),
    min_size=3,
    max_size=40,
).map(str.strip).filter(lambda s: len(s) >= 2)

# A single form tuple: (surface_form, entity_type, paper_id)
_form_tuple_st = st.tuples(
    _surface_form_st,
    _entity_type_st,
    st.text(min_size=1, max_size=20).map(str.strip).filter(lambda s: len(s) >= 1),
)

# A list of 1–50 form tuples (spec says 1–1000; 50 is sufficient for property testing)
_forms_list_st = st.lists(_form_tuple_st, min_size=1, max_size=50)

# Valid canonical IDs per entity type
_taxon_id_st = st.integers(min_value=1, max_value=999_999).map(str)
_disease_id_st = st.from_regex(r"[A-Z][0-9]{3,6}", fullmatch=True)
_method_id_st = st.from_regex(r"METHOD-[A-Za-z0-9]{1,10}", fullmatch=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_in_memory_registry_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the canonical_registry schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _apply_ddl(conn, _CANONICAL_REGISTRY_DDL)
    return conn


def _make_pipeline(
    registry_conn: sqlite3.Connection,
) -> Tuple[ResolutionPipeline, CanonicalRegistry, SynonymIndex]:
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
) -> bool:
    """Register a canonical entity; return True on success."""
    now = datetime.now(timezone.utc)
    record = CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=primary_name,
        entity_type=EntityType(entity_type),
        ontology_source="ncbi_taxonomy" if entity_type == "taxon" else "internal",
        synonyms=[],
        created_at=now,
        updated_at=now,
    )
    success, _ = registry.register(record)
    return success


def _compare_results(
    batch_result: ResolutionResult,
    individual_result: ResolutionResult,
    index: int,
) -> None:
    """
    Assert that two ResolutionResult objects are equal on the fields that
    must be consistent between batch_resolve() and individual resolve() calls.

    Timestamps are intentionally excluded because they are generated at call
    time and may differ between the two calls.
    """
    assert batch_result.canonical_id == individual_result.canonical_id, (
        f"[index={index}] canonical_id mismatch: "
        f"batch={batch_result.canonical_id!r} vs individual={individual_result.canonical_id!r}. "
        f"surface_form={batch_result.surface_form!r}"
    )
    assert batch_result.winning_strategy == individual_result.winning_strategy, (
        f"[index={index}] winning_strategy mismatch: "
        f"batch={batch_result.winning_strategy!r} vs individual={individual_result.winning_strategy!r}. "
        f"surface_form={batch_result.surface_form!r}"
    )
    assert batch_result.grounding_confidence == individual_result.grounding_confidence, (
        f"[index={index}] grounding_confidence mismatch: "
        f"batch={batch_result.grounding_confidence} vs individual={individual_result.grounding_confidence}. "
        f"surface_form={batch_result.surface_form!r}"
    )
    assert batch_result.grounded == individual_result.grounded, (
        f"[index={index}] grounded mismatch: "
        f"batch={batch_result.grounded} vs individual={individual_result.grounded}. "
        f"surface_form={batch_result.surface_form!r}"
    )


# ---------------------------------------------------------------------------
# Property 6: Batch Consistency
# **Validates: Requirements 8.1, 15.6**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(forms=_forms_list_st)
def test_property_batch_consistency(
    forms: List[Tuple[str, str, str]],
) -> None:
    """
    **Property 6: Batch Consistency**

    **Validates: Requirements 8.1, 15.6**

    For any list of 1–50 surface forms, assert that ``batch_resolve(forms)``
    returns the same results as ``[resolve(sf, et, pid) for sf, et, pid in forms]``
    element-wise on: ``canonical_id``, ``winning_strategy``,
    ``grounding_confidence``, and ``grounded``.

    The test registers a mix of canonical entities so that some forms resolve
    (grounded=True) and some do not (grounded=False), exercising both paths.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        # Register a small set of canonical entities so that some surface forms
        # in the generated list will resolve and some won't.
        # We register entities for the first few unique surface forms to create
        # a realistic mix of grounded and ungrounded results.
        registered_forms: set = set()
        for surface_form, entity_type, _paper_id in forms[:5]:  # register up to 5
            if surface_form in registered_forms:
                continue
            if entity_type == "taxon":
                # Use a hash-based ID to avoid collisions across examples
                canonical_id = str(abs(hash(surface_form)) % 900_000 + 1)
            elif entity_type == "disease":
                # Build a valid MeSH-style ID: one uppercase letter + digits
                h = abs(hash(surface_form)) % 999_999
                canonical_id = f"D{h:06d}"
            else:  # method
                h = abs(hash(surface_form)) % 99_999
                canonical_id = f"METHOD-M{h:05d}"

            success = _register_entity(
                registry, canonical_id, surface_form, entity_type
            )
            if success:
                registered_forms.add(surface_form)

        # --- Call batch_resolve() -----------------------------------------
        batch_results = pipeline.batch_resolve(forms)

        # --- Call resolve() individually for each form --------------------
        individual_results = [
            pipeline.resolve(surface_form, entity_type, paper_id)
            for surface_form, entity_type, paper_id in forms
        ]

        # --- Assert length matches ----------------------------------------
        assert len(batch_results) == len(forms), (
            f"batch_resolve() returned {len(batch_results)} results "
            f"but input had {len(forms)} forms"
        )
        assert len(individual_results) == len(forms)

        # --- Assert element-wise equality on key fields -------------------
        for i, (batch_res, indiv_res) in enumerate(
            zip(batch_results, individual_results)
        ):
            _compare_results(batch_res, indiv_res, i)

    finally:
        conn.close()


@settings(max_examples=100)
@given(forms=_forms_list_st)
def test_property_batch_preserves_input_order(
    forms: List[Tuple[str, str, str]],
) -> None:
    """
    **Property 6: Batch Consistency — input order preserved**

    **Validates: Requirements 8.1**

    ``batch_resolve()`` must return results in the same order as the input.
    The i-th result must correspond to the i-th input form.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        # Register a few entities to ensure some forms resolve
        for surface_form, entity_type, _paper_id in forms[:3]:
            if entity_type == "taxon":
                canonical_id = str(abs(hash(surface_form)) % 900_000 + 1)
            elif entity_type == "disease":
                h = abs(hash(surface_form)) % 999_999
                canonical_id = f"D{h:06d}"
            else:
                h = abs(hash(surface_form)) % 99_999
                canonical_id = f"METHOD-M{h:05d}"
            _register_entity(registry, canonical_id, surface_form, entity_type)

        batch_results = pipeline.batch_resolve(forms)

        assert len(batch_results) == len(forms), (
            f"batch_resolve() returned {len(batch_results)} results "
            f"but input had {len(forms)} forms"
        )

        # Verify each result's surface_form matches the corresponding input
        for i, ((surface_form, entity_type, paper_id), result) in enumerate(
            zip(forms, batch_results)
        ):
            assert result.surface_form == surface_form, (
                f"[index={i}] surface_form mismatch: "
                f"expected={surface_form!r}, got={result.surface_form!r}"
            )
            assert result.entity_type == entity_type, (
                f"[index={i}] entity_type mismatch: "
                f"expected={entity_type!r}, got={result.entity_type!r}"
            )
            assert result.paper_id == paper_id, (
                f"[index={i}] paper_id mismatch: "
                f"expected={paper_id!r}, got={result.paper_id!r}"
            )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests — explicit examples
# ---------------------------------------------------------------------------


def test_batch_resolve_single_grounded_form() -> None:
    """
    Explicit example: batch of one grounded form equals individual resolve().
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, _ = _make_pipeline(conn)
        _register_entity(registry, "562", "Escherichia coli", "taxon")

        forms = [("Escherichia coli", "taxon", "paper_001")]
        batch_results = pipeline.batch_resolve(forms)
        individual_result = pipeline.resolve("Escherichia coli", "taxon", "paper_001")

        assert len(batch_results) == 1
        _compare_results(batch_results[0], individual_result, 0)
        assert batch_results[0].grounded is True
        assert batch_results[0].canonical_id == "562"

    finally:
        conn.close()


def test_batch_resolve_single_ungrounded_form() -> None:
    """
    Explicit example: batch of one ungrounded form equals individual resolve().
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, _ = _make_pipeline(conn)
        # Do NOT register the surface form — it should be unresolved

        forms = [("unknown microbe xyz", "taxon", "paper_002")]
        batch_results = pipeline.batch_resolve(forms)
        individual_result = pipeline.resolve("unknown microbe xyz", "taxon", "paper_002")

        assert len(batch_results) == 1
        _compare_results(batch_results[0], individual_result, 0)
        assert batch_results[0].grounded is False
        assert batch_results[0].canonical_id is None

    finally:
        conn.close()


def test_batch_resolve_mixed_grounded_and_ungrounded() -> None:
    """
    Explicit example: batch with a mix of grounded and ungrounded forms.

    batch_resolve() must return the same results as individual resolve() calls
    for each form, in the same order.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, _ = _make_pipeline(conn)

        # Register two entities
        _register_entity(registry, "562", "Escherichia coli", "taxon")
        _register_entity(registry, "D006262", "Crohn disease", "disease")

        forms = [
            ("Escherichia coli", "taxon", "paper_001"),       # grounded
            ("unknown entity abc", "taxon", "paper_002"),     # ungrounded
            ("Crohn disease", "disease", "paper_003"),        # grounded
            ("mystery pathogen xyz", "disease", "paper_004"), # ungrounded
        ]

        batch_results = pipeline.batch_resolve(forms)
        individual_results = [
            pipeline.resolve(sf, et, pid) for sf, et, pid in forms
        ]

        assert len(batch_results) == 4

        for i, (batch_res, indiv_res) in enumerate(
            zip(batch_results, individual_results)
        ):
            _compare_results(batch_res, indiv_res, i)

        # Spot-check grounded flags
        assert batch_results[0].grounded is True
        assert batch_results[1].grounded is False
        assert batch_results[2].grounded is True
        assert batch_results[3].grounded is False

    finally:
        conn.close()


def test_batch_resolve_preserves_order_with_multiple_forms() -> None:
    """
    Explicit example: batch_resolve() preserves input order.

    Register 3 entities and submit them in reverse order; verify the results
    come back in the same (reversed) order.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, _ = _make_pipeline(conn)

        _register_entity(registry, "1", "Alpha bacterium", "taxon")
        _register_entity(registry, "2", "Beta bacterium", "taxon")
        _register_entity(registry, "3", "Gamma bacterium", "taxon")

        forms = [
            ("Gamma bacterium", "taxon", "paper_003"),
            ("Beta bacterium", "taxon", "paper_002"),
            ("Alpha bacterium", "taxon", "paper_001"),
        ]

        batch_results = pipeline.batch_resolve(forms)

        assert len(batch_results) == 3
        assert batch_results[0].canonical_id == "3"  # Gamma
        assert batch_results[1].canonical_id == "2"  # Beta
        assert batch_results[2].canonical_id == "1"  # Alpha

        # Verify surface_form order is preserved
        assert batch_results[0].surface_form == "Gamma bacterium"
        assert batch_results[1].surface_form == "Beta bacterium"
        assert batch_results[2].surface_form == "Alpha bacterium"

    finally:
        conn.close()


def test_batch_resolve_empty_list_returns_empty() -> None:
    """
    Edge case: batch_resolve([]) returns an empty list without error.

    Requirements: 8.1 (batch of 1–100,000; empty is outside spec but
    should not crash).
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, _ = _make_pipeline(conn)
        result = pipeline.batch_resolve([])
        assert result == []
    finally:
        conn.close()

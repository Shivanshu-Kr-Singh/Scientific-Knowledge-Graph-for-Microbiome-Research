"""
Property 3: No-Spurious-Merge

**Validates: Requirements 15.3, 6.5**

For two distinct canonical entities E₁ and E₂ with disjoint synonym sets S₁
and S₂, assert that no synonym of E₁ resolves to E₂'s ``canonical_id`` and
no synonym of E₂ resolves to E₁'s ``canonical_id``.

This property guards against the pipeline accidentally merging two separate
entities by routing a synonym of one entity to the other entity's canonical ID.

Requirements: 15.3, 6.5
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Set

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from entity_resolution.canonical_registry import CanonicalRegistry
from entity_resolution.models import (
    CanonicalEntityRecord,
    EntityType,
    SynonymProvenance,
    SynonymRecord,
)
from entity_resolution.resolution_pipeline import ResolutionPipeline
from entity_resolution.synonym_index import SynonymIndex
from entity_resolution.utils import normalize_surface_form

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-empty printable text for surface forms.
# Use a restricted alphabet to keep generated strings clean and avoid
# degenerate cases (pure whitespace, pure punctuation) that normalise to "".
_surface_form_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters=" -_.",
    ),
    min_size=3,
    max_size=40,
).map(str.strip).filter(lambda s: len(s) >= 2)

# Valid taxon canonical IDs: positive integer strings
_taxon_id_st = st.integers(min_value=1, max_value=999_999).map(str)

# Two distinct taxon IDs
_two_distinct_taxon_ids_st = st.tuples(_taxon_id_st, _taxon_id_st).filter(
    lambda pair: pair[0] != pair[1]
)

# A small non-empty list of surface forms (1–5 synonyms per entity)
_synonym_list_st = st.lists(_surface_form_st, min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Helpers
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
    # Wire the synonym index into the registry so both stay in sync
    registry.set_synonym_index(synonym_index)

    pipeline = ResolutionPipeline(
        registry=registry,
        synonym_index=synonym_index,
        # No audit store, cache, or optional strategies needed for this property
    )
    return pipeline, registry, synonym_index


def _register_entity_with_synonyms(
    registry: CanonicalRegistry,
    canonical_id: str,
    primary_name: str,
    extra_synonyms: List[str],
) -> None:
    """Register a taxon entity with the given primary name and extra synonyms."""
    now = datetime.now(timezone.utc)
    synonym_records = [
        SynonymRecord(
            surface_form=s,
            provenance=SynonymProvenance.CURATOR,
            added_by="test",
            added_at=now,
        )
        for s in extra_synonyms
    ]
    record = CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=primary_name,
        entity_type=EntityType.TAXON,
        ontology_source="ncbi_taxonomy",
        synonyms=synonym_records,
        created_at=now,
        updated_at=now,
    )
    success, error = registry.register(record)
    assert success, (
        f"Failed to register entity canonical_id={canonical_id!r} "
        f"primary_name={primary_name!r}: {error}"
    )


def _normalised_set(forms: List[str]) -> Set[str]:
    """Return the set of non-empty normalised surface forms."""
    result = set()
    for f in forms:
        n = normalize_surface_form(f)
        if n:
            result.add(n)
    return result


# ---------------------------------------------------------------------------
# Property 3: No-Spurious-Merge
# **Validates: Requirements 15.3, 6.5**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    ids=_two_distinct_taxon_ids_st,
    synonyms_e1=_synonym_list_st,
    synonyms_e2=_synonym_list_st,
)
def test_property_no_spurious_merge(
    ids: tuple[str, str],
    synonyms_e1: List[str],
    synonyms_e2: List[str],
) -> None:
    """
    **Property 3: No-Spurious-Merge**

    **Validates: Requirements 15.3, 6.5**

    For two distinct canonical entities E₁ and E₂ with disjoint synonym sets
    S₁ and S₂:

    1. For every synonym s₁ ∈ S₁, ``resolve(s₁)`` must NOT return E₂'s
       ``canonical_id``.
    2. For every synonym s₂ ∈ S₂, ``resolve(s₂)`` must NOT return E₁'s
       ``canonical_id``.

    This ensures the pipeline never spuriously merges two distinct entities
    by routing a synonym of one to the other's canonical ID.
    """
    id_e1, id_e2 = ids  # guaranteed distinct by the strategy filter

    # Build the full synonym sets (including the primary name, which is the
    # first element of each list).
    all_forms_e1 = synonyms_e1
    all_forms_e2 = synonyms_e2

    # Compute normalised sets to check for overlap
    norm_e1 = _normalised_set(all_forms_e1)
    norm_e2 = _normalised_set(all_forms_e2)

    # Precondition: the two synonym sets must be disjoint after normalisation
    # and each must have at least one non-empty normalised form.
    assume(len(norm_e1) > 0)
    assume(len(norm_e2) > 0)
    assume(norm_e1.isdisjoint(norm_e2))

    # Use a fresh in-memory database for each example to avoid cross-contamination
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        # Register E₁ with its synonyms
        primary_e1 = all_forms_e1[0]
        extra_e1 = all_forms_e1[1:]
        _register_entity_with_synonyms(registry, id_e1, primary_e1, extra_e1)

        # Register E₂ with its synonyms
        primary_e2 = all_forms_e2[0]
        extra_e2 = all_forms_e2[1:]
        _register_entity_with_synonyms(registry, id_e2, primary_e2, extra_e2)

        # Assert: no synonym of E₁ resolves to E₂'s canonical_id
        for s1 in all_forms_e1:
            norm = normalize_surface_form(s1)
            if not norm:
                continue  # skip forms that normalise to empty
            result = pipeline.resolve(s1, "taxon", "test_paper_no_spurious_merge")
            assert result.canonical_id != id_e2, (
                f"Spurious merge detected: synonym {s1!r} of E₁ (id={id_e1!r}) "
                f"resolved to E₂'s canonical_id={id_e2!r}. "
                f"winning_strategy={result.winning_strategy!r}, "
                f"grounding_confidence={result.grounding_confidence}"
            )

        # Assert: no synonym of E₂ resolves to E₁'s canonical_id
        for s2 in all_forms_e2:
            norm = normalize_surface_form(s2)
            if not norm:
                continue  # skip forms that normalise to empty
            result = pipeline.resolve(s2, "taxon", "test_paper_no_spurious_merge")
            assert result.canonical_id != id_e1, (
                f"Spurious merge detected: synonym {s2!r} of E₂ (id={id_e2!r}) "
                f"resolved to E₁'s canonical_id={id_e1!r}. "
                f"winning_strategy={result.winning_strategy!r}, "
                f"grounding_confidence={result.grounding_confidence}"
            )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests — explicit examples
# ---------------------------------------------------------------------------


def test_no_spurious_merge_explicit_ecoli_vs_bfragilis() -> None:
    """
    Explicit example: E. coli synonyms must not resolve to B. fragilis and
    vice versa.

    E₁: canonical_id="562", synonyms=["Escherichia coli", "E. coli", "ATCC 25922"]
    E₂: canonical_id="817", synonyms=["Bacteroides fragilis", "B. fragilis"]
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        now = datetime.now(timezone.utc)

        # Register E₁
        record_e1 = CanonicalEntityRecord(
            canonical_id="562",
            primary_name="Escherichia coli",
            entity_type=EntityType.TAXON,
            ontology_source="ncbi_taxonomy",
            synonyms=[
                SynonymRecord(
                    surface_form="E. coli",
                    provenance=SynonymProvenance.CURATOR,
                    added_by="test",
                    added_at=now,
                ),
                SynonymRecord(
                    surface_form="ATCC 25922",
                    provenance=SynonymProvenance.CURATOR,
                    added_by="test",
                    added_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        success, error = registry.register(record_e1)
        assert success, f"Failed to register E. coli: {error}"

        # Register E₂
        record_e2 = CanonicalEntityRecord(
            canonical_id="817",
            primary_name="Bacteroides fragilis",
            entity_type=EntityType.TAXON,
            ontology_source="ncbi_taxonomy",
            synonyms=[
                SynonymRecord(
                    surface_form="B. fragilis",
                    provenance=SynonymProvenance.CURATOR,
                    added_by="test",
                    added_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        success, error = registry.register(record_e2)
        assert success, f"Failed to register B. fragilis: {error}"

        # E₁ synonyms must not resolve to E₂'s canonical_id
        for s1 in ["Escherichia coli", "E. coli", "ATCC 25922"]:
            result = pipeline.resolve(s1, "taxon", "paper_001")
            assert result.canonical_id != "817", (
                f"Spurious merge: {s1!r} resolved to B. fragilis (817) "
                f"instead of E. coli (562). strategy={result.winning_strategy!r}"
            )
            assert result.canonical_id == "562", (
                f"Expected E. coli (562) for {s1!r}, got {result.canonical_id!r}"
            )

        # E₂ synonyms must not resolve to E₁'s canonical_id
        for s2 in ["Bacteroides fragilis", "B. fragilis"]:
            result = pipeline.resolve(s2, "taxon", "paper_001")
            assert result.canonical_id != "562", (
                f"Spurious merge: {s2!r} resolved to E. coli (562) "
                f"instead of B. fragilis (817). strategy={result.winning_strategy!r}"
            )
            assert result.canonical_id == "817", (
                f"Expected B. fragilis (817) for {s2!r}, got {result.canonical_id!r}"
            )

    finally:
        conn.close()


def test_no_spurious_merge_method_entities() -> None:
    """
    Explicit example with method entities: 16S rRNA and metagenomics synonyms
    must not cross-resolve.

    E₁: canonical_id="METHOD-16S", synonyms=["16S rRNA sequencing", "16S"]
    E₂: canonical_id="METHOD-META", synonyms=["metagenomics", "shotgun sequencing"]
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        now = datetime.now(timezone.utc)

        record_e1 = CanonicalEntityRecord(
            canonical_id="METHOD-16S",
            primary_name="16S rRNA sequencing",
            entity_type=EntityType.METHOD,
            ontology_source="internal",
            synonyms=[
                SynonymRecord(
                    surface_form="16S",
                    provenance=SynonymProvenance.CURATOR,
                    added_by="test",
                    added_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        success, error = registry.register(record_e1)
        assert success, f"Failed to register METHOD-16S: {error}"

        record_e2 = CanonicalEntityRecord(
            canonical_id="METHOD-META",
            primary_name="metagenomics",
            entity_type=EntityType.METHOD,
            ontology_source="internal",
            synonyms=[
                SynonymRecord(
                    surface_form="shotgun sequencing",
                    provenance=SynonymProvenance.CURATOR,
                    added_by="test",
                    added_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        success, error = registry.register(record_e2)
        assert success, f"Failed to register METHOD-META: {error}"

        # E₁ synonyms must not resolve to E₂
        for s1 in ["16S rRNA sequencing", "16S"]:
            result = pipeline.resolve(s1, "method", "paper_002")
            assert result.canonical_id != "METHOD-META", (
                f"Spurious merge: {s1!r} resolved to METHOD-META. "
                f"strategy={result.winning_strategy!r}"
            )

        # E₂ synonyms must not resolve to E₁
        for s2 in ["metagenomics", "shotgun sequencing"]:
            result = pipeline.resolve(s2, "method", "paper_002")
            assert result.canonical_id != "METHOD-16S", (
                f"Spurious merge: {s2!r} resolved to METHOD-16S. "
                f"strategy={result.winning_strategy!r}"
            )

    finally:
        conn.close()

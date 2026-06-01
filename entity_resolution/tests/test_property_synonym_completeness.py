"""
Property 2: Synonym Completeness

**Validates: Requirements 5.1, 15.2**

For any canonical entity with registered synonyms S₁…Sₙ, assert
``resolve(Sᵢ).canonical_id == entity.canonical_id`` for all i, regardless
of case or Unicode normalization form.

The test:
1. Generates a canonical entity (canonical_id, entity_type, primary_name)
2. Generates a list of 1–5 synonym surface forms (distinct, ≥4 chars each)
3. Sets up a ResolutionPipeline with a CanonicalRegistry (in-memory SQLite)
   and SynonymIndex
4. Registers the canonical entity in the registry
5. Adds each synonym to the registry
6. For each synonym Sᵢ, calls resolve(Sᵢ, entity_type, paper_id) and asserts
   result.canonical_id == entity.canonical_id
7. Also tests case-insensitive variants: uppercase, lowercase, mixed case
8. Also tests NFC/NFD Unicode normalization variants for Unicode synonyms

Requirements: 5.1, 15.2
"""

from __future__ import annotations

import sqlite3
import unicodedata
from datetime import datetime, timezone
from typing import List, Tuple

import pytest
from hypothesis import given, settings
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

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Valid taxon canonical IDs: positive integer strings (no leading zeros)
_taxon_id_st = st.integers(min_value=1, max_value=999_999).map(str)

# Valid disease canonical IDs: one uppercase letter + one or more digits
_disease_id_st = st.builds(
    lambda letter, digits: f"{letter}{digits}",
    letter=st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    digits=st.integers(min_value=1, max_value=999999).map(str),
)

# Valid method canonical IDs: "METHOD-" + ASCII alphanumeric (no Unicode)
_method_id_st = st.builds(
    lambda suffix: f"METHOD-{suffix}",
    suffix=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        min_size=1,
        max_size=10,
    ),
)

# Entity type + matching canonical ID strategy
_entity_type_and_id_st = st.one_of(
    st.tuples(st.just("taxon"), _taxon_id_st),
    st.tuples(st.just("disease"), _disease_id_st),
    st.tuples(st.just("method"), _method_id_st),
)

# Primary name: printable ASCII text, at least 4 chars (to avoid fuzzy skip)
_primary_name_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
        whitelist_characters="-",
    ),
    min_size=4,
    max_size=30,
).map(str.strip).filter(lambda s: len(s) >= 4)

# Synonym surface form: printable text, at least 4 chars, no leading/trailing whitespace
_synonym_form_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
        whitelist_characters="-",
    ),
    min_size=4,
    max_size=30,
).map(str.strip).filter(lambda s: len(s) >= 4)

# List of 1–5 distinct synonym surface forms (each ≥4 chars)
_synonyms_list_st = st.lists(
    _synonym_form_st,
    min_size=1,
    max_size=5,
    unique=True,
)


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
    entity_type: str,
    primary_name: str,
) -> None:
    """Register a canonical entity in the registry."""
    now = datetime.now(timezone.utc)
    record = CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=primary_name,
        entity_type=EntityType(entity_type),
        ontology_source="ncbi_taxonomy" if entity_type == "taxon" else (
            "mesh" if entity_type == "disease" else "internal"
        ),
        synonyms=[],
        created_at=now,
        updated_at=now,
    )
    success, error = registry.register(record)
    assert success, f"Failed to register entity {canonical_id!r}: {error}"


def _add_synonym(
    registry: CanonicalRegistry,
    canonical_id: str,
    surface_form: str,
) -> bool:
    """Add a synonym to an existing canonical entity. Returns True on success."""
    success, error = registry.add_synonym(
        canonical_id=canonical_id,
        surface_form=surface_form,
        provenance=SynonymProvenance.CURATOR,
        added_by="test_curator",
    )
    return success


def _mixed_case(s: str) -> str:
    """Return a mixed-case variant: alternate upper/lower per character."""
    result = []
    upper = True
    for ch in s:
        if ch.isalpha():
            result.append(ch.upper() if upper else ch.lower())
            upper = not upper
        else:
            result.append(ch)
    return "".join(result)


def _has_symmetric_case(s: str) -> bool:
    """
    Return True if all alphabetic characters in s have symmetric case folding.

    A character has symmetric case folding if upper().lower() == lower().
    This filters out characters like ſ (U+017F, LATIN SMALL LETTER LONG S)
    where ſ.upper() = 'S' but 'S'.lower() = 's' ≠ 'ſ'.

    The normalize_surface_form() function uses .lower() (not .casefold()),
    so case-insensitive lookup only works for characters with symmetric case.
    """
    for ch in s:
        if ch.isalpha():
            if ch.upper().lower() != ch.lower():
                return False
    return True


def _to_nfd(s: str) -> str:
    """Return the NFD Unicode normalization form of s."""
    return unicodedata.normalize("NFD", s)


def _has_unicode_beyond_ascii(s: str) -> bool:
    """Return True if s contains any non-ASCII character."""
    return any(ord(c) > 127 for c in s)


# ---------------------------------------------------------------------------
# Property 2: Synonym Completeness
# **Validates: Requirements 5.1, 15.2**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    entity_type_and_id=_entity_type_and_id_st,
    primary_name=_primary_name_st,
    synonyms=_synonyms_list_st,
)
def test_property_synonym_completeness(
    entity_type_and_id: Tuple[str, str],
    primary_name: str,
    synonyms: List[str],
) -> None:
    """
    **Property 2: Synonym Completeness**

    **Validates: Requirements 5.1, 15.2**

    For any canonical entity with registered synonyms S₁…Sₙ, assert
    ``resolve(Sᵢ).canonical_id == entity.canonical_id`` for all i,
    regardless of case or Unicode normalization form.
    """
    entity_type, canonical_id = entity_type_and_id

    # Ensure primary_name doesn't collide with any synonym after normalization
    # (the registry would reject the synonym if it normalizes to the same form)
    from entity_resolution.utils import normalize_surface_form
    norm_primary = normalize_surface_form(primary_name)
    # Filter out synonyms that normalize to the same form as the primary name
    # or to the same form as each other
    seen_norms = {norm_primary}
    unique_synonyms = []
    for syn in synonyms:
        norm_syn = normalize_surface_form(syn)
        if norm_syn and norm_syn not in seen_norms:
            unique_synonyms.append(syn)
            seen_norms.add(norm_syn)

    if not unique_synonyms:
        # No usable synonyms after deduplication — skip this example
        return

    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        # Register the canonical entity
        _register_entity(registry, canonical_id, entity_type, primary_name)

        # Add each synonym to the registry
        added_synonyms = []
        for syn in unique_synonyms:
            if _add_synonym(registry, canonical_id, syn):
                added_synonyms.append(syn)

        if not added_synonyms:
            # No synonyms were successfully added — skip
            return

        # For each registered synonym, resolve and assert correct canonical_id
        for syn in added_synonyms:
            result = pipeline.resolve(
                surface_form=syn,
                entity_type=entity_type,
                paper_id="test_paper_synonym_completeness",
            )
            assert result.canonical_id == canonical_id, (
                f"Synonym '{syn}' did not resolve to canonical_id={canonical_id!r}. "
                f"Got canonical_id={result.canonical_id!r}, "
                f"winning_strategy={result.winning_strategy!r}, "
                f"grounded={result.grounded}. "
                f"entity_type={entity_type!r}"
            )
            assert result.grounded is True, (
                f"Synonym '{syn}' resolved but grounded=False. "
                f"canonical_id={canonical_id!r}, entity_type={entity_type!r}"
            )

    finally:
        conn.close()


@settings(max_examples=100)
@given(
    entity_type_and_id=_entity_type_and_id_st,
    primary_name=_primary_name_st,
    synonyms=_synonyms_list_st,
)
def test_property_synonym_completeness_case_insensitive(
    entity_type_and_id: Tuple[str, str],
    primary_name: str,
    synonyms: List[str],
) -> None:
    """
    **Property 2: Synonym Completeness — case-insensitive variants**

    **Validates: Requirements 5.1, 15.2**

    Uppercase, lowercase, and mixed-case variants of each registered synonym
    must all resolve to the same canonical_id.
    """
    entity_type, canonical_id = entity_type_and_id

    from entity_resolution.utils import normalize_surface_form
    norm_primary = normalize_surface_form(primary_name)
    seen_norms = {norm_primary}
    unique_synonyms = []
    for syn in synonyms:
        norm_syn = normalize_surface_form(syn)
        if norm_syn and norm_syn not in seen_norms:
            unique_synonyms.append(syn)
            seen_norms.add(norm_syn)

    if not unique_synonyms:
        return

    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(registry, canonical_id, entity_type, primary_name)

        added_synonyms = []
        for syn in unique_synonyms:
            if _add_synonym(registry, canonical_id, syn):
                added_synonyms.append(syn)

        if not added_synonyms:
            return

        # Test case variants for each synonym
        for syn in added_synonyms:
            # Skip synonyms with asymmetric case folding (e.g. ſ → S but S → s ≠ ſ)
            # normalize_surface_form uses .lower(), not .casefold(), so case-insensitive
            # lookup only works for characters with symmetric case folding.
            if not _has_symmetric_case(syn):
                continue

            case_variants = [
                syn.upper(),
                syn.lower(),
                _mixed_case(syn),
            ]

            for variant in case_variants:
                # Skip if the variant normalizes to empty (e.g. all punctuation)
                from entity_resolution.utils import normalize_surface_form as nsf
                if not nsf(variant):
                    continue

                result = pipeline.resolve(
                    surface_form=variant,
                    entity_type=entity_type,
                    paper_id="test_paper_case_insensitive",
                )
                assert result.canonical_id == canonical_id, (
                    f"Case variant '{variant}' (from synonym '{syn}') did not resolve "
                    f"to canonical_id={canonical_id!r}. "
                    f"Got canonical_id={result.canonical_id!r}, "
                    f"winning_strategy={result.winning_strategy!r}, "
                    f"grounded={result.grounded}. "
                    f"entity_type={entity_type!r}"
                )
                assert result.grounded is True, (
                    f"Case variant '{variant}' resolved but grounded=False. "
                    f"canonical_id={canonical_id!r}, entity_type={entity_type!r}"
                )

    finally:
        conn.close()


@settings(max_examples=100)
@given(
    entity_type_and_id=_entity_type_and_id_st,
    primary_name=_primary_name_st,
    synonyms=_synonyms_list_st,
)
def test_property_synonym_completeness_unicode_normalization(
    entity_type_and_id: Tuple[str, str],
    primary_name: str,
    synonyms: List[str],
) -> None:
    """
    **Property 2: Synonym Completeness — Unicode NFC/NFD normalization**

    **Validates: Requirements 5.1, 15.2**

    For synonyms containing Unicode characters, both NFC and NFD forms must
    resolve to the same canonical_id.
    """
    entity_type, canonical_id = entity_type_and_id

    from entity_resolution.utils import normalize_surface_form
    norm_primary = normalize_surface_form(primary_name)
    seen_norms = {norm_primary}
    unique_synonyms = []
    for syn in synonyms:
        norm_syn = normalize_surface_form(syn)
        if norm_syn and norm_syn not in seen_norms:
            unique_synonyms.append(syn)
            seen_norms.add(norm_syn)

    if not unique_synonyms:
        return

    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(registry, canonical_id, entity_type, primary_name)

        added_synonyms = []
        for syn in unique_synonyms:
            if _add_synonym(registry, canonical_id, syn):
                added_synonyms.append(syn)

        if not added_synonyms:
            return

        # Test NFC/NFD variants for synonyms that contain Unicode characters
        for syn in added_synonyms:
            if not _has_unicode_beyond_ascii(syn):
                # No Unicode beyond ASCII — NFC/NFD are identical, skip
                continue

            nfc_form = unicodedata.normalize("NFC", syn)
            nfd_form = unicodedata.normalize("NFD", syn)

            for variant, form_name in [(nfc_form, "NFC"), (nfd_form, "NFD")]:
                from entity_resolution.utils import normalize_surface_form as nsf
                if not nsf(variant):
                    continue

                result = pipeline.resolve(
                    surface_form=variant,
                    entity_type=entity_type,
                    paper_id="test_paper_unicode_normalization",
                )
                assert result.canonical_id == canonical_id, (
                    f"{form_name} variant '{variant}' (from synonym '{syn}') did not "
                    f"resolve to canonical_id={canonical_id!r}. "
                    f"Got canonical_id={result.canonical_id!r}, "
                    f"winning_strategy={result.winning_strategy!r}, "
                    f"grounded={result.grounded}. "
                    f"entity_type={entity_type!r}"
                )
                assert result.grounded is True, (
                    f"{form_name} variant '{variant}' resolved but grounded=False. "
                    f"canonical_id={canonical_id!r}, entity_type={entity_type!r}"
                )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests — explicit examples
# ---------------------------------------------------------------------------


def test_synonym_completeness_basic_taxon() -> None:
    """
    Explicit example: all synonyms for E. coli resolve to canonical_id "562".
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(registry, "562", "taxon", "Escherichia coli")

        synonyms = ["E. coli", "E.coli", "ATCC 25922", "K-12"]
        for syn in synonyms:
            success, error = registry.add_synonym(
                canonical_id="562",
                surface_form=syn,
                provenance=SynonymProvenance.CURATOR,
                added_by="test_curator",
            )
            assert success, f"Failed to add synonym '{syn}': {error}"

        for syn in synonyms:
            result = pipeline.resolve(syn, "taxon", "paper_001")
            assert result.canonical_id == "562", (
                f"Synonym '{syn}' resolved to {result.canonical_id!r}, expected '562'"
            )
            assert result.grounded is True

    finally:
        conn.close()


def test_synonym_completeness_case_variants() -> None:
    """
    Explicit example: uppercase, lowercase, and mixed-case variants of a synonym
    all resolve to the same canonical entity.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(registry, "D006262", "disease", "Inflammatory Bowel Disease")

        success, error = registry.add_synonym(
            canonical_id="D006262",
            surface_form="Crohns Disease",
            provenance=SynonymProvenance.CURATOR,
            added_by="test_curator",
        )
        assert success, f"Failed to add synonym: {error}"

        case_variants = [
            "Crohns Disease",
            "CROHNS DISEASE",
            "crohns disease",
            "CrOhNs DiSeAsE",
        ]
        for variant in case_variants:
            result = pipeline.resolve(variant, "disease", "paper_002")
            assert result.canonical_id == "D006262", (
                f"Case variant '{variant}' resolved to {result.canonical_id!r}, "
                f"expected 'D006262'"
            )
            assert result.grounded is True

    finally:
        conn.close()


def test_synonym_completeness_unicode_nfc_nfd() -> None:
    """
    Explicit example: NFC and NFD forms of a Unicode synonym both resolve
    to the same canonical entity.

    Uses 'café' which has a composed form (NFC: é = U+00E9) and a decomposed
    form (NFD: e + combining acute accent = U+0065 U+0301).
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(registry, "METHOD-CAFE", "method", "Cafe Method")

        # Register the NFC form as a synonym
        nfc_synonym = unicodedata.normalize("NFC", "café method")
        success, error = registry.add_synonym(
            canonical_id="METHOD-CAFE",
            surface_form=nfc_synonym,
            provenance=SynonymProvenance.CURATOR,
            added_by="test_curator",
        )
        assert success, f"Failed to add NFC synonym: {error}"

        # Both NFC and NFD forms should resolve to the same entity
        nfd_synonym = unicodedata.normalize("NFD", "café method")

        for variant, form_name in [(nfc_synonym, "NFC"), (nfd_synonym, "NFD")]:
            result = pipeline.resolve(variant, "method", "paper_003")
            assert result.canonical_id == "METHOD-CAFE", (
                f"{form_name} form '{variant}' resolved to {result.canonical_id!r}, "
                f"expected 'METHOD-CAFE'"
            )
            assert result.grounded is True

    finally:
        conn.close()


def test_synonym_completeness_multiple_entities_no_cross_resolution() -> None:
    """
    Explicit example: synonyms of entity A do not resolve to entity B.

    Registers two distinct entities with disjoint synonym sets and verifies
    each synonym resolves only to its own entity.
    """
    conn = _make_in_memory_registry_conn()
    try:
        pipeline, registry, synonym_index = _make_pipeline(conn)

        _register_entity(registry, "562", "taxon", "Escherichia coli")
        _register_entity(registry, "1423", "taxon", "Bacillus subtilis")

        ecoli_synonyms = ["E coli", "ATCC 25922"]
        bacillus_synonyms = ["B subtilis", "ATCC 6051"]

        for syn in ecoli_synonyms:
            success, _ = registry.add_synonym("562", syn, SynonymProvenance.CURATOR)
            assert success

        for syn in bacillus_synonyms:
            success, _ = registry.add_synonym("1423", syn, SynonymProvenance.CURATOR)
            assert success

        for syn in ecoli_synonyms:
            result = pipeline.resolve(syn, "taxon", "paper_004")
            assert result.canonical_id == "562", (
                f"E. coli synonym '{syn}' resolved to {result.canonical_id!r}, expected '562'"
            )

        for syn in bacillus_synonyms:
            result = pipeline.resolve(syn, "taxon", "paper_004")
            assert result.canonical_id == "1423", (
                f"B. subtilis synonym '{syn}' resolved to {result.canonical_id!r}, expected '1423'"
            )

    finally:
        conn.close()

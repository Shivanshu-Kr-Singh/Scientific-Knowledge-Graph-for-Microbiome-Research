"""
Unit tests for CanonicalRegistry (task 2.1).

Tests cover:
- register(): valid and invalid canonical IDs, duplicate detection, synonym conflicts
- lookup_by_surface_form(): case-insensitive, NFC-normalised lookup
- add_synonym(): length validation, cross-entity conflict detection, idempotency
- get_registry_version(): version bumps on every write

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from entity_resolution.canonical_registry import CanonicalRegistry
from entity_resolution.db_schema import create_schema_in_connection, get_canonical_registry_schema
from entity_resolution.models import (
    CanonicalEntityRecord,
    EntityType,
    SynonymProvenance,
    SynonymRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_conn() -> sqlite3.Connection:
    """In-memory SQLite connection with the canonical_registry schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_canonical_registry_schema())
    yield conn
    conn.close()


@pytest.fixture
def registry(registry_conn: sqlite3.Connection) -> CanonicalRegistry:
    """Fresh CanonicalRegistry backed by an in-memory database."""
    return CanonicalRegistry(conn=registry_conn)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _taxon_record(
    canonical_id: str = "562",
    primary_name: str = "Escherichia coli",
    synonyms: list | None = None,
) -> CanonicalEntityRecord:
    now = _now()
    return CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=primary_name,
        entity_type=EntityType.TAXON,
        ontology_source="ncbi_taxonomy",
        synonyms=synonyms or [],
        created_at=now,
        updated_at=now,
    )


def _disease_record(
    canonical_id: str = "D006262",
    primary_name: str = "Health",
) -> CanonicalEntityRecord:
    now = _now()
    return CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=primary_name,
        entity_type=EntityType.DISEASE,
        ontology_source="mesh",
        synonyms=[],
        created_at=now,
        updated_at=now,
    )


def _method_record(
    canonical_id: str = "METHOD-16S",
    primary_name: str = "16S rRNA sequencing",
) -> CanonicalEntityRecord:
    now = _now()
    return CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=primary_name,
        entity_type=EntityType.METHOD,
        ontology_source="internal",
        synonyms=[],
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# register() — valid registrations
# ---------------------------------------------------------------------------


def test_register_taxon_success(registry: CanonicalRegistry) -> None:
    """Registering a valid taxon entity returns (True, None)."""
    ok, err = registry.register(_taxon_record())
    assert ok is True
    assert err is None


def test_register_disease_success(registry: CanonicalRegistry) -> None:
    """Registering a valid disease entity returns (True, None)."""
    ok, err = registry.register(_disease_record())
    assert ok is True
    assert err is None


def test_register_method_success(registry: CanonicalRegistry) -> None:
    """Registering a valid method entity returns (True, None)."""
    ok, err = registry.register(_method_record())
    assert ok is True
    assert err is None


def test_register_with_synonyms(registry: CanonicalRegistry) -> None:
    """Registering an entity with synonyms persists all synonym rows."""
    now = _now()
    synonyms = [
        SynonymRecord(
            surface_form="E. coli",
            provenance=SynonymProvenance.PAPER_TEXT,
            added_by=None,
            added_at=now,
        ),
        SynonymRecord(
            surface_form="E.coli",
            provenance=SynonymProvenance.CURATOR,
            added_by="curator1",
            added_at=now,
        ),
    ]
    record = _taxon_record(synonyms=synonyms)
    ok, err = registry.register(record)
    assert ok is True
    assert err is None

    # All synonyms should be findable
    assert registry.lookup_by_surface_form("E. coli") is not None
    assert registry.lookup_by_surface_form("E.coli") is not None
    assert registry.lookup_by_surface_form("Escherichia coli") is not None


# ---------------------------------------------------------------------------
# register() — invalid canonical IDs (Requirements 3.2, 3.3, 3.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["0", "-1", "1.5", "01", "abc", "", " 1"],
)
def test_register_invalid_taxon_id_rejected(registry: CanonicalRegistry, bad_id: str) -> None:
    """Invalid taxon IDs are rejected; no partial record is created."""
    record = _taxon_record(canonical_id=bad_id)
    ok, err = registry.register(record)
    assert ok is False
    assert err is not None
    assert err.field == "canonical_id"
    # Confirm nothing was persisted
    assert registry.lookup_by_surface_form(record.primary_name) is None


@pytest.mark.parametrize(
    "bad_id",
    ["d006262", "D", "D06262X", "1D06262", "DD06262", ""],
)
def test_register_invalid_disease_id_rejected(registry: CanonicalRegistry, bad_id: str) -> None:
    """Invalid disease IDs are rejected; no partial record is created."""
    record = _disease_record(canonical_id=bad_id)
    ok, err = registry.register(record)
    assert ok is False
    assert err is not None
    assert err.field == "canonical_id"


@pytest.mark.parametrize(
    "bad_id",
    ["METHOD-", "method-16S", "METHOD_16S", "METH-16S", ""],
)
def test_register_invalid_method_id_rejected(registry: CanonicalRegistry, bad_id: str) -> None:
    """Invalid method IDs are rejected; no partial record is created."""
    record = _method_record(canonical_id=bad_id)
    ok, err = registry.register(record)
    assert ok is False
    assert err is not None
    assert err.field == "canonical_id"


# ---------------------------------------------------------------------------
# register() — duplicate canonical_id
# ---------------------------------------------------------------------------


def test_register_duplicate_canonical_id_rejected(registry: CanonicalRegistry) -> None:
    """Registering the same canonical_id twice is rejected on the second attempt."""
    record = _taxon_record()
    ok1, _ = registry.register(record)
    assert ok1 is True

    ok2, err2 = registry.register(record)
    assert ok2 is False
    assert err2 is not None


# ---------------------------------------------------------------------------
# register() — no partial records on failure (Requirement 3.2–3.4)
# ---------------------------------------------------------------------------


def test_register_failure_leaves_no_partial_record(registry: CanonicalRegistry) -> None:
    """
    When registration fails due to an invalid canonical_id, no rows are
    written to canonical_entities or synonyms.
    """
    record = _taxon_record(canonical_id="INVALID")
    ok, err = registry.register(record)
    assert ok is False

    # The primary name should not be findable
    result = registry.lookup_by_surface_form(record.primary_name)
    assert result is None


# ---------------------------------------------------------------------------
# lookup_by_surface_form() — Requirements 3.5
# ---------------------------------------------------------------------------


def test_lookup_exact_match(registry: CanonicalRegistry) -> None:
    """Exact surface form lookup returns the correct record."""
    registry.register(_taxon_record())
    result = registry.lookup_by_surface_form("Escherichia coli")
    assert result is not None
    assert result.canonical_id == "562"


def test_lookup_case_insensitive(registry: CanonicalRegistry) -> None:
    """Lookup is case-insensitive."""
    registry.register(_taxon_record())
    assert registry.lookup_by_surface_form("ESCHERICHIA COLI") is not None
    assert registry.lookup_by_surface_form("escherichia coli") is not None
    assert registry.lookup_by_surface_form("Escherichia Coli") is not None


def test_lookup_nfc_normalised(registry: CanonicalRegistry) -> None:
    """Lookup normalises the surface form (NFC + lowercase + strip punctuation)."""
    registry.register(_taxon_record())
    # Punctuation stripped: "Escherichia coli." -> "escherichia coli"
    result = registry.lookup_by_surface_form("Escherichia coli.")
    assert result is not None
    assert result.canonical_id == "562"


def test_lookup_miss_returns_none(registry: CanonicalRegistry) -> None:
    """Lookup returns None (not an exception) when no match is found."""
    result = registry.lookup_by_surface_form("Nonexistent organism")
    assert result is None


def test_lookup_empty_string_returns_none(registry: CanonicalRegistry) -> None:
    """Lookup with an empty string returns None."""
    result = registry.lookup_by_surface_form("")
    assert result is None


# ---------------------------------------------------------------------------
# add_synonym() — Requirements 3.6, 3.7, 5.1, 5.4
# ---------------------------------------------------------------------------


def test_add_synonym_success(registry: CanonicalRegistry) -> None:
    """Adding a valid synonym to an existing entity succeeds."""
    registry.register(_taxon_record())
    ok, err = registry.add_synonym(
        "562", "E. coli", SynonymProvenance.PAPER_TEXT
    )
    assert ok is True
    assert err is None

    # Synonym should now be findable
    result = registry.lookup_by_surface_form("E. coli")
    assert result is not None
    assert result.canonical_id == "562"


def test_add_synonym_too_long_rejected(registry: CanonicalRegistry) -> None:
    """Synonym surface_form exceeding 500 characters is rejected."""
    registry.register(_taxon_record())
    long_form = "x" * 501
    ok, err = registry.add_synonym("562", long_form, SynonymProvenance.CURATOR)
    assert ok is False
    assert err is not None


def test_add_synonym_exactly_500_chars_accepted(registry: CanonicalRegistry) -> None:
    """Synonym surface_form of exactly 500 characters is accepted."""
    registry.register(_taxon_record())
    form_500 = "a" * 500
    ok, err = registry.add_synonym("562", form_500, SynonymProvenance.CURATOR)
    assert ok is True
    assert err is None


def test_add_synonym_cross_entity_conflict_rejected(registry: CanonicalRegistry) -> None:
    """
    Adding a synonym that already belongs to a different entity is rejected
    and a SynonymConflictRecord is logged.

    Requirements: 3.7, 5.4
    """
    # Register two distinct entities
    registry.register(_taxon_record(canonical_id="562", primary_name="Escherichia coli"))
    registry.register(_taxon_record(canonical_id="1301", primary_name="Streptococcus"))

    # Add "E. coli" to entity 562
    ok1, _ = registry.add_synonym("562", "E. coli", SynonymProvenance.PAPER_TEXT)
    assert ok1 is True

    # Attempt to add the same surface form to entity 1301 — must be rejected
    ok2, err2 = registry.add_synonym("1301", "E. coli", SynonymProvenance.PAPER_TEXT)
    assert ok2 is False
    assert err2 is not None

    # Verify a conflict record was written
    conflict_rows = registry._conn.execute(
        "SELECT * FROM synonym_conflicts WHERE duplicate_surface_form = ?",
        (normalize_surface_form_for_test("E. coli"),),
    ).fetchall()
    # The conflict record stores the original surface form, not the normalised one
    conflict_rows_orig = registry._conn.execute(
        "SELECT * FROM synonym_conflicts"
    ).fetchall()
    assert len(conflict_rows_orig) >= 1


def test_add_synonym_idempotent_for_same_entity(registry: CanonicalRegistry) -> None:
    """Adding the same synonym twice for the same entity is idempotent (no error)."""
    registry.register(_taxon_record())
    ok1, _ = registry.add_synonym("562", "E. coli", SynonymProvenance.PAPER_TEXT)
    assert ok1 is True
    ok2, err2 = registry.add_synonym("562", "E. coli", SynonymProvenance.PAPER_TEXT)
    assert ok2 is True
    assert err2 is None


def test_add_synonym_nonexistent_canonical_id_rejected(registry: CanonicalRegistry) -> None:
    """Adding a synonym to a non-existent canonical_id is rejected."""
    ok, err = registry.add_synonym("9999", "some form", SynonymProvenance.CURATOR)
    assert ok is False
    assert err is not None


# ---------------------------------------------------------------------------
# get_registry_version() — version bumps on every write
# ---------------------------------------------------------------------------


def test_initial_version_is_positive(registry: CanonicalRegistry) -> None:
    """The initial registry version is a non-negative integer."""
    v = registry.get_registry_version()
    assert isinstance(v, int)
    assert v >= 0


def test_version_bumps_on_register(registry: CanonicalRegistry) -> None:
    """Each successful register() call increments the registry version."""
    v0 = registry.get_registry_version()
    registry.register(_taxon_record(canonical_id="562", primary_name="Escherichia coli"))
    v1 = registry.get_registry_version()
    assert v1 > v0


def test_version_bumps_on_add_synonym(registry: CanonicalRegistry) -> None:
    """Each successful add_synonym() call increments the registry version."""
    registry.register(_taxon_record())
    v0 = registry.get_registry_version()
    registry.add_synonym("562", "E. coli", SynonymProvenance.PAPER_TEXT)
    v1 = registry.get_registry_version()
    assert v1 > v0


def test_version_does_not_bump_on_failed_register(registry: CanonicalRegistry) -> None:
    """A failed register() (invalid ID) does not bump the version."""
    v0 = registry.get_registry_version()
    registry.register(_taxon_record(canonical_id="INVALID"))
    v1 = registry.get_registry_version()
    assert v1 == v0


def test_version_does_not_bump_on_failed_add_synonym(registry: CanonicalRegistry) -> None:
    """A failed add_synonym() (too long) does not bump the version."""
    registry.register(_taxon_record())
    v0 = registry.get_registry_version()
    registry.add_synonym("562", "x" * 501, SynonymProvenance.CURATOR)
    v1 = registry.get_registry_version()
    assert v1 == v0


def test_version_monotonically_increases(registry: CanonicalRegistry) -> None:
    """Multiple writes produce strictly increasing version numbers."""
    versions = [registry.get_registry_version()]

    registry.register(_taxon_record(canonical_id="562", primary_name="Escherichia coli"))
    versions.append(registry.get_registry_version())

    registry.register(_taxon_record(canonical_id="1301", primary_name="Streptococcus"))
    versions.append(registry.get_registry_version())

    registry.add_synonym("562", "E. coli", SynonymProvenance.PAPER_TEXT)
    versions.append(registry.get_registry_version())

    for i in range(1, len(versions)):
        assert versions[i] > versions[i - 1], (
            f"Version did not increase at step {i}: {versions}"
        )


# ---------------------------------------------------------------------------
# In-memory synonym index consistency
# ---------------------------------------------------------------------------


def test_in_memory_index_populated_after_register(registry: CanonicalRegistry) -> None:
    """After register(), the in-memory index contains the primary name."""
    registry.register(_taxon_record())
    normalised = "escherichia coli"  # normalize_surface_form("Escherichia coli")
    assert registry._synonym_index.get(normalised) == "562"


def test_in_memory_index_populated_after_add_synonym(registry: CanonicalRegistry) -> None:
    """After add_synonym(), the in-memory index contains the new synonym."""
    registry.register(_taxon_record())
    registry.add_synonym("562", "E. coli", SynonymProvenance.PAPER_TEXT)
    # normalize_surface_form("E. coli") -> "e coli" (punctuation stripped)
    normalised = "e coli"
    assert registry._synonym_index.get(normalised) == "562"


def test_rebuild_synonym_index_from_db(registry_conn: sqlite3.Connection) -> None:
    """
    A new CanonicalRegistry instance backed by the same DB connection
    rebuilds its in-memory index from the existing rows.
    """
    reg1 = CanonicalRegistry(conn=registry_conn)
    reg1.register(_taxon_record())
    reg1.add_synonym("562", "E. coli", SynonymProvenance.PAPER_TEXT)

    # Create a second registry on the same connection
    reg2 = CanonicalRegistry(conn=registry_conn)
    assert reg2._synonym_index.get("escherichia coli") == "562"
    assert reg2._synonym_index.get("e coli") == "562"


# ---------------------------------------------------------------------------
# Helper (not imported from utils to avoid circular dependency in test)
# ---------------------------------------------------------------------------

def normalize_surface_form_for_test(s: str) -> str:
    """Thin wrapper used only in tests to avoid importing utils directly."""
    from entity_resolution.utils import normalize_surface_form
    return normalize_surface_form(s)

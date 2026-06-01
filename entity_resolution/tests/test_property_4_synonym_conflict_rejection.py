"""
Property 4: Synonym Conflict Rejection

**Validates: Requirements 3.7, 5.4**

Generate two distinct canonical entities and a shared surface form; assert
the second registration is rejected and a ``SynonymConflictRecord`` is written
to the ``synonym_conflicts`` table.

The property is tested via two code paths:
1. ``register()`` — the shared surface form is included in the second entity's
   synonym list at registration time.
2. ``add_synonym()`` — the shared surface form is added to the second entity
   after both entities are already registered.
"""

from __future__ import annotations

import sqlite3
import string
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.canonical_registry import CanonicalRegistry
from entity_resolution.db_schema import create_schema_in_connection, get_canonical_registry_schema
from entity_resolution.models import (
    CanonicalEntityRecord,
    EntityType,
    SynonymProvenance,
    SynonymRecord,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Valid taxon IDs: positive integer strings (no leading zeros)
_taxon_id_st = st.integers(min_value=1, max_value=999_999).map(str)

# Two *distinct* taxon IDs
_two_distinct_taxon_ids_st = st.tuples(_taxon_id_st, _taxon_id_st).filter(
    lambda pair: pair[0] != pair[1]
)

# Surface forms: non-empty printable text that normalises to a non-empty string.
# We use letters + digits + spaces so that normalize_surface_form() keeps at
# least one character.  Min length 3 ensures the normalised form is non-empty
# after stripping punctuation.
_surface_form_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters=" ",
    ),
    min_size=3,
    max_size=40,
).map(str.strip).filter(lambda s: len(s) >= 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> tuple[CanonicalRegistry, sqlite3.Connection]:
    """Return a fresh CanonicalRegistry backed by an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_canonical_registry_schema())
    registry = CanonicalRegistry(conn=conn)
    return registry, conn


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_taxon_record(
    canonical_id: str,
    primary_name: str,
    synonyms: list[SynonymRecord] | None = None,
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


def _conflict_count(conn: sqlite3.Connection) -> int:
    """Return the number of rows in the synonym_conflicts table."""
    row = conn.execute("SELECT COUNT(*) AS cnt FROM synonym_conflicts").fetchone()
    return int(row["cnt"])


# ---------------------------------------------------------------------------
# Property 4 — via register()
# **Validates: Requirements 3.7, 5.4**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    ids=_two_distinct_taxon_ids_st,
    shared_surface_form=_surface_form_st,
)
def test_property_synonym_conflict_rejection_via_register(
    ids: tuple[str, str],
    shared_surface_form: str,
) -> None:
    """
    **Property 4: Synonym Conflict Rejection (via register)**

    **Validates: Requirements 3.7, 5.4**

    When entity A is registered with a surface form S, attempting to register
    entity B with the same surface form S must:
    1. Return ``(False, RegistrationError)`` — the second registration is rejected.
    2. Write at least one ``SynonymConflictRecord`` to the ``synonym_conflicts``
       table.
    """
    id_a, id_b = ids  # guaranteed distinct by the filter above

    registry, conn = _make_registry()
    try:
        conflicts_before = _conflict_count(conn)

        # Register entity A with shared_surface_form as its primary name
        record_a = _make_taxon_record(
            canonical_id=id_a,
            primary_name=shared_surface_form,
        )
        ok_a, err_a = registry.register(record_a)
        assert ok_a is True, (
            f"First registration should succeed. "
            f"canonical_id={id_a!r}, surface_form={shared_surface_form!r}, "
            f"error={err_a}"
        )

        # Register entity B with the *same* surface form as its primary name
        # Use a different primary_name string that still normalises to the same
        # value — simplest approach: use the exact same string.
        record_b = _make_taxon_record(
            canonical_id=id_b,
            primary_name=shared_surface_form,
        )
        ok_b, err_b = registry.register(record_b)

        # Assertion 1: second registration must be rejected
        assert ok_b is False, (
            f"Second registration with the same surface form should be rejected. "
            f"id_a={id_a!r}, id_b={id_b!r}, surface_form={shared_surface_form!r}"
        )
        assert err_b is not None, (
            f"RegistrationError must be returned when second registration is rejected. "
            f"id_a={id_a!r}, id_b={id_b!r}, surface_form={shared_surface_form!r}"
        )

        # Assertion 2: a SynonymConflictRecord must have been written
        conflicts_after = _conflict_count(conn)
        assert conflicts_after > conflicts_before, (
            f"Expected at least one SynonymConflictRecord to be written, "
            f"but conflict count did not increase "
            f"(before={conflicts_before}, after={conflicts_after}). "
            f"id_a={id_a!r}, id_b={id_b!r}, surface_form={shared_surface_form!r}"
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Property 4 — via add_synonym()
# **Validates: Requirements 3.7, 5.4**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    ids=_two_distinct_taxon_ids_st,
    shared_surface_form=_surface_form_st,
    name_a=_surface_form_st,
    name_b=_surface_form_st,
)
def test_property_synonym_conflict_rejection_via_add_synonym(
    ids: tuple[str, str],
    shared_surface_form: str,
    name_a: str,
    name_b: str,
) -> None:
    """
    **Property 4: Synonym Conflict Rejection (via add_synonym)**

    **Validates: Requirements 3.7, 5.4**

    When a surface form S is added as a synonym of entity A via
    ``add_synonym()``, attempting to add the same surface form S to entity B
    must:
    1. Return ``(False, error_message)`` — the second add_synonym is rejected.
    2. Write at least one ``SynonymConflictRecord`` to the ``synonym_conflicts``
       table.
    """
    id_a, id_b = ids  # guaranteed distinct

    # Ensure the primary names are different from the shared surface form and
    # from each other so that registration of both entities succeeds.
    # We append the canonical_id to guarantee uniqueness.
    primary_a = f"{name_a} {id_a}"
    primary_b = f"{name_b} {id_b}"

    registry, conn = _make_registry()
    try:
        # Register both entities with distinct primary names
        ok_a, err_a = registry.register(
            _make_taxon_record(canonical_id=id_a, primary_name=primary_a)
        )
        assert ok_a is True, (
            f"Registration of entity A failed: {err_a}. "
            f"canonical_id={id_a!r}, primary_name={primary_a!r}"
        )

        ok_b, err_b = registry.register(
            _make_taxon_record(canonical_id=id_b, primary_name=primary_b)
        )
        assert ok_b is True, (
            f"Registration of entity B failed: {err_b}. "
            f"canonical_id={id_b!r}, primary_name={primary_b!r}"
        )

        conflicts_before = _conflict_count(conn)

        # Add shared_surface_form as a synonym of entity A — must succeed
        ok_syn_a, err_syn_a = registry.add_synonym(
            id_a, shared_surface_form, SynonymProvenance.PAPER_TEXT
        )
        assert ok_syn_a is True, (
            f"First add_synonym() should succeed. "
            f"canonical_id={id_a!r}, surface_form={shared_surface_form!r}, "
            f"error={err_syn_a}"
        )

        # Attempt to add the same surface form to entity B — must be rejected
        ok_syn_b, err_syn_b = registry.add_synonym(
            id_b, shared_surface_form, SynonymProvenance.PAPER_TEXT
        )

        # Assertion 1: second add_synonym must be rejected
        assert ok_syn_b is False, (
            f"Second add_synonym() with the same surface form should be rejected. "
            f"id_a={id_a!r}, id_b={id_b!r}, surface_form={shared_surface_form!r}"
        )
        assert err_syn_b is not None, (
            f"Error message must be returned when add_synonym() is rejected. "
            f"id_a={id_a!r}, id_b={id_b!r}, surface_form={shared_surface_form!r}"
        )

        # Assertion 2: a SynonymConflictRecord must have been written
        conflicts_after = _conflict_count(conn)
        assert conflicts_after > conflicts_before, (
            f"Expected at least one SynonymConflictRecord to be written, "
            f"but conflict count did not increase "
            f"(before={conflicts_before}, after={conflicts_after}). "
            f"id_a={id_a!r}, id_b={id_b!r}, surface_form={shared_surface_form!r}"
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Property 4 — conflict record content validation
# **Validates: Requirements 3.7, 5.4**
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    ids=_two_distinct_taxon_ids_st,
    shared_surface_form=_surface_form_st,
    name_a=_surface_form_st,
    name_b=_surface_form_st,
)
def test_property_synonym_conflict_record_content(
    ids: tuple[str, str],
    shared_surface_form: str,
    name_a: str,
    name_b: str,
) -> None:
    """
    **Property 4: Synonym Conflict Rejection — conflict record content**

    **Validates: Requirements 3.7, 5.4**

    The ``SynonymConflictRecord`` written to ``synonym_conflicts`` must contain:
    - The conflicting surface form (``duplicate_surface_form``)
    - Both entity IDs (``entity_a_id`` and ``entity_b_id``)
    - A non-null ``timestamp``

    The two entity IDs in the record must be the two distinct canonical IDs
    involved in the conflict.
    """
    id_a, id_b = ids

    primary_a = f"{name_a} {id_a}"
    primary_b = f"{name_b} {id_b}"

    registry, conn = _make_registry()
    try:
        # Register both entities
        ok_a, _ = registry.register(
            _make_taxon_record(canonical_id=id_a, primary_name=primary_a)
        )
        assert ok_a is True

        ok_b, _ = registry.register(
            _make_taxon_record(canonical_id=id_b, primary_name=primary_b)
        )
        assert ok_b is True

        # Add shared_surface_form to entity A
        ok_syn_a, _ = registry.add_synonym(
            id_a, shared_surface_form, SynonymProvenance.CURATOR
        )
        assert ok_syn_a is True

        # Attempt to add the same surface form to entity B (triggers conflict)
        ok_syn_b, _ = registry.add_synonym(
            id_b, shared_surface_form, SynonymProvenance.CURATOR
        )
        assert ok_syn_b is False

        # Fetch the conflict record(s)
        conflict_rows = conn.execute(
            "SELECT * FROM synonym_conflicts"
        ).fetchall()

        assert len(conflict_rows) >= 1, (
            "At least one conflict record must exist after a synonym conflict."
        )

        # Validate the most recent conflict record
        latest = conflict_rows[-1]

        # The duplicate_surface_form must be the shared surface form
        assert latest["duplicate_surface_form"] == shared_surface_form, (
            f"Expected duplicate_surface_form={shared_surface_form!r}, "
            f"got {latest['duplicate_surface_form']!r}"
        )

        # Both entity IDs must appear in the record (order may vary)
        recorded_ids = {latest["entity_a_id"], latest["entity_b_id"]}
        assert id_a in recorded_ids, (
            f"entity_a_id={id_a!r} not found in conflict record ids={recorded_ids!r}"
        )
        assert id_b in recorded_ids, (
            f"entity_b_id={id_b!r} not found in conflict record ids={recorded_ids!r}"
        )

        # Timestamp must be non-null
        assert latest["timestamp"] is not None, (
            "SynonymConflictRecord timestamp must not be null."
        )

    finally:
        conn.close()

"""
Property 5: Audit Completeness

**Validates: Requirements 7.1, 7.2, 15.4**

For any ``ResolutionRecord`` written to the audit store, assert a record
exists in the audit store with:
  - non-empty ``winning_strategy``
  - non-null ``timestamp``
  - matching ``paper_id``

Also tests:
  - ``write()`` returns ``True`` on success
  - ``query()`` finds the written record by ``paper_id``
  - Write failures (e.g. bad DB path) return ``False`` and do not raise
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.audit_store import ResolutionAuditStore
from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_resolution_audit_schema,
)
from entity_resolution.models import AuditQuery, CandidateScore, ResolutionRecord


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
    """Non-empty printable ASCII text."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
        min_size=1,
        max_size=max_size,
    ).map(str.strip).filter(lambda s: len(s) > 0)


def _st_candidate_score() -> st.SearchStrategy[CandidateScore]:
    """Generate an arbitrary CandidateScore."""
    return st.builds(
        CandidateScore,
        canonical_id=_st_nonempty_text(20),
        strategy=st.sampled_from(_STRATEGIES),
        grounding_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        composite_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )


def _st_resolution_record() -> st.SearchStrategy[ResolutionRecord]:
    """Generate an arbitrary ResolutionRecord with all required fields."""
    return st.builds(
        ResolutionRecord,
        record_id=st.uuids().map(str),
        surface_form=_st_nonempty_text(80),
        entity_type=st.sampled_from(_ENTITY_TYPES),
        timestamp=st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2099, 12, 31),
            timezones=st.just(timezone.utc),
        ),
        winning_strategy=st.sampled_from(_STRATEGIES),
        canonical_id=st.one_of(st.none(), _st_nonempty_text(30)),
        grounding_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        conflict_set=st.lists(_st_candidate_score(), min_size=0, max_size=5),
        paper_id=_st_nonempty_text(40),
        high_conflict=st.booleans(),
        hierarchy_match=st.booleans(),
        hierarchy_level=st.one_of(st.none(), st.integers(min_value=1, max_value=3)),
        curator_override=st.one_of(st.none(), _st_nonempty_text(20)),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_conn() -> sqlite3.Connection:
    """In-memory SQLite connection with the resolution_audit schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_resolution_audit_schema())
    yield conn
    conn.close()


@pytest.fixture
def audit_store(audit_conn: sqlite3.Connection) -> ResolutionAuditStore:
    """Fresh ResolutionAuditStore backed by an in-memory database."""
    return ResolutionAuditStore(conn=audit_conn)


# ---------------------------------------------------------------------------
# Unit tests — write() and query() basic behaviour
# ---------------------------------------------------------------------------


def _make_record(**kwargs) -> ResolutionRecord:
    """Helper to build a minimal ResolutionRecord."""
    defaults = dict(
        record_id=str(uuid.uuid4()),
        surface_form="Escherichia coli",
        entity_type="taxon",
        timestamp=datetime.now(timezone.utc),
        winning_strategy="exact",
        canonical_id="562",
        grounding_confidence=0.95,
        conflict_set=[],
        paper_id="paper-001",
        high_conflict=False,
        hierarchy_match=False,
        hierarchy_level=None,
        curator_override=None,
    )
    defaults.update(kwargs)
    return ResolutionRecord(**defaults)


def test_write_returns_true_on_success(audit_store: ResolutionAuditStore) -> None:
    """write() returns True when the record is persisted successfully."""
    record = _make_record()
    result = audit_store.write(record)
    assert result is True


def test_write_persists_record(audit_store: ResolutionAuditStore) -> None:
    """A written record can be retrieved via query()."""
    record = _make_record(paper_id="paper-xyz")
    audit_store.write(record)

    results = audit_store.query(AuditQuery(paper_id="paper-xyz"))
    assert len(results) == 1
    assert results[0].record_id == record.record_id


def test_query_returns_empty_list_on_no_match(audit_store: ResolutionAuditStore) -> None:
    """query() returns [] when no records match the filter."""
    results = audit_store.query(AuditQuery(paper_id="nonexistent-paper"))
    assert results == []


def test_query_descending_timestamp_order(audit_store: ResolutionAuditStore) -> None:
    """query() returns records in descending timestamp order."""
    t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
    t3 = datetime(2024, 12, 1, tzinfo=timezone.utc)

    for ts in [t1, t3, t2]:  # insert out of order
        audit_store.write(_make_record(
            record_id=str(uuid.uuid4()),
            timestamp=ts,
            paper_id="order-test",
        ))

    results = audit_store.query(AuditQuery(paper_id="order-test"))
    assert len(results) == 3
    timestamps = [r.timestamp for r in results]
    # Should be t3, t2, t1 (descending)
    assert timestamps[0] >= timestamps[1] >= timestamps[2]


def test_query_filter_by_surface_form(audit_store: ResolutionAuditStore) -> None:
    """query() filters correctly by surface_form."""
    audit_store.write(_make_record(surface_form="E. coli", paper_id="p1"))
    audit_store.write(_make_record(
        record_id=str(uuid.uuid4()),
        surface_form="Bacteroides fragilis",
        paper_id="p2",
    ))

    results = audit_store.query(AuditQuery(surface_form="E. coli"))
    assert len(results) == 1
    assert results[0].surface_form == "E. coli"


def test_query_filter_by_canonical_id(audit_store: ResolutionAuditStore) -> None:
    """query() filters correctly by canonical_id."""
    audit_store.write(_make_record(canonical_id="562", paper_id="p1"))
    audit_store.write(_make_record(
        record_id=str(uuid.uuid4()),
        canonical_id="1301",
        paper_id="p2",
    ))

    results = audit_store.query(AuditQuery(canonical_id="562"))
    assert len(results) == 1
    assert results[0].canonical_id == "562"


def test_query_filter_by_winning_strategy(audit_store: ResolutionAuditStore) -> None:
    """query() filters correctly by winning_strategy."""
    audit_store.write(_make_record(winning_strategy="exact", paper_id="p1"))
    audit_store.write(_make_record(
        record_id=str(uuid.uuid4()),
        winning_strategy="fuzzy",
        paper_id="p2",
    ))

    results = audit_store.query(AuditQuery(winning_strategy="exact"))
    assert len(results) == 1
    assert results[0].winning_strategy == "exact"


def test_query_filter_by_date_range(audit_store: ResolutionAuditStore) -> None:
    """query() filters correctly by date_from and date_to."""
    t_early = datetime(2023, 1, 1, tzinfo=timezone.utc)
    t_mid = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t_late = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for ts in [t_early, t_mid, t_late]:
        audit_store.write(_make_record(
            record_id=str(uuid.uuid4()),
            timestamp=ts,
            paper_id="date-test",
        ))

    results = audit_store.query(AuditQuery(
        date_from=datetime(2023, 6, 1, tzinfo=timezone.utc),
        date_to=datetime(2024, 6, 1, tzinfo=timezone.utc),
    ))
    assert len(results) == 1
    assert results[0].timestamp == t_mid


def test_query_multiple_filters_anded(audit_store: ResolutionAuditStore) -> None:
    """Multiple query filters are AND-chained."""
    audit_store.write(_make_record(
        surface_form="E. coli",
        canonical_id="562",
        paper_id="p1",
    ))
    audit_store.write(_make_record(
        record_id=str(uuid.uuid4()),
        surface_form="E. coli",
        canonical_id="1301",
        paper_id="p2",
    ))

    # Both have surface_form="E. coli" but only one has canonical_id="562"
    results = audit_store.query(AuditQuery(surface_form="E. coli", canonical_id="562"))
    assert len(results) == 1
    assert results[0].canonical_id == "562"


def test_query_limit_respected(audit_store: ResolutionAuditStore) -> None:
    """query() respects the limit parameter."""
    for i in range(10):
        audit_store.write(_make_record(
            record_id=str(uuid.uuid4()),
            paper_id="limit-test",
        ))

    results = audit_store.query(AuditQuery(paper_id="limit-test"), limit=3)
    assert len(results) == 3


def test_write_with_conflict_set(audit_store: ResolutionAuditStore) -> None:
    """write() correctly serialises and query() deserialises conflict_set."""
    conflict_set = [
        CandidateScore(
            canonical_id="562",
            strategy="exact",
            grounding_confidence=1.0,
            composite_score=0.95,
        ),
        CandidateScore(
            canonical_id="1301",
            strategy="fuzzy",
            grounding_confidence=0.8,
            composite_score=0.48,
        ),
    ]
    record = _make_record(conflict_set=conflict_set, paper_id="conflict-test")
    audit_store.write(record)

    results = audit_store.query(AuditQuery(paper_id="conflict-test"))
    assert len(results) == 1
    assert len(results[0].conflict_set) == 2
    assert results[0].conflict_set[0].canonical_id == "562"
    assert results[0].conflict_set[1].canonical_id == "1301"


def test_write_failure_returns_false_and_does_not_raise() -> None:
    """
    write() returns False and does not raise when the underlying connection
    has been closed (simulating a DB failure at write time).

    Requirements: 7.5
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_resolution_audit_schema())
    store = ResolutionAuditStore(conn=conn)

    # Close the connection to force a write failure
    conn.close()

    record = _make_record()
    # Must not raise — must return False
    result = store.write(record)
    assert result is False


# ---------------------------------------------------------------------------
# Property 5: Audit Completeness
# **Validates: Requirements 7.1, 7.2, 15.4**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(record=_st_resolution_record())
def test_property_audit_completeness(record: ResolutionRecord) -> None:
    """
    **Property 5: Audit Completeness**

    **Validates: Requirements 7.1, 7.2, 15.4**

    For any ResolutionRecord written to the audit store:
    - write() returns True
    - query() by paper_id finds at least one record
    - The found record has a non-empty winning_strategy
    - The found record has a non-null timestamp
    - The found record has a matching paper_id
    """
    # Use a fresh in-memory store for each example to avoid cross-contamination
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_resolution_audit_schema())
    store = ResolutionAuditStore(conn=conn)

    try:
        # write() must return True
        success = store.write(record)
        assert success is True, "write() should return True on success"

        # query() by paper_id must find the record
        results = store.query(AuditQuery(paper_id=record.paper_id))
        assert len(results) >= 1, (
            f"query(paper_id={record.paper_id!r}) returned no results after write"
        )

        # Find our specific record (there may be multiple if paper_id collides,
        # but since we use a fresh DB per example, there will be exactly one)
        found = next(
            (r for r in results if r.record_id == record.record_id), None
        )
        assert found is not None, (
            f"Written record {record.record_id!r} not found in query results"
        )

        # Non-empty winning_strategy
        assert found.winning_strategy, (
            "winning_strategy must be non-empty"
        )

        # Non-null timestamp
        assert found.timestamp is not None, "timestamp must be non-null"

        # Matching paper_id
        assert found.paper_id == record.paper_id, (
            f"paper_id mismatch: expected {record.paper_id!r}, got {found.paper_id!r}"
        )

    finally:
        conn.close()


@settings(max_examples=100)
@given(record=_st_resolution_record())
def test_property_write_failure_does_not_raise(record: ResolutionRecord) -> None:
    """
    **Validates: Requirements 7.5**

    write() on a broken store (closed connection) returns False and never raises.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_resolution_audit_schema())
    store = ResolutionAuditStore(conn=conn)

    # Close the connection to simulate a write failure
    conn.close()

    # Must not raise
    result = store.write(record)
    assert result is False, "write() must return False when the DB connection is closed"

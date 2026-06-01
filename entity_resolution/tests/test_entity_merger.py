"""
Tests for EntityMerger — atomic Neo4j merge, relationship deduplication, rollback.

Uses a mock Neo4j driver so no live Neo4j instance is required.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest

from entity_resolution.entity_merger import EntityMerger
from entity_resolution.models import MergeLogEntry, MergeRollbackEntry


# ---------------------------------------------------------------------------
# Mock Neo4j infrastructure
# ---------------------------------------------------------------------------


class _MockRecord:
    """Minimal mock for a Neo4j Record."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


class _MockResult:
    """Minimal mock for a Neo4j Result (iterable of records)."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = [_MockRecord(r) for r in records]
        self._index = 0

    def single(self) -> Optional[_MockRecord]:
        return self._records[0] if self._records else None

    def __iter__(self):
        return iter(self._records)


class _MockTransaction:
    """
    Mock Neo4j transaction that records all Cypher calls and can be
    configured to return specific results per query pattern.
    """

    def __init__(self) -> None:
        self.queries: List[Dict[str, Any]] = []
        self._responses: List[_MockResult] = []
        self.committed = False
        self.rolled_back = False
        self._response_index = 0

    def add_response(self, records: List[Dict[str, Any]]) -> None:
        """Queue a response to be returned by the next run() call."""
        self._responses.append(_MockResult(records))

    def run(self, query: str, **params) -> _MockResult:
        self.queries.append({"query": query, "params": params})
        if self._response_index < len(self._responses):
            result = self._responses[self._response_index]
            self._response_index += 1
            return result
        return _MockResult([])

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _MockSession:
    """Mock Neo4j session that returns a pre-configured transaction."""

    def __init__(self, tx: _MockTransaction) -> None:
        self._tx = tx

    def begin_transaction(self) -> _MockTransaction:
        return self._tx

    def execute_write(self, func, *args, **kwargs):
        """Simulate execute_write by calling func with a fresh mock transaction."""
        write_tx = _MockTransaction()
        # For ensure_canonical_node, return a node_id
        write_tx.add_response([{"node_id": "element-id-canonical"}])
        return func(write_tx, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _MockDriver:
    """Mock Neo4j driver that returns a pre-configured session."""

    def __init__(self, session: _MockSession) -> None:
        self._session = session

    def session(self):
        return self._session

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers to build an EntityMerger with a mock driver
# ---------------------------------------------------------------------------


def _make_merger(tx: _MockTransaction) -> EntityMerger:
    """
    Create an EntityMerger whose Neo4j driver is replaced with a mock
    that uses the given transaction.
    """
    session = _MockSession(tx)
    driver = _MockDriver(session)

    with patch("entity_resolution.entity_merger.GraphDatabase") as mock_gdb:
        mock_gdb.driver.return_value = driver
        merger = EntityMerger(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
    # Replace the internal driver directly so session() calls work
    merger._driver = driver
    return merger


# ---------------------------------------------------------------------------
# Tests for ensure_canonical_node
# ---------------------------------------------------------------------------


class TestEnsureCanonicalNode:
    """Tests for EntityMerger.ensure_canonical_node()."""

    def test_returns_node_id(self) -> None:
        """ensure_canonical_node() returns the Neo4j element ID from MERGE."""
        tx = _MockTransaction()
        merger = _make_merger(tx)

        node_id = merger.ensure_canonical_node(
            canonical_id="562",
            entity_type="taxon",
            primary_name="Escherichia coli",
        )
        assert node_id == "element-id-canonical"

    def test_idempotent_merge(self) -> None:
        """
        Calling ensure_canonical_node() twice with the same canonical_id
        should both succeed (MERGE is idempotent).
        """
        tx = _MockTransaction()
        merger = _make_merger(tx)

        id1 = merger.ensure_canonical_node("562", "taxon", "Escherichia coli")
        id2 = merger.ensure_canonical_node("562", "taxon", "Escherichia coli")
        assert id1 == id2 == "element-id-canonical"


# ---------------------------------------------------------------------------
# Tests for merge() — happy path
# ---------------------------------------------------------------------------


class TestMergeHappyPath:
    """Tests for EntityMerger.merge() when everything succeeds."""

    def _build_tx_for_successful_merge(
        self,
        source_type: str = "taxon",
        target_type: str = "taxon",
        source_rels: Optional[List[Dict]] = None,
        target_rels: Optional[List[Dict]] = None,
    ) -> _MockTransaction:
        """
        Build a mock transaction pre-loaded with responses for a successful merge.

        Query order in merge():
          1. _get_entity_types: source node type  (1 query)
          2. _get_entity_types: target node type  (1 query)
          3. _get_all_relationships: single UNION ALL query for source rels (1 query)
          4. _transfer_relationships: single UNION ALL query for target rels (1 query)
          5+. _delete_relationship_by_id / _create_relationship calls (empty results)
          N. _delete_node (empty result)
        """
        tx = _MockTransaction()

        # Step 1: source entity_type
        tx.add_response([{"entity_type": source_type}])
        # Step 2: target entity_type
        tx.add_response([{"entity_type": target_type}])

        # Step 3: all rels of source (single UNION ALL query)
        all_src_rels = source_rels or []
        tx.add_response(all_src_rels)

        # Step 4: all rels of target (single UNION ALL query, for deduplication)
        all_tgt_rels = target_rels or []
        tx.add_response(all_tgt_rels)

        # Any _create_relationship, _delete_relationship, _delete_node calls
        for _ in range(20):
            tx.add_response([])

        return tx

    def test_successful_merge_returns_true(self) -> None:
        """merge() returns (True, None) on success."""
        tx = self._build_tx_for_successful_merge()
        merger = _make_merger(tx)

        success, error = merger.merge(
            source_node_id="src-element-id",
            target_canonical_id="562",
            triggering_surface_form="E. coli",
        )

        assert success is True
        assert error is None

    def test_successful_merge_commits_transaction(self) -> None:
        """merge() commits the transaction on success."""
        tx = self._build_tx_for_successful_merge()
        merger = _make_merger(tx)

        merger.merge("src-element-id", "562", "E. coli")

        assert tx.committed is True
        assert tx.rolled_back is False

    def test_successful_merge_writes_log_entry(self) -> None:
        """merge() appends a MergeLogEntry to merge_log on success."""
        tx = self._build_tx_for_successful_merge()
        merger = _make_merger(tx)

        merger.merge("src-element-id", "562", "E. coli")

        assert len(merger.merge_log) == 1
        entry = merger.merge_log[0]
        assert isinstance(entry, MergeLogEntry)
        assert entry.source_node_ids == ["src-element-id"]
        assert entry.target_canonical_id == "562"
        assert entry.triggering_resolution == "E. coli"
        assert isinstance(entry.timestamp, datetime)

    def test_merge_log_entry_has_correct_counts_no_rels(self) -> None:
        """MergeLogEntry has 0 transferred and 0 deduplicated when no rels exist."""
        tx = self._build_tx_for_successful_merge()
        merger = _make_merger(tx)

        merger.merge("src-element-id", "562", "E. coli")

        entry = merger.merge_log[0]
        assert entry.relationships_transferred == 0
        assert entry.relationships_deduplicated == 0

    def test_merge_transfers_outbound_relationship(self) -> None:
        """
        When source has an outbound relationship not present on target,
        it is transferred (transferred_count=1, deduplicated_count=0).
        """
        source_rels = [
            {
                "rel_id": "rel-1",
                "rel_type": "ASSOCIATED_WITH",
                "direction": "outbound",
                "counterpart_id": "other-node-id",
                "confidence": 0.8,
            }
        ]
        tx = self._build_tx_for_successful_merge(source_rels=source_rels)
        merger = _make_merger(tx)

        success, error = merger.merge("src-element-id", "562", "E. coli")

        assert success is True
        entry = merger.merge_log[0]
        assert entry.relationships_transferred == 1
        assert entry.relationships_deduplicated == 0

    def test_merge_deduplicates_relationship_keeps_higher_confidence(self) -> None:
        """
        When source and target both have the same (type, counterpart, direction),
        the higher-confidence one is kept and deduplicated_count is incremented.
        """
        source_rels = [
            {
                "rel_id": "rel-src",
                "rel_type": "ASSOCIATED_WITH",
                "direction": "outbound",
                "counterpart_id": "other-node-id",
                "confidence": 0.9,  # higher than target
            }
        ]
        target_rels = [
            {
                "rel_id": "rel-tgt",
                "rel_type": "ASSOCIATED_WITH",
                "direction": "outbound",
                "counterpart_id": "other-node-id",
                "confidence": 0.5,  # lower
            }
        ]
        tx = self._build_tx_for_successful_merge(
            source_rels=source_rels, target_rels=target_rels
        )
        merger = _make_merger(tx)

        success, error = merger.merge("src-element-id", "562", "E. coli")

        assert success is True
        entry = merger.merge_log[0]
        assert entry.relationships_transferred == 0  # not a new rel, it's a dedup
        assert entry.relationships_deduplicated == 1

    def test_merge_deduplicates_keeps_existing_when_lower_confidence(self) -> None:
        """
        When target has higher confidence than source for the same rel,
        the existing target relationship is kept (source discarded).
        deduplicated_count is still incremented.
        """
        source_rels = [
            {
                "rel_id": "rel-src",
                "rel_type": "ASSOCIATED_WITH",
                "direction": "outbound",
                "counterpart_id": "other-node-id",
                "confidence": 0.3,  # lower than target
            }
        ]
        target_rels = [
            {
                "rel_id": "rel-tgt",
                "rel_type": "ASSOCIATED_WITH",
                "direction": "outbound",
                "counterpart_id": "other-node-id",
                "confidence": 0.9,  # higher
            }
        ]
        tx = self._build_tx_for_successful_merge(
            source_rels=source_rels, target_rels=target_rels
        )
        merger = _make_merger(tx)

        success, error = merger.merge("src-element-id", "562", "E. coli")

        assert success is True
        entry = merger.merge_log[0]
        assert entry.relationships_deduplicated == 1


# ---------------------------------------------------------------------------
# Tests for merge() — type conflict (Requirement 6.5)
# ---------------------------------------------------------------------------


class TestMergeTypeConflict:
    """Tests for EntityMerger.merge() when entity types differ."""

    def _build_tx_type_conflict(
        self, source_type: str = "taxon", target_type: str = "disease"
    ) -> _MockTransaction:
        tx = _MockTransaction()
        tx.add_response([{"entity_type": source_type}])
        tx.add_response([{"entity_type": target_type}])
        return tx

    def test_type_conflict_returns_false(self) -> None:
        """merge() returns (False, error_msg) when entity types differ."""
        tx = self._build_tx_type_conflict("taxon", "disease")
        merger = _make_merger(tx)

        success, error = merger.merge("src-id", "D006262", "gut bacteria")

        assert success is False
        assert error is not None
        assert "taxon" in error
        assert "disease" in error

    def test_type_conflict_writes_rollback_entry(self) -> None:
        """merge() writes a MergeRollbackEntry on type conflict."""
        tx = self._build_tx_type_conflict("taxon", "disease")
        merger = _make_merger(tx)

        merger.merge("src-id", "D006262", "gut bacteria")

        assert len(merger.rollback_log) == 1
        entry = merger.rollback_log[0]
        assert isinstance(entry, MergeRollbackEntry)
        assert entry.source_node_ids == ["src-id"]
        assert entry.target_canonical_id == "D006262"
        assert entry.failed_step == "type_check"
        assert "taxon" in entry.error_message
        assert "disease" in entry.error_message

    def test_type_conflict_does_not_write_merge_log(self) -> None:
        """merge() does NOT write a MergeLogEntry on type conflict."""
        tx = self._build_tx_type_conflict("taxon", "disease")
        merger = _make_merger(tx)

        merger.merge("src-id", "D006262", "gut bacteria")

        assert len(merger.merge_log) == 0

    def test_type_conflict_rolls_back_transaction(self) -> None:
        """merge() rolls back the transaction on type conflict."""
        tx = self._build_tx_type_conflict("taxon", "disease")
        merger = _make_merger(tx)

        merger.merge("src-id", "D006262", "gut bacteria")

        assert tx.rolled_back is True
        assert tx.committed is False

    def test_same_type_does_not_conflict(self) -> None:
        """merge() proceeds normally when both nodes have the same entity_type."""
        tx = _MockTransaction()
        # Both taxon
        tx.add_response([{"entity_type": "taxon"}])
        tx.add_response([{"entity_type": "taxon"}])
        # No source rels (single UNION ALL query)
        tx.add_response([])
        # No target rels (single UNION ALL query)
        tx.add_response([])
        # _delete_node
        tx.add_response([])

        merger = _make_merger(tx)
        success, error = merger.merge("src-id", "562", "E. coli")

        assert success is True
        assert error is None


# ---------------------------------------------------------------------------
# Tests for merge() — rollback on exception (Requirement 6.6, 6.7)
# ---------------------------------------------------------------------------


class TestMergeRollback:
    """Tests for EntityMerger.merge() rollback behaviour on unexpected errors."""

    def test_exception_during_get_entity_types_triggers_rollback(self) -> None:
        """
        If _get_entity_types raises (e.g., source node not found),
        merge() rolls back and writes a MergeRollbackEntry.
        """
        tx = _MockTransaction()
        # Source node not found — single() returns None
        tx.add_response([])  # empty result for source type query

        merger = _make_merger(tx)
        success, error = merger.merge("nonexistent-id", "562", "E. coli")

        assert success is False
        assert error is not None
        assert len(merger.rollback_log) == 1
        assert tx.rolled_back is True

    def test_rollback_entry_contains_correct_fields(self) -> None:
        """MergeRollbackEntry has correct source_node_ids and target_canonical_id."""
        tx = _MockTransaction()
        tx.add_response([])  # source not found

        merger = _make_merger(tx)
        merger.merge("src-node-id", "target-canonical", "surface form")

        entry = merger.rollback_log[0]
        assert entry.source_node_ids == ["src-node-id"]
        assert entry.target_canonical_id == "target-canonical"
        assert isinstance(entry.timestamp, datetime)
        assert entry.error_message  # non-empty

    def test_rollback_does_not_write_merge_log(self) -> None:
        """On rollback, no MergeLogEntry is written."""
        tx = _MockTransaction()
        tx.add_response([])  # source not found

        merger = _make_merger(tx)
        merger.merge("src-id", "562", "E. coli")

        assert len(merger.merge_log) == 0

    def test_multiple_merges_accumulate_log_entries(self) -> None:
        """Multiple successful merges accumulate separate MergeLogEntry records."""

        def _make_success_tx() -> _MockTransaction:
            tx = _MockTransaction()
            tx.add_response([{"entity_type": "taxon"}])
            tx.add_response([{"entity_type": "taxon"}])
            tx.add_response([])
            tx.add_response([])
            tx.add_response([])
            tx.add_response([])
            return tx

        # We need separate sessions for each merge call
        # Patch the driver to return fresh sessions each time
        merger = EntityMerger.__new__(EntityMerger)
        merger.merge_log = []
        merger.rollback_log = []

        txs = [_make_success_tx(), _make_success_tx()]
        call_count = [0]

        class _MultiSessionDriver:
            def session(self):
                idx = call_count[0]
                call_count[0] += 1
                return _MockSession(txs[idx] if idx < len(txs) else _make_success_tx())

            def close(self):
                pass

        merger._driver = _MultiSessionDriver()

        merger.merge("src-1", "562", "E. coli")
        merger.merge("src-2", "562", "Escherichia coli")

        assert len(merger.merge_log) == 2
        assert merger.merge_log[0].source_node_ids == ["src-1"]
        assert merger.merge_log[1].source_node_ids == ["src-2"]


# ---------------------------------------------------------------------------
# Tests for audit log content (Requirement 6.4, 6.7)
# ---------------------------------------------------------------------------


class TestAuditLog:
    """Tests verifying MergeLogEntry and MergeRollbackEntry content."""

    def test_merge_log_entry_timestamp_is_utc(self) -> None:
        """MergeLogEntry.timestamp is timezone-aware (UTC)."""
        tx = _MockTransaction()
        tx.add_response([{"entity_type": "method"}])
        tx.add_response([{"entity_type": "method"}])
        tx.add_response([])  # source rels
        tx.add_response([])  # target rels
        tx.add_response([])  # delete node

        merger = _make_merger(tx)
        merger.merge("src-id", "METHOD-16S", "16S rRNA")

        entry = merger.merge_log[0]
        assert entry.timestamp.tzinfo is not None

    def test_rollback_log_entry_timestamp_is_utc(self) -> None:
        """MergeRollbackEntry.timestamp is timezone-aware (UTC)."""
        tx = _MockTransaction()
        tx.add_response([])  # source not found

        merger = _make_merger(tx)
        merger.merge("src-id", "562", "E. coli")

        entry = merger.rollback_log[0]
        assert entry.timestamp.tzinfo is not None

    def test_merge_log_entry_triggering_resolution(self) -> None:
        """MergeLogEntry.triggering_resolution matches the surface form passed to merge()."""
        tx = _MockTransaction()
        tx.add_response([{"entity_type": "disease"}])
        tx.add_response([{"entity_type": "disease"}])
        tx.add_response([])  # source rels
        tx.add_response([])  # target rels
        tx.add_response([])  # delete node

        merger = _make_merger(tx)
        merger.merge("src-id", "D006262", "inflammatory bowel disease")

        entry = merger.merge_log[0]
        assert entry.triggering_resolution == "inflammatory bowel disease"

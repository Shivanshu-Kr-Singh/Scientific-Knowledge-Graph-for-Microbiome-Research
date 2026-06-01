"""
ResolutionAuditStore — physically separate SQLite audit log for resolution records.

Backed by ``resolution_audit.db`` (a distinct database file from
``canonical_registry.db`` and ``resolution_cache.db``).

Requirements: 7.1, 7.2, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import List, Optional

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_resolution_audit_schema,
)
from entity_resolution.models import AuditQuery, CandidateScore, ResolutionRecord

logger = logging.getLogger(__name__)


class ResolutionAuditStore:
    """
    Physically separate SQLite store for resolution audit records.

    Physical separation: stored in a distinct SQLite database file
    (``resolution_audit.db``) that is not co-located with
    ``canonical_registry.db`` or ``resolution_cache.db``.

    Write-failure tolerance: if ``write()`` fails for any reason, the error
    is logged to ``logging.error`` and ``False`` is returned.  The pipeline
    is never blocked and no exception is ever raised from ``write()``.

    Requirements: 7.1, 7.2, 7.4, 7.5, 7.6
    """

    def __init__(
        self,
        db_path: str = "resolution_audit.db",
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """
        Initialise the audit store.

        Args:
            db_path: Path to the ``resolution_audit.db`` SQLite file.
                     Ignored when *conn* is provided (useful for tests).
            conn:    An already-open :class:`sqlite3.Connection`.  When
                     supplied the store uses this connection directly and
                     does **not** open *db_path*.  The caller is responsible
                     for closing the connection.
        """
        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._owns_conn = True
            create_schema_in_connection(self._conn, get_resolution_audit_schema())

    # ------------------------------------------------------------------
    # write()
    # ------------------------------------------------------------------

    def write(self, record: ResolutionRecord) -> bool:
        """
        Persist a :class:`ResolutionRecord` to the audit store.

        Serialises ``conflict_set`` as a JSON array before inserting.
        Catches **all** exceptions, logs them via ``logging.error``, and
        returns ``False`` on failure — never raises.

        Args:
            record: The resolution record to persist.

        Returns:
            ``True`` on success, ``False`` on any failure.

        Requirements: 7.1, 7.2, 7.5
        """
        try:
            conflict_set_json = json.dumps(
                [cs.model_dump() for cs in record.conflict_set]
            )

            self._conn.execute(
                """
                INSERT INTO resolution_records (
                    record_id,
                    surface_form,
                    entity_type,
                    timestamp,
                    winning_strategy,
                    canonical_id,
                    grounding_confidence,
                    conflict_set_json,
                    paper_id,
                    high_conflict,
                    hierarchy_match,
                    hierarchy_level,
                    curator_override
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.surface_form,
                    record.entity_type,
                    record.timestamp.isoformat(),
                    record.winning_strategy,
                    record.canonical_id,
                    record.grounding_confidence,
                    conflict_set_json,
                    record.paper_id,
                    int(record.high_conflict),
                    int(record.hierarchy_match),
                    record.hierarchy_level,
                    record.curator_override,
                ),
            )
            self._conn.commit()
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ResolutionAuditStore.write() failed for surface_form=%r paper_id=%r: %s",
                record.surface_form,
                record.paper_id,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # query()
    # ------------------------------------------------------------------

    def query(
        self,
        query: AuditQuery,
        limit: int = 1000,
    ) -> List[ResolutionRecord]:
        """
        Query resolution records with AND-chained filters.

        Supported filter fields (all optional):
        - ``surface_form``    — exact match
        - ``canonical_id``    — exact match
        - ``winning_strategy``— exact match
        - ``date_from``       — timestamp >= date_from (inclusive)
        - ``date_to``         — timestamp <= date_to   (inclusive)
        - ``paper_id``        — exact match

        Results are returned in **descending timestamp order**.
        Returns ``[]`` on no match or on any error.

        Args:
            query: :class:`AuditQuery` instance with optional filter fields.
            limit: Maximum number of records to return (default 1 000).

        Returns:
            A list of :class:`ResolutionRecord` objects, newest first.

        Requirements: 7.6
        """
        try:
            conditions: List[str] = []
            params: List[object] = []

            if query.surface_form is not None:
                conditions.append("surface_form = ?")
                params.append(query.surface_form)

            if query.canonical_id is not None:
                conditions.append("canonical_id = ?")
                params.append(query.canonical_id)

            if query.winning_strategy is not None:
                conditions.append("winning_strategy = ?")
                params.append(query.winning_strategy)

            if query.date_from is not None:
                conditions.append("timestamp >= ?")
                params.append(query.date_from.isoformat())

            if query.date_to is not None:
                conditions.append("timestamp <= ?")
                params.append(query.date_to.isoformat())

            if query.paper_id is not None:
                conditions.append("paper_id = ?")
                params.append(query.paper_id)

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            sql = (
                f"SELECT * FROM resolution_records "
                f"{where_clause} "
                f"ORDER BY timestamp DESC "
                f"LIMIT ?"
            )
            params.append(limit)

            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

            records: List[ResolutionRecord] = []
            for row in rows:
                records.append(self._row_to_record(row))
            return records

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ResolutionAuditStore.query() failed: %s",
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ResolutionRecord:
        """Convert a SQLite row to a :class:`ResolutionRecord`."""
        conflict_set_raw = json.loads(row["conflict_set_json"])
        conflict_set = [CandidateScore(**cs) for cs in conflict_set_raw]

        return ResolutionRecord(
            record_id=row["record_id"],
            surface_form=row["surface_form"],
            entity_type=row["entity_type"],
            timestamp=row["timestamp"],
            winning_strategy=row["winning_strategy"],
            canonical_id=row["canonical_id"],
            grounding_confidence=row["grounding_confidence"],
            conflict_set=conflict_set,
            paper_id=row["paper_id"],
            high_conflict=bool(row["high_conflict"]),
            hierarchy_match=bool(row["hierarchy_match"]),
            hierarchy_level=row["hierarchy_level"],
            curator_override=row["curator_override"],
        )

    def close(self) -> None:
        """Close the underlying database connection if owned by this instance."""
        if self._owns_conn:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

"""
ResolutionMetrics — per-run and per-entity-type resolution quality tracking.

Tracks resolution rates, confidence averages, and unresolved counts.
Persists snapshots to the ``metrics_snapshots`` table in ``resolution_audit.db``.
Emits warnings when resolution rate < 70% for any entity type.
Detects 5-point degradation across historical snapshots.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_resolution_audit_schema,
)
from entity_resolution.models import EntityTypeMetrics, ResolutionResult, RunMetricsSnapshot

logger = logging.getLogger(__name__)

# Warning threshold for resolution rate (Requirement 10.5)
_RESOLUTION_RATE_WARNING_THRESHOLD = 0.70

# Degradation threshold in percentage points (Requirement 10.4)
_DEGRADATION_THRESHOLD_POINTS = 5.0


class _EntityTypeAccumulator:
    """In-memory accumulator for a single entity type within a run."""

    __slots__ = (
        "total_submitted",
        "resolved_count",
        "unresolved_count",
        "confidence_sum",
        "per_strategy_counts",
    )

    def __init__(self) -> None:
        self.total_submitted: int = 0
        self.resolved_count: int = 0
        self.unresolved_count: int = 0
        self.confidence_sum: float = 0.0
        self.per_strategy_counts: Dict[str, int] = defaultdict(int)

    def record(self, result: ResolutionResult) -> None:
        self.total_submitted += 1
        if result.grounded:
            self.resolved_count += 1
            self.confidence_sum += result.grounding_confidence
        else:
            self.unresolved_count += 1
        self.per_strategy_counts[result.winning_strategy] += 1

    def to_entity_type_metrics(self, entity_type: str) -> EntityTypeMetrics:
        resolution_rate = (
            self.resolved_count / self.total_submitted
            if self.total_submitted > 0
            else 0.0
        )
        avg_confidence = (
            self.confidence_sum / self.resolved_count
            if self.resolved_count > 0
            else 0.0
        )
        return EntityTypeMetrics(
            entity_type=entity_type,
            resolution_rate=resolution_rate,
            avg_grounding_confidence=avg_confidence,
            unresolved_count=self.unresolved_count,
        )


class ResolutionMetrics:
    """
    Per-run and per-entity-type resolution quality metrics.

    Usage pattern::

        metrics = ResolutionMetrics(db_path="resolution_audit.db")
        for result in run_results:
            metrics.record_resolution(result)
        snapshot = metrics.finalize_run(run_id="run-001", paper_ids=["p1", "p2"])

    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
    """

    def __init__(
        self,
        db_path: str = "resolution_audit.db",
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """
        Initialise the metrics tracker.

        Args:
            db_path: Path to ``resolution_audit.db``.  Ignored when *conn*
                     is provided (useful for in-memory test databases).
            conn:    An already-open :class:`sqlite3.Connection`.  When
                     supplied the instance uses this connection directly and
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

        # In-memory accumulators keyed by entity_type
        self._accumulators: Dict[str, _EntityTypeAccumulator] = defaultdict(
            _EntityTypeAccumulator
        )
        # Global per-strategy counts across all entity types
        self._global_strategy_counts: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # record_resolution()
    # ------------------------------------------------------------------

    def record_resolution(self, result: ResolutionResult) -> None:
        """
        Accumulate metrics for a single resolution result.

        Updates per-entity-type counters and confidence sums in memory.
        This method is non-blocking and never raises.

        Args:
            result: A :class:`ResolutionResult` from the pipeline.

        Requirements: 10.1, 10.2
        """
        entity_type = result.entity_type
        self._accumulators[entity_type].record(result)
        self._global_strategy_counts[result.winning_strategy] += 1

    # ------------------------------------------------------------------
    # finalize_run()
    # ------------------------------------------------------------------

    def finalize_run(
        self,
        run_id: str,
        paper_ids: List[str],
    ) -> RunMetricsSnapshot:
        """
        Compute a :class:`RunMetricsSnapshot` and persist it to SQLite.

        Persistence failures are logged (not raised) so the pipeline run
        always completes normally.  A ``logging.warning`` is emitted for
        every entity type whose ``resolution_rate < 0.70``.

        Args:
            run_id:    Unique identifier for this pipeline run.
            paper_ids: List of paper IDs processed in this run.

        Returns:
            The computed :class:`RunMetricsSnapshot`.

        Requirements: 10.1, 10.3, 10.5
        """
        timestamp = datetime.now(timezone.utc)

        # Aggregate totals across all entity types
        total_forms = sum(
            acc.total_submitted for acc in self._accumulators.values()
        )
        resolved_count = sum(
            acc.resolved_count for acc in self._accumulators.values()
        )
        unresolved_count = sum(
            acc.unresolved_count for acc in self._accumulators.values()
        )
        overall_rate = resolved_count / total_forms if total_forms > 0 else 0.0

        # Build per-entity-type metrics
        entity_type_metrics: List[EntityTypeMetrics] = []
        for entity_type, acc in self._accumulators.items():
            etm = acc.to_entity_type_metrics(entity_type)
            entity_type_metrics.append(etm)

            # Requirement 10.5: warn if any entity type rate < 0.70
            if acc.total_submitted > 0 and etm.resolution_rate < _RESOLUTION_RATE_WARNING_THRESHOLD:
                logger.warning(
                    "Resolution rate below threshold: run_id=%r entity_type=%r "
                    "observed_rate=%.4f threshold=%.2f",
                    run_id,
                    entity_type,
                    etm.resolution_rate,
                    _RESOLUTION_RATE_WARNING_THRESHOLD,
                )

        snapshot = RunMetricsSnapshot(
            run_id=run_id,
            timestamp=timestamp,
            paper_ids=paper_ids,
            total_forms=total_forms,
            resolved_count=resolved_count,
            unresolved_count=unresolved_count,
            resolution_rate=overall_rate,
            per_strategy_counts=dict(self._global_strategy_counts),
            entity_type_metrics=entity_type_metrics,
        )

        # Persist to SQLite — failure is logged, not raised (Requirement 10.3)
        self._persist_snapshot(snapshot)

        return snapshot

    # ------------------------------------------------------------------
    # query_snapshots()
    # ------------------------------------------------------------------

    def query_snapshots(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> List[RunMetricsSnapshot]:
        """
        Query historical metric snapshots in ascending timestamp order.

        Flags entity types where the most recent snapshot's resolution rate
        is more than 5 percentage points below the historical average across
        all snapshots in the queried range (Requirement 10.4).

        The degradation flag is communicated by appending a synthetic
        ``EntityTypeMetrics`` entry with ``entity_type`` prefixed by
        ``"DEGRADED:"`` to the most recent snapshot's ``entity_type_metrics``
        list.  This preserves the :class:`RunMetricsSnapshot` model without
        requiring schema changes.

        Args:
            date_from: Start of the date range (inclusive).
            date_to:   End of the date range (inclusive).

        Returns:
            List of :class:`RunMetricsSnapshot` in ascending timestamp order.

        Requirements: 10.4
        """
        try:
            cursor = self._conn.execute(
                """
                SELECT run_id, timestamp, paper_ids_json,
                       total_forms, resolved_count, unresolved_count,
                       resolution_rate, per_strategy_counts_json,
                       entity_type_metrics_json
                FROM metrics_snapshots
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (date_from.isoformat(), date_to.isoformat()),
            )
            rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.error("ResolutionMetrics.query_snapshots() failed: %s", exc)
            return []

        if not rows:
            return []

        snapshots: List[RunMetricsSnapshot] = []
        for row in rows:
            snapshots.append(self._row_to_snapshot(row))

        # Degradation detection (Requirement 10.4)
        # For each entity type, compute the historical average rate across
        # all snapshots except the most recent, then compare to the most
        # recent snapshot's rate.
        if len(snapshots) >= 2:
            most_recent = snapshots[-1]
            prior_snapshots = snapshots[:-1]

            # Collect rates per entity type from prior snapshots
            prior_rates: Dict[str, List[float]] = defaultdict(list)
            for snap in prior_snapshots:
                for etm in snap.entity_type_metrics:
                    prior_rates[etm.entity_type].append(etm.resolution_rate)

            # Check most recent snapshot for degradation
            degraded_types: List[str] = []
            for etm in most_recent.entity_type_metrics:
                rates = prior_rates.get(etm.entity_type)
                if not rates:
                    continue
                historical_avg = sum(rates) / len(rates)
                drop = (historical_avg - etm.resolution_rate) * 100.0
                if drop > _DEGRADATION_THRESHOLD_POINTS:
                    degraded_types.append(etm.entity_type)

            if degraded_types:
                # Annotate the most recent snapshot with degradation markers
                degradation_markers = [
                    EntityTypeMetrics(
                        entity_type=f"DEGRADED:{dt}",
                        resolution_rate=0.0,
                        avg_grounding_confidence=0.0,
                        unresolved_count=0,
                    )
                    for dt in degraded_types
                ]
                # Replace the last snapshot with an annotated version
                annotated = RunMetricsSnapshot(
                    run_id=most_recent.run_id,
                    timestamp=most_recent.timestamp,
                    paper_ids=most_recent.paper_ids,
                    total_forms=most_recent.total_forms,
                    resolved_count=most_recent.resolved_count,
                    unresolved_count=most_recent.unresolved_count,
                    resolution_rate=most_recent.resolution_rate,
                    per_strategy_counts=most_recent.per_strategy_counts,
                    entity_type_metrics=most_recent.entity_type_metrics + degradation_markers,
                )
                snapshots[-1] = annotated

        return snapshots

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_snapshot(self, snapshot: RunMetricsSnapshot) -> bool:
        """
        Persist a snapshot to the ``metrics_snapshots`` table.

        Returns ``True`` on success, ``False`` on failure (failure is logged).

        Requirements: 10.3
        """
        try:
            paper_ids_json = json.dumps(snapshot.paper_ids)
            per_strategy_json = json.dumps(snapshot.per_strategy_counts)
            entity_type_metrics_json = json.dumps(
                [etm.model_dump() for etm in snapshot.entity_type_metrics]
            )

            self._conn.execute(
                """
                INSERT OR REPLACE INTO metrics_snapshots (
                    run_id,
                    timestamp,
                    paper_ids_json,
                    total_forms,
                    resolved_count,
                    unresolved_count,
                    resolution_rate,
                    per_strategy_counts_json,
                    entity_type_metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.run_id,
                    snapshot.timestamp.isoformat(),
                    paper_ids_json,
                    snapshot.total_forms,
                    snapshot.resolved_count,
                    snapshot.unresolved_count,
                    snapshot.resolution_rate,
                    per_strategy_json,
                    entity_type_metrics_json,
                ),
            )
            self._conn.commit()
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ResolutionMetrics._persist_snapshot() failed for run_id=%r: %s",
                snapshot.run_id,
                exc,
            )
            return False

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> RunMetricsSnapshot:
        """Convert a SQLite row to a :class:`RunMetricsSnapshot`."""
        paper_ids = json.loads(row["paper_ids_json"])
        per_strategy_counts = json.loads(row["per_strategy_counts_json"])
        entity_type_metrics_raw = json.loads(row["entity_type_metrics_json"])
        entity_type_metrics = [
            EntityTypeMetrics(**etm) for etm in entity_type_metrics_raw
        ]

        return RunMetricsSnapshot(
            run_id=row["run_id"],
            timestamp=row["timestamp"],
            paper_ids=paper_ids,
            total_forms=row["total_forms"],
            resolved_count=row["resolved_count"],
            unresolved_count=row["unresolved_count"],
            resolution_rate=row["resolution_rate"],
            per_strategy_counts=per_strategy_counts,
            entity_type_metrics=entity_type_metrics,
        )

    def reset(self) -> None:
        """
        Reset in-memory accumulators for a new run.

        Call this between runs if reusing the same ``ResolutionMetrics``
        instance across multiple pipeline runs.
        """
        self._accumulators = defaultdict(_EntityTypeAccumulator)
        self._global_strategy_counts = defaultdict(int)

    def close(self) -> None:
        """Close the underlying database connection if owned by this instance."""
        if self._owns_conn:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

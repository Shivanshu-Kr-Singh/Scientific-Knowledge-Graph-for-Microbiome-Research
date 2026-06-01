"""
ManualOverrideManager — curator-defined surface-form → canonical-ID mappings.

Backed by the ``manual_overrides`` table in ``canonical_registry.db``.
Integrates with :class:`~entity_resolution.resolution_cache.ResolutionCache`
to invalidate stale cache entries whenever an override is set or removed.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7, 9.8
"""

from __future__ import annotations

import csv
import io
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_canonical_registry_schema,
)
from entity_resolution.models import BulkImportResult, ManualOverride
from entity_resolution.utils import validate_canonical_id

logger = logging.getLogger(__name__)

# Required CSV columns for bulk import (Requirement 9.7)
_REQUIRED_CSV_COLUMNS = {
    "surface_form",
    "canonical_id",
    "entity_type",
    "curator_id",
    "justification",
}


class ManualOverrideManager:
    """
    Manages curator-defined Manual_Override mappings.

    Each override pins a ``surface_form`` to a specific ``canonical_id``,
    taking precedence over all automated resolution strategies (Req 9.1, 9.2).

    The manager is backed by the ``manual_overrides`` table in
    ``canonical_registry.db`` and optionally wired to a
    :class:`~entity_resolution.resolution_cache.ResolutionCache` instance
    for cache invalidation on set/remove (Req 9.5).

    The ``manual_overrides`` table may use either ``timestamp`` or
    ``created_at`` as the timestamp column name depending on which DDL was
    used to create the schema.  This class detects the correct column name
    at initialisation time and uses it consistently.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_path: str = "canonical_registry.db",
        conn: Optional[sqlite3.Connection] = None,
        resolution_cache: Optional[object] = None,
    ) -> None:
        """
        Initialise the manager.

        Args:
            db_path:          Path to the ``canonical_registry.db`` SQLite file.
                              Ignored when *conn* is provided.
            conn:             An already-open :class:`sqlite3.Connection`.
                              When supplied the manager uses this connection
                              directly and does **not** close it on GC.
            resolution_cache: An optional
                              :class:`~entity_resolution.resolution_cache.ResolutionCache`
                              instance.  When provided, ``set_override()`` and
                              ``remove_override()`` will call
                              ``resolution_cache.invalidate_version()`` with the
                              current registry version to evict stale entries
                              (Req 9.5).
        """
        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._owns_conn = True
            create_schema_in_connection(self._conn, get_canonical_registry_schema())

        # Ensure row_factory is set even for injected connections
        if self._conn.row_factory is None:
            self._conn.row_factory = sqlite3.Row

        self._cache: Optional[object] = resolution_cache

        # Detect which timestamp column name the table uses.
        # db_schema.py DDL uses "timestamp"; conftest.py fixture uses "created_at".
        self._ts_col: str = self._detect_timestamp_column()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_override(self, surface_form: str) -> Optional[ManualOverride]:
        """
        Retrieve the Manual_Override for *surface_form*, or ``None`` on miss.

        The lookup is case-sensitive (surface forms are stored as-is).

        Args:
            surface_form: The raw surface form to look up.

        Returns:
            A :class:`~entity_resolution.models.ManualOverride` if an override
            exists, ``None`` otherwise.

        Requirements: 9.1, 9.2
        """
        if not surface_form:
            return None

        try:
            row = self._conn.execute(
                f"SELECT surface_form, canonical_id, entity_type, curator_id, "
                f"justification, {self._ts_col} AS ts "
                f"FROM manual_overrides WHERE surface_form = ?",
                (surface_form,),
            ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.error("ManualOverrideManager.get_override() failed: %s", exc)
            return None

        if row is None:
            return None

        return self._row_to_override(row)

    def set_override(
        self,
        surface_form: str,
        canonical_id: str,
        entity_type: str,
        curator_id: str,
        justification: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Create or replace a Manual_Override for *surface_form*.

        Validates:
        - ``surface_form`` is non-empty
        - ``canonical_id`` format via :func:`~entity_resolution.utils.validate_canonical_id`
        - ``justification`` length ≤ 500 characters
        - ``curator_id`` is non-empty

        On success the :class:`~entity_resolution.resolution_cache.ResolutionCache`
        is invalidated for the current registry version so that the next
        ``resolve()`` call re-executes the strategy sequence and picks up the
        new override (Req 9.5).

        Args:
            surface_form:  The surface form to pin.
            canonical_id:  The canonical ID to pin it to.
            entity_type:   One of ``"taxon"``, ``"disease"``, ``"method"``.
            curator_id:    ID of the curator setting the override.
            justification: Optional note (≤ 500 characters).

        Returns:
            ``(True, None)`` on success.
            ``(False, error_message)`` on validation or DB failure.

        Requirements: 9.3, 9.5
        """
        # --- Validation ---------------------------------------------------
        if not surface_form or not surface_form.strip():
            return False, "surface_form must not be empty"

        if not canonical_id or not canonical_id.strip():
            return False, "canonical_id must not be empty"

        if not validate_canonical_id(canonical_id, entity_type):
            return False, (
                f"canonical_id '{canonical_id}' is not valid for "
                f"entity_type '{entity_type}'"
            )

        if justification is not None and len(justification) > 500:
            return False, (
                f"justification exceeds 500 characters (length={len(justification)})"
            )

        if not curator_id or not curator_id.strip():
            return False, "curator_id must not be empty"

        # --- Persist ------------------------------------------------------
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO manual_overrides "
                f"(surface_form, canonical_id, entity_type, curator_id, "
                f"justification, {self._ts_col}) "
                f"VALUES (?, ?, ?, ?, ?, ?)",
                (surface_form, canonical_id, entity_type, curator_id, justification, now),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ManualOverrideManager.set_override() DB write failed: %s", exc
            )
            return False, f"Database error: {exc}"

        # --- Cache invalidation -------------------------------------------
        self._invalidate_cache()

        logger.debug(
            "ManualOverride set: surface_form=%r canonical_id=%r curator=%r",
            surface_form,
            canonical_id,
            curator_id,
        )
        return True, None

    def remove_override(self, surface_form: str) -> bool:
        """
        Delete the Manual_Override for *surface_form* (if it exists).

        After deletion the :class:`~entity_resolution.resolution_cache.ResolutionCache`
        is invalidated so that the next ``resolve()`` call recomputes using
        automated strategies (Req 9.5).

        Args:
            surface_form: The surface form whose override should be removed.

        Returns:
            ``True`` on success (including when no override existed),
            ``False`` on a database error.

        Requirements: 9.5
        """
        if not surface_form:
            return True  # Nothing to remove — not an error

        try:
            self._conn.execute(
                "DELETE FROM manual_overrides WHERE surface_form = ?",
                (surface_form,),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ManualOverrideManager.remove_override() DB delete failed: %s", exc
            )
            return False

        # --- Cache invalidation -------------------------------------------
        self._invalidate_cache()

        logger.debug("ManualOverride removed: surface_form=%r", surface_form)
        return True

    def bulk_import_csv(
        self,
        csv_content: str,
    ) -> BulkImportResult:
        """
        Import Manual_Overrides from a CSV string.

        Expected columns (order-independent, header row required)::

            surface_form, canonical_id, entity_type, curator_id, justification

        Skips and logs malformed rows:
        - Missing required columns (any of the five above)
        - Invalid ``canonical_id`` format for the given ``entity_type``
        - ``surface_form`` already has a Manual_Override for a *different*
          ``canonical_id`` (duplicate conflict)
        - ``justification`` exceeds 500 characters

        Processing continues after each skipped row without aborting the import.

        Args:
            csv_content: The full CSV text (including header row).

        Returns:
            A :class:`~entity_resolution.models.BulkImportResult` with counts
            of imported and skipped rows plus details of each skipped row.

        Requirements: 9.7, 9.8
        """
        imported_count = 0
        skipped_rows: list[dict] = []

        try:
            reader = csv.DictReader(io.StringIO(csv_content))
        except Exception as exc:  # noqa: BLE001
            logger.error("bulk_import_csv() failed to parse CSV: %s", exc)
            return BulkImportResult(
                total_rows=0,
                imported_count=0,
                skipped_count=0,
                skipped_rows=[],
            )

        # Validate that the header contains all required columns
        if reader.fieldnames is None:
            logger.warning("bulk_import_csv() received empty CSV (no header)")
            return BulkImportResult(
                total_rows=0,
                imported_count=0,
                skipped_count=0,
                skipped_rows=[],
            )

        header_cols = {col.strip() for col in reader.fieldnames}
        missing_header_cols = _REQUIRED_CSV_COLUMNS - header_cols
        if missing_header_cols:
            reason = (
                f"CSV header is missing required columns: {sorted(missing_header_cols)}"
            )
            logger.warning("bulk_import_csv() %s", reason)
            return BulkImportResult(
                total_rows=0,
                imported_count=0,
                skipped_count=0,
                skipped_rows=[{"row_number": 0, "reason": reason}],
            )

        rows = list(reader)
        total_rows = len(rows)

        for row_number, row in enumerate(rows, start=2):  # row 1 is the header
            # Strip whitespace from all values
            row = {k.strip(): (v.strip() if v else "") for k, v in row.items()}

            # --- Check for missing required field values ------------------
            # justification is optional (can be empty), all others are required
            missing_values = [
                col
                for col in _REQUIRED_CSV_COLUMNS
                if col != "justification" and not row.get(col, "")
            ]
            if missing_values:
                reason = f"Missing required field values: {sorted(missing_values)}"
                logger.warning(
                    "bulk_import_csv() row %d skipped — %s", row_number, reason
                )
                skipped_rows.append(
                    {"row_number": row_number, "reason": reason, "row": dict(row)}
                )
                continue

            surface_form = row["surface_form"]
            canonical_id = row["canonical_id"]
            entity_type = row["entity_type"]
            curator_id = row["curator_id"]
            justification = row.get("justification") or None

            # --- Validate canonical_id format -----------------------------
            if not validate_canonical_id(canonical_id, entity_type):
                reason = (
                    f"Invalid canonical_id '{canonical_id}' for "
                    f"entity_type '{entity_type}'"
                )
                logger.warning(
                    "bulk_import_csv() row %d skipped — %s", row_number, reason
                )
                skipped_rows.append(
                    {"row_number": row_number, "reason": reason, "row": dict(row)}
                )
                continue

            # --- Validate justification length ----------------------------
            if justification is not None and len(justification) > 500:
                reason = (
                    f"justification exceeds 500 characters "
                    f"(length={len(justification)})"
                )
                logger.warning(
                    "bulk_import_csv() row %d skipped — %s", row_number, reason
                )
                skipped_rows.append(
                    {"row_number": row_number, "reason": reason, "row": dict(row)}
                )
                continue

            # --- Check for duplicate override conflict --------------------
            existing = self.get_override(surface_form)
            if existing is not None and existing.canonical_id != canonical_id:
                reason = (
                    f"surface_form '{surface_form}' already has a Manual_Override "
                    f"for canonical_id '{existing.canonical_id}' "
                    f"(conflict with '{canonical_id}')"
                )
                logger.warning(
                    "bulk_import_csv() row %d skipped — %s", row_number, reason
                )
                skipped_rows.append(
                    {"row_number": row_number, "reason": reason, "row": dict(row)}
                )
                continue

            # --- Persist the override -------------------------------------
            ok, err = self.set_override(
                surface_form=surface_form,
                canonical_id=canonical_id,
                entity_type=entity_type,
                curator_id=curator_id,
                justification=justification,
            )
            if ok:
                imported_count += 1
            else:
                reason = f"set_override() failed: {err}"
                logger.warning(
                    "bulk_import_csv() row %d skipped — %s", row_number, reason
                )
                skipped_rows.append(
                    {"row_number": row_number, "reason": reason, "row": dict(row)}
                )

        return BulkImportResult(
            total_rows=total_rows,
            imported_count=imported_count,
            skipped_count=len(skipped_rows),
            skipped_rows=skipped_rows,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_timestamp_column(self) -> str:
        """
        Detect whether the ``manual_overrides`` table uses ``created_at``
        or ``timestamp`` as the timestamp column name.

        The ``db_schema.py`` DDL uses ``timestamp``; the ``conftest.py``
        fixture uses ``created_at``.  We probe the table info to pick the
        right name so the manager works with both schemas.

        Returns:
            ``"timestamp"`` or ``"created_at"`` depending on which column
            exists in the table.  Defaults to ``"timestamp"`` if the table
            does not yet exist (matches db_schema.py DDL).
        """
        try:
            rows = self._conn.execute(
                "PRAGMA table_info(manual_overrides)"
            ).fetchall()
            # PRAGMA table_info returns rows with columns:
            # cid, name, type, notnull, dflt_value, pk
            # We access by index since row_factory may not be set yet.
            col_names: set[str] = set()
            for row in rows:
                try:
                    # Try dict-style access first (sqlite3.Row)
                    col_names.add(row["name"])
                except (TypeError, IndexError):
                    # Fall back to positional access
                    col_names.add(row[1])

            if "created_at" in col_names:
                return "created_at"
            # Default to "timestamp" (matches db_schema.py DDL)
            return "timestamp"
        except Exception:  # noqa: BLE001
            # Default to "timestamp" (matches db_schema.py DDL)
            return "timestamp"

    def _row_to_override(self, row: sqlite3.Row) -> ManualOverride:
        """Convert a SQLite row to a :class:`ManualOverride` model."""
        ts_str = row["ts"]
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        return ManualOverride(
            surface_form=row["surface_form"],
            canonical_id=row["canonical_id"],
            entity_type=row["entity_type"],
            curator_id=row["curator_id"],
            justification=row["justification"],
            timestamp=ts,
        )

    def _invalidate_cache(self) -> None:
        """
        Invalidate the :class:`~entity_resolution.resolution_cache.ResolutionCache`
        by calling ``invalidate_version()`` with the current registry version.

        This ensures that any cached resolution results are evicted so that
        the next ``resolve()`` call re-executes the full strategy sequence
        and picks up the new or removed override (Req 9.5).

        No-op when no cache is wired.
        """
        if self._cache is None:
            return

        try:
            row = self._conn.execute(
                "SELECT version FROM registry_version WHERE id = 1"
            ).fetchone()
            if row is not None:
                try:
                    current_version = int(row["version"])
                except (TypeError, KeyError):
                    current_version = int(row[0])
                self._cache.invalidate_version(current_version)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ManualOverrideManager._invalidate_cache() failed: %s", exc
            )

    def close(self) -> None:
        """Close the underlying database connection if owned by this instance."""
        if self._owns_conn:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

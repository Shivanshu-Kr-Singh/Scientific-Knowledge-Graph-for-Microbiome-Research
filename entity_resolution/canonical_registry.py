"""
CanonicalRegistry — SQLite-backed persistent store for canonical entity records.

Validates ID formats, maintains an in-memory synonym lookup dict transactionally,
and detects duplicate surface-form conflicts.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_canonical_registry_schema,
)
from entity_resolution.models import (
    CanonicalEntityRecord,
    EntityType,
    RegistrationError,
    SynonymConflictRecord,
    SynonymProvenance,
    SynonymRecord,
)
from entity_resolution.utils import normalize_surface_form, validate_canonical_id

logger = logging.getLogger(__name__)


class CanonicalRegistry:
    """
    SQLite-backed persistent store for canonical entity records.

    The registry maintains:
    - A SQLite database (``canonical_registry.db``) with tables:
        ``canonical_entities``, ``synonyms``, ``synonym_conflicts``,
        ``registry_version``, ``manual_overrides``, ``abbreviation_table``
    - An in-memory dict ``_synonym_index: dict[str, str]`` mapping
      NFC-normalised, lowercased surface forms to canonical IDs.
      This dict is designed so that a ``SynonymIndex`` object (task 3.1)
      can be injected later via :meth:`set_synonym_index`.

    Thread safety: a ``threading.RLock`` guards all writes to both the
    in-memory dict and the SQLite database so that concurrent lookups
    always see a consistent state.

    ID validation rules (Requirements 3.2, 3.3, 3.4):
    - taxon:   canonical_id must be a positive integer string (e.g. "562")
    - disease: canonical_id must match ``^[A-Z]\\d+$`` (e.g. "D006262")
    - method:  canonical_id must match ``^METHOD-[A-Za-z0-9]+$`` (e.g. "METHOD-16S")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_path: str = "canonical_registry.db",
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """
        Initialise the registry.

        Args:
            db_path: Path to the SQLite database file.  Ignored when *conn*
                     is provided (useful for in-memory test databases).
            conn:    An already-open :class:`sqlite3.Connection`.  When
                     supplied the registry uses this connection directly and
                     does **not** close it on garbage collection.
        """
        self._lock = threading.RLock()

        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._owns_conn = True

        # Ensure schema exists (idempotent)
        create_schema_in_connection(self._conn, get_canonical_registry_schema())

        # In-memory synonym lookup: normalised_surface_form -> canonical_id
        # This is the "own" index; a SynonymIndex can be injected later.
        self._synonym_index: Dict[str, str] = {}

        # External SynonymIndex object (injected via set_synonym_index)
        self._external_synonym_index: Optional[object] = None

        # Populate in-memory index from existing DB rows
        self._rebuild_synonym_index()

    def __del__(self) -> None:
        if self._owns_conn:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SynonymIndex injection (for task 3.1 wiring)
    # ------------------------------------------------------------------

    def set_synonym_index(self, synonym_index: object) -> None:
        """
        Inject an external ``SynonymIndex`` instance.

        When set, :meth:`register` and :meth:`add_synonym` will call
        ``synonym_index.add(surface_form, canonical_id)`` in addition to
        updating the internal dict.  This allows the SynonymIndex
        implemented in task 3.1 to be wired in without changing this class.
        """
        self._external_synonym_index = synonym_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self, record: CanonicalEntityRecord
    ) -> Tuple[bool, Optional[RegistrationError]]:
        """
        Register a new canonical entity.

        Validates the canonical_id format, persists the ``canonical_entities``
        row, and inserts all synonym rows — all within a single SQLite
        transaction.  The in-memory synonym index is updated only after the
        transaction commits successfully.

        Returns:
            ``(True, None)`` on success.
            ``(False, RegistrationError)`` on any validation or DB failure;
            no partial record is created.

        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
        """
        # --- Validation ---------------------------------------------------
        if not record.canonical_id:
            return False, RegistrationError(
                field="canonical_id", message="canonical_id must not be empty"
            )

        if not validate_canonical_id(record.canonical_id, record.entity_type.value):
            return False, RegistrationError(
                field="canonical_id",
                message=(
                    f"canonical_id '{record.canonical_id}' is not valid for "
                    f"entity_type '{record.entity_type.value}'"
                ),
            )

        if not record.primary_name or not record.primary_name.strip():
            return False, RegistrationError(
                field="primary_name", message="primary_name must not be empty"
            )

        # Validate synonym lengths up-front so we can reject before touching DB
        for syn in record.synonyms:
            if len(syn.surface_form) > 500:
                return False, RegistrationError(
                    field="synonyms",
                    message=(
                        f"Synonym surface_form exceeds 500 characters: "
                        f"'{syn.surface_form[:60]}...'"
                    ),
                )

        # --- Atomic DB write + in-memory update ---------------------------
        with self._lock:
            # Collect synonym conflicts detected during the transaction so they
            # can be persisted in a separate transaction after rollback.
            pending_conflicts: List[SynonymConflictRecord] = []
            try:
                with self._conn:  # begins a transaction; commits or rolls back
                    # 1. Insert canonical entity row
                    self._conn.execute(
                        """
                        INSERT INTO canonical_entities
                            (canonical_id, primary_name, entity_type,
                             ontology_source, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.canonical_id,
                            record.primary_name,
                            record.entity_type.value,
                            record.ontology_source,
                            record.created_at.isoformat(),
                            record.updated_at.isoformat(),
                        ),
                    )

                    # 2. Insert synonym rows (including primary_name as a synonym)
                    all_surface_forms: List[Tuple[str, SynonymProvenance, Optional[str], str]] = []

                    # Primary name is always registered as a synonym
                    all_surface_forms.append(
                        (
                            record.primary_name,
                            SynonymProvenance.ONTOLOGY,
                            None,
                            record.created_at.isoformat(),
                        )
                    )
                    for syn in record.synonyms:
                        all_surface_forms.append(
                            (
                                syn.surface_form,
                                syn.provenance,
                                syn.added_by,
                                syn.added_at.isoformat(),
                            )
                        )

                    # Collect normalised forms to update in-memory index after commit
                    pending_index_updates: List[Tuple[str, str]] = []

                    for surface_form, provenance, added_by, added_at in all_surface_forms:
                        normalised = normalize_surface_form(surface_form)
                        if not normalised:
                            continue

                        # Check for cross-entity conflict in DB
                        existing = self._conn.execute(
                            "SELECT canonical_id FROM synonyms WHERE surface_form_normalized = ?",
                            (normalised,),
                        ).fetchone()

                        if existing and existing["canonical_id"] != record.canonical_id:
                            # Collect conflict to log after rollback (Requirements 3.7, 5.4)
                            conflict = SynonymConflictRecord(
                                duplicate_surface_form=surface_form,
                                entity_a_id=record.canonical_id,
                                entity_b_id=existing["canonical_id"],
                                timestamp=datetime.now(timezone.utc),
                                provenance_source=provenance.value,
                            )
                            pending_conflicts.append(conflict)
                            raise sqlite3.IntegrityError(
                                f"Surface form '{surface_form}' already registered "
                                f"for canonical_id '{existing['canonical_id']}'"
                            )

                        # Skip if already registered for the same canonical_id
                        if existing and existing["canonical_id"] == record.canonical_id:
                            pending_index_updates.append((normalised, record.canonical_id))
                            continue

                        self._conn.execute(
                            """
                            INSERT OR IGNORE INTO synonyms
                                (canonical_id, surface_form, surface_form_normalized,
                                 provenance, added_by, added_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                record.canonical_id,
                                surface_form,
                                normalised,
                                provenance.value,
                                added_by,
                                added_at,
                            ),
                        )
                        pending_index_updates.append((normalised, record.canonical_id))

                    # 3. Bump registry version
                    self._bump_version_in_transaction()

                # Transaction committed — now update in-memory index
                for normalised, cid in pending_index_updates:
                    self._synonym_index[normalised] = cid
                    if self._external_synonym_index is not None:
                        try:
                            self._external_synonym_index.add(normalised, cid)
                        except Exception as exc:
                            logger.warning(
                                "External SynonymIndex.add() failed for '%s': %s",
                                normalised,
                                exc,
                            )

                return True, None

            except sqlite3.IntegrityError as exc:
                # Duplicate canonical_id or synonym conflict — transaction rolled back.
                # Persist any collected synonym conflict records in a separate
                # transaction so they survive the rollback (Requirements 3.7, 5.4).
                for conflict in pending_conflicts:
                    try:
                        with self._conn:
                            self._log_conflict_in_transaction(conflict)
                    except Exception as log_exc:
                        logger.warning(
                            "Failed to persist SynonymConflictRecord after rollback: %s",
                            log_exc,
                        )
                logger.debug("register() IntegrityError: %s", exc)
                return False, RegistrationError(
                    field="canonical_id",
                    message=str(exc),
                )
            except Exception as exc:
                logger.error("register() unexpected error: %s", exc)
                return False, RegistrationError(
                    field="canonical_id",
                    message=f"Unexpected error during registration: {exc}",
                )

    def lookup_by_surface_form(
        self, surface_form: str
    ) -> Optional[CanonicalEntityRecord]:
        """
        Case-insensitive, NFC-normalised lookup.

        Returns the :class:`CanonicalEntityRecord` if found, ``None`` otherwise.
        Never raises an exception on a miss.

        Requirements: 3.5
        """
        if not surface_form:
            return None

        normalised = normalize_surface_form(surface_form)

        # Fast path: in-memory index
        with self._lock:
            canonical_id = self._synonym_index.get(normalised)

        if canonical_id is None:
            # Fallback: query DB directly (handles cases where index is stale)
            row = self._conn.execute(
                "SELECT canonical_id FROM synonyms WHERE surface_form_normalized = ?",
                (normalised,),
            ).fetchone()
            if row is None:
                return None
            canonical_id = row["canonical_id"]

        return self._load_record(canonical_id)

    def add_synonym(
        self,
        canonical_id: str,
        surface_form: str,
        provenance: SynonymProvenance,
        added_by: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Add a synonym to an existing canonical entity.

        Validates:
        - ``surface_form`` length ≤ 500 characters
        - ``surface_form`` is not already registered for a *different* canonical_id

        Updates both the ``synonyms`` table and the in-memory index atomically.
        Logs a :class:`SynonymConflictRecord` on duplicate.

        Returns:
            ``(True, None)`` on success.
            ``(False, error_message)`` on failure; no partial state is created.

        Requirements: 3.6, 3.7, 5.1, 5.4
        """
        if len(surface_form) > 500:
            return False, (
                f"surface_form exceeds 500 characters (length={len(surface_form)})"
            )

        normalised = normalize_surface_form(surface_form)
        if not normalised:
            return False, "surface_form normalises to an empty string"

        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            try:
                with self._conn:
                    # Check canonical entity exists
                    entity_row = self._conn.execute(
                        "SELECT canonical_id FROM canonical_entities WHERE canonical_id = ?",
                        (canonical_id,),
                    ).fetchone()
                    if entity_row is None:
                        return False, f"canonical_id '{canonical_id}' not found in registry"

                    # Check for cross-entity conflict
                    existing = self._conn.execute(
                        "SELECT canonical_id FROM synonyms WHERE surface_form_normalized = ?",
                        (normalised,),
                    ).fetchone()

                    if existing:
                        if existing["canonical_id"] == canonical_id:
                            # Already registered for the same entity — idempotent success
                            return True, None
                        else:
                            # Conflict: surface form belongs to a different entity
                            conflict = SynonymConflictRecord(
                                duplicate_surface_form=surface_form,
                                entity_a_id=canonical_id,
                                entity_b_id=existing["canonical_id"],
                                timestamp=datetime.now(timezone.utc),
                                provenance_source=provenance.value,
                            )
                            self._log_conflict_in_transaction(conflict)
                            return False, (
                                f"Surface form '{surface_form}' is already registered "
                                f"for canonical_id '{existing['canonical_id']}'"
                            )

                    # Insert synonym
                    self._conn.execute(
                        """
                        INSERT INTO synonyms
                            (canonical_id, surface_form, surface_form_normalized,
                             provenance, added_by, added_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            canonical_id,
                            surface_form,
                            normalised,
                            provenance.value,
                            added_by,
                            now,
                        ),
                    )

                    # Bump version
                    self._bump_version_in_transaction()

                # Transaction committed — update in-memory index
                self._synonym_index[normalised] = canonical_id
                if self._external_synonym_index is not None:
                    try:
                        self._external_synonym_index.add(normalised, canonical_id)
                    except Exception as exc:
                        logger.warning(
                            "External SynonymIndex.add() failed for '%s': %s",
                            normalised,
                            exc,
                        )

                return True, None

            except sqlite3.IntegrityError as exc:
                logger.debug("add_synonym() IntegrityError: %s", exc)
                return False, str(exc)
            except Exception as exc:
                logger.error("add_synonym() unexpected error: %s", exc)
                return False, f"Unexpected error: {exc}"

    def get_registry_version(self) -> int:
        """
        Return the current monotonically increasing registry version.

        The version is stored in the ``registry_version`` table (single row,
        id=1) and is bumped on every write (register, add_synonym).

        Requirements: 3.1 (implied by version-based cache invalidation in 8.5)
        """
        row = self._conn.execute(
            "SELECT version FROM registry_version WHERE id = 1"
        ).fetchone()
        if row is None:
            return 0
        return int(row["version"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bump_version_in_transaction(self) -> None:
        """
        Increment the registry version by 1.

        Must be called inside an active SQLite transaction (i.e. inside a
        ``with self._conn:`` block).
        """
        self._conn.execute(
            "UPDATE registry_version SET version = version + 1 WHERE id = 1"
        )

    def _log_conflict_in_transaction(self, conflict: SynonymConflictRecord) -> None:
        """
        Persist a :class:`SynonymConflictRecord` to ``synonym_conflicts``.

        Must be called inside an active SQLite transaction.
        """
        self._conn.execute(
            """
            INSERT INTO synonym_conflicts
                (duplicate_surface_form, entity_a_id, entity_b_id,
                 timestamp, provenance_source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conflict.duplicate_surface_form,
                conflict.entity_a_id,
                conflict.entity_b_id,
                conflict.timestamp.isoformat(),
                conflict.provenance_source,
            ),
        )

    def _rebuild_synonym_index(self) -> None:
        """
        Populate the in-memory synonym index from the ``synonyms`` table.

        Called once during ``__init__`` to restore state from a persistent DB.
        """
        with self._lock:
            self._synonym_index.clear()
            rows = self._conn.execute(
                "SELECT surface_form_normalized, canonical_id FROM synonyms"
            ).fetchall()
            for row in rows:
                self._synonym_index[row["surface_form_normalized"]] = row["canonical_id"]

    def _load_record(self, canonical_id: str) -> Optional[CanonicalEntityRecord]:
        """
        Load a full :class:`CanonicalEntityRecord` from the database.

        Returns ``None`` if the canonical_id does not exist.
        """
        entity_row = self._conn.execute(
            """
            SELECT canonical_id, primary_name, entity_type, ontology_source,
                   created_at, updated_at
            FROM canonical_entities
            WHERE canonical_id = ?
            """,
            (canonical_id,),
        ).fetchone()

        if entity_row is None:
            return None

        synonym_rows = self._conn.execute(
            """
            SELECT surface_form, provenance, added_by, added_at
            FROM synonyms
            WHERE canonical_id = ?
            """,
            (canonical_id,),
        ).fetchall()

        synonyms = [
            SynonymRecord(
                surface_form=row["surface_form"],
                provenance=SynonymProvenance(row["provenance"]),
                added_by=row["added_by"],
                added_at=datetime.fromisoformat(row["added_at"]),
            )
            for row in synonym_rows
        ]

        return CanonicalEntityRecord(
            canonical_id=entity_row["canonical_id"],
            primary_name=entity_row["primary_name"],
            entity_type=EntityType(entity_row["entity_type"]),
            ontology_source=entity_row["ontology_source"],
            synonyms=synonyms,
            created_at=datetime.fromisoformat(entity_row["created_at"]),
            updated_at=datetime.fromisoformat(entity_row["updated_at"]),
        )

    def lookup_by_canonical_id(
        self, canonical_id: str
    ) -> Optional[CanonicalEntityRecord]:
        """
        Look up a canonical entity record by its canonical_id.

        Returns the :class:`CanonicalEntityRecord` if found, ``None`` otherwise.
        Never raises an exception on a miss.

        Used by OntologyTraverser to check whether an ancestor ID exists in
        the registry.

        Requirements: 13.1 (OntologyTraverser registry check)
        """
        if not canonical_id:
            return None
        return self._load_record(canonical_id)

    # ------------------------------------------------------------------
    # Convenience: get all surface forms for a canonical_id (used by FuzzyMatcher)
    # ------------------------------------------------------------------

    def get_all_surface_forms(
        self, entity_type: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Return all ``(surface_form_normalized, canonical_id)`` pairs.

        Optionally filtered by *entity_type*.  Used by FuzzyMatcher (task 8).
        """
        if entity_type:
            rows = self._conn.execute(
                """
                SELECT s.surface_form_normalized, s.canonical_id
                FROM synonyms s
                JOIN canonical_entities e ON s.canonical_id = e.canonical_id
                WHERE e.entity_type = ?
                """,
                (entity_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT surface_form_normalized, canonical_id FROM synonyms"
            ).fetchall()

        return [(row["surface_form_normalized"], row["canonical_id"]) for row in rows]

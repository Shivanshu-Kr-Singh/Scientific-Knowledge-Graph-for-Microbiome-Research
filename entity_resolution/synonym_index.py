"""
SynonymIndex — in-memory inverted index with RW lock and SQLite backing.

Maps NFC-normalised, lowercased surface forms to canonical entity IDs.
Protected by a threading.RLock for atomic updates with ≤100ms blocking.
Backed by the ``synonyms`` table in ``canonical_registry.db`` for
persistence across restarts.

Requirements: 5.1, 5.2, 5.5
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import TYPE_CHECKING, Dict, List, Optional

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_canonical_registry_schema,
)
from entity_resolution.models import EntityType, SynonymIndexEntry
from entity_resolution.utils import normalize_surface_form

if TYPE_CHECKING:
    from entity_resolution.canonical_registry import CanonicalRegistry

logger = logging.getLogger(__name__)

# Maximum number of results returned by prefix_lookup (Requirement 5.5)
_PREFIX_LOOKUP_CAP = 50


class SynonymIndex:
    """
    In-memory inverted index with RW lock for atomic updates.

    Internal structure:
    - ``_index``: ``dict[str, str]``  — normalised_surface_form -> canonical_id
    - ``_lock``:  ``threading.RLock`` — reentrant RW lock

    Invariants:
    - All keys are NFC-normalised and lowercased (via ``normalize_surface_form``).
    - Each surface form maps to exactly one canonical_id.
    - Index is consistent with CanonicalRegistry (updated in same transaction).

    The ``_entity_type_map`` is an auxiliary dict that stores
    ``normalised_surface_form -> entity_type`` so that ``prefix_lookup``
    can return full :class:`~entity_resolution.models.SynonymIndexEntry`
    objects without an extra DB round-trip.

    Requirements: 5.1, 5.2, 5.5
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
        Initialise the SynonymIndex.

        Args:
            db_path: Path to the ``canonical_registry.db`` SQLite file.
                     Ignored when *conn* is provided (useful for in-memory
                     test databases).
            conn:    An already-open :class:`sqlite3.Connection`.  When
                     supplied the index uses this connection directly and
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

        # Ensure schema exists (idempotent — safe to call on existing DB)
        create_schema_in_connection(self._conn, get_canonical_registry_schema())

        # Primary in-memory index: normalised_surface_form -> canonical_id
        self._index: Dict[str, str] = {}

        # Auxiliary map: normalised_surface_form -> entity_type (str)
        # Populated alongside _index so prefix_lookup can return full entries.
        self._entity_type_map: Dict[str, str] = {}

        # Populate from existing DB rows
        self._load_from_db()

    def __del__(self) -> None:
        if self._owns_conn:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, surface_form: str) -> Optional[str]:
        """
        Return the canonical_id for *surface_form*, or ``None`` on a miss.

        The surface form is NFC-normalised and lowercased before the dict
        lookup.  The read lock is held for the duration of the lookup.

        O(1) average case.

        Requirements: 5.1
        """
        if not surface_form:
            return None

        normalised = normalize_surface_form(surface_form)

        with self._lock:
            return self._index.get(normalised)

    def add(self, surface_form: str, canonical_id: str, entity_type: str = "") -> None:
        """
        Atomically add a surface-form → canonical_id mapping.

        The write lock is acquired before any mutation.  Both the in-memory
        ``_index`` and the SQLite ``synonyms`` table are updated in the same
        operation so that concurrent lookups always see a consistent state.

        If *surface_form* is already mapped to *canonical_id* the call is a
        no-op (idempotent).  If it is mapped to a *different* canonical_id a
        ``ValueError`` is raised and no state is changed.

        Args:
            surface_form:  Raw surface form (will be normalised internally).
            canonical_id:  The canonical entity ID to map to.
            entity_type:   Optional entity type string (stored for
                           ``prefix_lookup``).  Defaults to empty string when
                           not provided (e.g. when called from
                           ``CanonicalRegistry`` which does not pass it).

        Raises:
            ValueError: If *surface_form* is already mapped to a different
                        canonical_id.

        Requirements: 5.2
        """
        if not surface_form:
            return

        normalised = normalize_surface_form(surface_form)
        if not normalised:
            return

        with self._lock:
            existing = self._index.get(normalised)
            if existing is not None:
                if existing == canonical_id:
                    # Idempotent — already correct
                    return
                raise ValueError(
                    f"Surface form '{surface_form}' (normalised: '{normalised}') "
                    f"is already mapped to canonical_id '{existing}'; "
                    f"cannot remap to '{canonical_id}'"
                )

            # Persist to SQLite first; if it fails we do NOT update _index
            # so the in-memory state stays consistent with the DB.
            try:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO synonyms
                        (canonical_id, surface_form, surface_form_normalized,
                         provenance, added_by, added_at)
                    VALUES (?, ?, ?, 'ontology', NULL, datetime('now'))
                    """,
                    (canonical_id, surface_form, normalised),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.error(
                    "SynonymIndex.add(): SQLite error persisting '%s' -> '%s': %s",
                    normalised,
                    canonical_id,
                    exc,
                )
                raise

            # Update in-memory index only after successful DB write
            self._index[normalised] = canonical_id
            if entity_type:
                self._entity_type_map[normalised] = entity_type

    def prefix_lookup(self, prefix: str) -> List[SynonymIndexEntry]:
        """
        Return all entries whose normalised surface form starts with the
        normalised *prefix*.

        Results are:
        - Capped at 50 entries (Requirement 5.5).
        - Sorted lexicographically by ``surface_form_normalized``.
        - Returned as :class:`~entity_resolution.models.SynonymIndexEntry`
          objects.

        The read lock is held for the duration of the iteration.

        Requirements: 5.5
        """
        if not prefix:
            return []

        normalised_prefix = normalize_surface_form(prefix)

        results: List[SynonymIndexEntry] = []

        with self._lock:
            # Iterate over a snapshot of the keys to avoid mutation during
            # iteration (the lock prevents concurrent writes, but being
            # explicit is safer).
            for norm_form, canonical_id in self._index.items():
                if norm_form.startswith(normalised_prefix):
                    entity_type_str = self._entity_type_map.get(norm_form, "")
                    # Resolve entity_type to the enum; fall back to TAXON if unknown
                    try:
                        entity_type = EntityType(entity_type_str)
                    except ValueError:
                        entity_type = EntityType.TAXON

                    results.append(
                        SynonymIndexEntry(
                            surface_form_normalized=norm_form,
                            canonical_id=canonical_id,
                            entity_type=entity_type,
                        )
                    )

                    # Early exit once we have more than the cap to avoid
                    # scanning the entire index unnecessarily.
                    if len(results) > _PREFIX_LOOKUP_CAP:
                        break

        # Sort lexicographically by surface_form_normalized, then cap at 50
        results.sort(key=lambda e: e.surface_form_normalized)
        return results[:_PREFIX_LOOKUP_CAP]

    def rebuild_from_registry(self, registry: "CanonicalRegistry") -> None:
        """
        Rebuild the in-memory index from the SQLite ``synonyms`` table.

        Clears ``_index`` and ``_entity_type_map``, then reloads all rows
        from the database.  Used on startup to restore state from a
        persistent DB.

        The write lock is held for the entire operation so that concurrent
        lookups see either the old complete state or the new complete state.

        Requirements: 5.1 (consistency with CanonicalRegistry)
        """
        with self._lock:
            self._index.clear()
            self._entity_type_map.clear()
            self._load_from_db_locked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        """
        Populate ``_index`` and ``_entity_type_map`` from the DB.

        Called from ``__init__`` (no lock needed — object not yet shared).
        """
        rows = self._conn.execute(
            """
            SELECT s.surface_form_normalized, s.canonical_id, e.entity_type
            FROM synonyms s
            LEFT JOIN canonical_entities e ON s.canonical_id = e.canonical_id
            """
        ).fetchall()
        for row in rows:
            norm = row[0]
            cid = row[1]
            etype = row[2] or ""
            self._index[norm] = cid
            if etype:
                self._entity_type_map[norm] = etype

    def _load_from_db_locked(self) -> None:
        """
        Same as ``_load_from_db`` but assumes the lock is already held.

        Used by ``rebuild_from_registry`` which acquires the lock itself.
        """
        rows = self._conn.execute(
            """
            SELECT s.surface_form_normalized, s.canonical_id, e.entity_type
            FROM synonyms s
            LEFT JOIN canonical_entities e ON s.canonical_id = e.canonical_id
            """
        ).fetchall()
        for row in rows:
            norm = row[0]
            cid = row[1]
            etype = row[2] or ""
            self._index[norm] = cid
            if etype:
                self._entity_type_map[norm] = etype

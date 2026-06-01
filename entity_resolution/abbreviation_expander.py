"""
AbbreviationExpander — curated abbreviation table + genus-initial pattern matching.

Backed by the ``abbreviation_table`` in ``canonical_registry.db``.
Supports hot-reload of new mappings without pipeline restart.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_canonical_registry_schema,
)
from entity_resolution.models import ExpansionCandidate

logger = logging.getLogger(__name__)

# Genus-initial pattern: single uppercase letter + '.' + space + species epithet
# e.g. "E. coli", "B. subtilis"
_GENUS_INITIAL_RE = re.compile(r"^([A-Z])\. (\w+)$")


class AbbreviationExpander:
    """
    Curated abbreviation table + genus-initial pattern matching.

    The expander maintains an in-memory copy of the ``abbreviation_table``
    from ``canonical_registry.db`` for fast lookups.  New mappings added via
    :meth:`add_mapping` are persisted to SQLite **and** immediately reflected
    in the in-memory table (hot-reload — no restart required).

    Genus-initial pattern
    ---------------------
    A surface form matching ``^[A-Z]\\. \\w+$`` (e.g. "E. coli") triggers a
    secondary lookup: all ``full_form`` values in the abbreviation table whose
    first character equals the initial letter are returned as candidates.

    Confidence rules (Requirements 11.2, 11.4)
    -------------------------------------------
    - N = 1 candidate  →  confidence = 1.0
    - N > 1 candidates →  confidence = 1.0 / N  (equal for all candidates)

    Thread safety
    -------------
    A ``threading.RLock`` guards all reads and writes to the in-memory table
    so that concurrent calls to :meth:`expand` and :meth:`add_mapping` are
    safe.
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
        Initialise the expander.

        Args:
            db_path: Path to the SQLite database file.  Ignored when *conn*
                     is provided (useful for in-memory test databases).
            conn:    An already-open :class:`sqlite3.Connection`.  When
                     supplied the expander uses this connection directly and
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

        # Ensure schema exists (idempotent — creates abbreviation_table if absent)
        create_schema_in_connection(self._conn, get_canonical_registry_schema())

        # In-memory table: abbreviated_form -> list of full_forms
        # e.g. {"E. coli": ["Escherichia coli"], "SCFA": ["short-chain fatty acid"]}
        self._table: Dict[str, List[str]] = {}

        # Populate from DB
        self._reload_table()

    def __del__(self) -> None:
        if self._owns_conn:
            try:
                self._conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, surface_form: str) -> List[ExpansionCandidate]:
        """
        Expand an abbreviated surface form to candidate full forms.

        Strategy (applied in order):
        1. Exact match against the curated abbreviation table.
        2. Genus-initial pattern (``^[A-Z]\\. \\w+$``): look up all full forms
           whose first character equals the initial letter.

        Returns:
            A list of :class:`ExpansionCandidate` objects sorted
            lexicographically by ``expanded_form``.  Returns ``[]`` when no
            match is found — never raises.

        Requirements: 11.1, 11.2, 11.3, 11.4
        """
        if not surface_form:
            return []

        with self._lock:
            # --- Step 1: exact match in curated table ---
            exact_matches = self._table.get(surface_form)
            if exact_matches:
                return self._build_candidates(exact_matches)

            # --- Step 2: genus-initial pattern ---
            m = _GENUS_INITIAL_RE.match(surface_form)
            if m:
                initial = m.group(1)          # e.g. "E"
                species_epithet = m.group(2)  # e.g. "coli"
                genus_candidates = self._genus_initial_lookup(initial, species_epithet)
                if genus_candidates:
                    return self._build_candidates(genus_candidates)

        return []

    def add_mapping(
        self,
        abbreviated_form: str,
        full_form: str,
        added_by: str,
    ) -> None:
        """
        Add a new abbreviation mapping.

        Persists the mapping to the ``abbreviation_table`` in SQLite and
        immediately reloads the in-memory table so that all subsequent
        :meth:`expand` calls reflect the new mapping (hot-reload).

        Args:
            abbreviated_form: The abbreviated surface form (e.g. "E. coli").
            full_form:        The full canonical form (e.g. "Escherichia coli").
            added_by:         Identifier of the curator adding the mapping.

        Requirements: 11.5
        """
        if not abbreviated_form or not full_form:
            raise ValueError(
                "abbreviated_form and full_form must both be non-empty strings"
            )

        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO abbreviation_table
                            (abbreviated_form, full_form, added_by, added_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (abbreviated_form, full_form, added_by, now),
                    )
                # Hot-reload: update in-memory table immediately
                self._reload_table_locked()
            except Exception as exc:
                logger.error(
                    "add_mapping() failed for '%s' -> '%s': %s",
                    abbreviated_form,
                    full_form,
                    exc,
                )
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reload_table(self) -> None:
        """Reload the in-memory table from SQLite (acquires lock)."""
        with self._lock:
            self._reload_table_locked()

    def _reload_table_locked(self) -> None:
        """
        Reload the in-memory table from SQLite.

        Must be called while ``self._lock`` is already held.
        """
        rows = self._conn.execute(
            "SELECT abbreviated_form, full_form FROM abbreviation_table"
        ).fetchall()

        new_table: Dict[str, List[str]] = {}
        for row in rows:
            abbr = row["abbreviated_form"]
            full = row["full_form"]
            if abbr not in new_table:
                new_table[abbr] = []
            if full not in new_table[abbr]:
                new_table[abbr].append(full)

        self._table = new_table

    def _genus_initial_lookup(
        self, initial: str, species_epithet: str
    ) -> List[str]:
        """
        Return candidate binomial names for a genus-initial abbreviation.

        Iterates over all full forms stored in the in-memory table, extracts
        the genus (first word of each full form), and for every genus whose
        first character matches *initial*, constructs the candidate binomial
        ``genus + " " + species_epithet``.

        This means "E. coli" with genera "Escherichia" and "Enterococcus"
        (both starting with "E") yields candidates "Escherichia coli" and
        "Enterococcus coli".

        Must be called while ``self._lock`` is already held.
        """
        seen_genera: set = set()
        candidates: List[str] = []

        for full_forms in self._table.values():
            for full_form in full_forms:
                if not full_form:
                    continue
                # Extract the genus (first word of the full form)
                genus = full_form.split()[0]
                if genus and genus[0].upper() == initial and genus not in seen_genera:
                    seen_genera.add(genus)
                    # Construct the candidate binomial
                    candidate = f"{genus} {species_epithet}"
                    candidates.append(candidate)

        return sorted(candidates)

    @staticmethod
    def _build_candidates(full_forms: List[str]) -> List[ExpansionCandidate]:
        """
        Build a sorted list of :class:`ExpansionCandidate` objects.

        Confidence = 1.0 / N where N is the number of candidates.
        Results are sorted lexicographically by ``expanded_form``.

        Requirements: 11.2, 11.4
        """
        sorted_forms = sorted(full_forms)
        n = len(sorted_forms)
        confidence = 1.0 / n if n > 0 else 0.0
        return [
            ExpansionCandidate(expanded_form=form, confidence=confidence)
            for form in sorted_forms
        ]

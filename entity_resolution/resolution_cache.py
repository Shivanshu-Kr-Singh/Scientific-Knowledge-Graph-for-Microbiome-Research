"""
ResolutionCache — two-tier cache (in-memory LRU + SQLite persistent) with
version-based invalidation.

SLAs:
  - In-memory hit:  ≤10ms
  - SQLite hit:     ≤100ms

Cache validity: an entry is valid iff ``entry.registry_version == current_registry_version``.
Invalid entries are treated as cache misses.

Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 2.5
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_resolution_cache_schema,
)
from entity_resolution.models import CacheEntry, ResolutionResult
from entity_resolution.utils import normalize_surface_form

logger = logging.getLogger(__name__)


class ResolutionCache:
    """
    Two-tier cache: in-memory LRU (default 10,000 entries) + SQLite persistent.

    The in-memory tier is an :class:`collections.OrderedDict` used as an LRU
    cache.  The most-recently-used entry is moved to the end on every access;
    when capacity is exceeded the oldest entry (front of the dict) is evicted.

    The SQLite tier stores serialised :class:`ResolutionResult` JSON alongside
    the ``registry_version`` so that stale entries can be detected and evicted
    on version advance.

    Cache key: NFC-normalised, lowercased surface form (via
    :func:`~entity_resolution.utils.normalize_surface_form`).

    Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 2.5
    """

    def __init__(
        self,
        capacity: int = 10_000,
        db_path: str = "resolution_cache.db",
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """
        Initialise the cache.

        Args:
            capacity: Maximum number of entries in the in-memory LRU tier.
            db_path:  Path to the SQLite cache database file.
                      Ignored when *conn* is provided (useful for tests).
            conn:     An already-open :class:`sqlite3.Connection`.  When
                      supplied the cache uses this connection directly and
                      does **not** open *db_path*.  The caller is responsible
                      for closing the connection.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self._capacity = capacity
        # OrderedDict used as LRU: most-recently-used at the end.
        # Values are CacheEntry instances.
        self._lru: OrderedDict[str, CacheEntry] = OrderedDict()

        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._owns_conn = True
            create_schema_in_connection(self._conn, get_resolution_cache_schema())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        surface_form: str,
        current_registry_version: int,
    ) -> Optional[ResolutionResult]:
        """
        Look up a cached resolution result.

        Checks the in-memory LRU tier first (≤10ms SLA), then the SQLite
        persistent tier (≤100ms SLA).  Returns ``None`` on a cache miss or
        when the cached entry's ``registry_version`` does not match
        ``current_registry_version``.

        Args:
            surface_form:             The raw surface form to look up.
            current_registry_version: The current version of the
                                      :class:`CanonicalRegistry`.

        Returns:
            The cached :class:`ResolutionResult`, or ``None`` on miss /
            version mismatch.

        Requirements: 8.3
        """
        key = normalize_surface_form(surface_form)

        # --- Tier 1: in-memory LRU ---
        entry = self._lru.get(key)
        if entry is not None:
            if entry.registry_version == current_registry_version:
                # Move to end (most-recently-used)
                self._lru.move_to_end(key)
                return entry.resolution_result
            else:
                # Stale — evict from memory tier
                del self._lru[key]

        # --- Tier 2: SQLite ---
        try:
            row = self._conn.execute(
                "SELECT resolution_result_json, cache_timestamp, registry_version "
                "FROM cache_entries WHERE surface_form_normalized = ?",
                (key,),
            ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.error("ResolutionCache.get() SQLite read failed: %s", exc)
            return None

        if row is None:
            return None

        if row["registry_version"] != current_registry_version:
            # Stale SQLite entry — do not promote to memory tier
            return None

        # Deserialise and promote to memory tier
        try:
            result = ResolutionResult(**json.loads(row["resolution_result_json"]))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ResolutionCache.get() failed to deserialise result for key=%r: %s",
                key,
                exc,
            )
            return None

        cache_entry = CacheEntry(
            surface_form=surface_form,
            resolution_result=result,
            cache_timestamp=datetime.fromisoformat(row["cache_timestamp"]),
            registry_version=row["registry_version"],
        )
        self._lru_put(key, cache_entry)
        return result

    def put(
        self,
        surface_form: str,
        result: ResolutionResult,
        registry_version: int,
    ) -> None:
        """
        Store a resolution result in both the in-memory LRU and SQLite tiers.

        Args:
            surface_form:     The raw surface form (will be normalised as key).
            result:           The :class:`ResolutionResult` to cache.
            registry_version: The current :class:`CanonicalRegistry` version.

        Requirements: 8.4, 8.6
        """
        key = normalize_surface_form(surface_form)
        now = datetime.now(timezone.utc)

        entry = CacheEntry(
            surface_form=surface_form,
            resolution_result=result,
            cache_timestamp=now,
            registry_version=registry_version,
        )

        # --- Tier 1: in-memory LRU ---
        self._lru_put(key, entry)

        # --- Tier 2: SQLite ---
        try:
            result_json = result.model_dump_json()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                    (surface_form_normalized, resolution_result_json,
                     cache_timestamp, registry_version)
                VALUES (?, ?, ?, ?)
                """,
                (key, result_json, now.isoformat(), registry_version),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ResolutionCache.put() SQLite write failed for key=%r: %s",
                key,
                exc,
            )

    def invalidate_version(self, old_version: int) -> int:
        """
        Remove all cache entries created under *old_version* from both tiers.

        Returns the total count of invalidated entries (memory + SQLite,
        counting each unique key once).

        Args:
            old_version: The registry version whose entries should be evicted.

        Returns:
            Number of entries invalidated.

        Requirements: 8.5, 2.5
        """
        # --- Tier 1: in-memory LRU ---
        stale_keys = [
            k for k, v in self._lru.items() if v.registry_version == old_version
        ]
        for k in stale_keys:
            del self._lru[k]
        memory_count = len(stale_keys)

        # --- Tier 2: SQLite ---
        sqlite_count = 0
        try:
            cursor = self._conn.execute(
                "DELETE FROM cache_entries WHERE registry_version = ?",
                (old_version,),
            )
            self._conn.commit()
            sqlite_count = cursor.rowcount
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ResolutionCache.invalidate_version() SQLite delete failed "
                "for version=%d: %s",
                old_version,
                exc,
            )

        # Return the total unique entries invalidated.
        # Memory entries that were also in SQLite are counted once (from SQLite).
        # Memory-only entries (not yet flushed or already evicted from SQLite)
        # are counted separately.
        # To avoid double-counting: SQLite is the source of truth for persisted
        # entries; memory-only entries are those whose keys were NOT in SQLite.
        # However, since put() always writes to both tiers, the counts should
        # be consistent.  We return the larger of the two to be safe, but in
        # practice they should be equal.
        return max(memory_count, sqlite_count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lru_put(self, key: str, entry: CacheEntry) -> None:
        """
        Insert or update *key* in the in-memory LRU dict, evicting the
        least-recently-used entry if capacity is exceeded.
        """
        if key in self._lru:
            # Update existing entry and move to end (most-recently-used)
            self._lru[key] = entry
            self._lru.move_to_end(key)
        else:
            if len(self._lru) >= self._capacity:
                # Evict the least-recently-used entry (front of OrderedDict)
                self._lru.popitem(last=False)
            self._lru[key] = entry

    def close(self) -> None:
        """Close the underlying database connection if owned by this instance."""
        if self._owns_conn:
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

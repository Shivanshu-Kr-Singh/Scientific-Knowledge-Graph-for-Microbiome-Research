"""
SQLite schema initialisation module for the Deterministic Entity Resolution Pipeline.

Creates and initialises three physically separate SQLite databases:
  - canonical_registry.db  : canonical entities, synonyms, abbreviations, overrides
  - resolution_cache.db    : two-tier resolution cache entries
  - resolution_audit.db    : resolution audit records and metrics snapshots

Requirements: 3.1, 7.4, 8.2
"""

import os
import sqlite3
from typing import Dict


# ---------------------------------------------------------------------------
# Schema DDL strings
# ---------------------------------------------------------------------------

_CANONICAL_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS canonical_entities (
    canonical_id TEXT PRIMARY KEY,
    primary_name TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('taxon', 'disease', 'method')),
    ontology_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS synonyms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id TEXT NOT NULL REFERENCES canonical_entities(canonical_id),
    surface_form TEXT NOT NULL,
    surface_form_normalized TEXT NOT NULL,
    provenance TEXT NOT NULL CHECK(provenance IN ('ontology', 'paper_text', 'curator')),
    added_by TEXT,
    added_at TEXT NOT NULL,
    UNIQUE(surface_form_normalized)
);
CREATE INDEX IF NOT EXISTS idx_synonyms_normalized ON synonyms(surface_form_normalized);
CREATE INDEX IF NOT EXISTS idx_synonyms_canonical_id ON synonyms(canonical_id);

CREATE TABLE IF NOT EXISTS abbreviation_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    abbreviated_form TEXT NOT NULL,
    full_form TEXT NOT NULL,
    added_by TEXT NOT NULL,
    added_at TEXT NOT NULL,
    UNIQUE(abbreviated_form, full_form)
);
CREATE INDEX IF NOT EXISTS idx_abbrev_form ON abbreviation_table(abbreviated_form);

CREATE TABLE IF NOT EXISTS manual_overrides (
    surface_form TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    curator_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    justification TEXT
);

CREATE TABLE IF NOT EXISTS synonym_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    duplicate_surface_form TEXT NOT NULL,
    entity_a_id TEXT NOT NULL,
    entity_b_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    provenance_source TEXT
);

CREATE TABLE IF NOT EXISTS registry_version (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    version INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO registry_version(id, version) VALUES (1, 0);
"""

_RESOLUTION_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    surface_form_normalized TEXT PRIMARY KEY,
    resolution_result_json  TEXT NOT NULL,
    cache_timestamp         TEXT NOT NULL,
    registry_version        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_version ON cache_entries(registry_version);
"""

_RESOLUTION_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS resolution_records (
    record_id TEXT PRIMARY KEY,
    surface_form TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    winning_strategy TEXT NOT NULL,
    canonical_id TEXT,
    grounding_confidence REAL NOT NULL,
    conflict_set_json TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    high_conflict INTEGER NOT NULL DEFAULT 0,
    hierarchy_match INTEGER NOT NULL DEFAULT 0,
    hierarchy_level INTEGER,
    curator_override TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_surface_form ON resolution_records(surface_form);
CREATE INDEX IF NOT EXISTS idx_audit_canonical_id ON resolution_records(canonical_id);
CREATE INDEX IF NOT EXISTS idx_audit_paper_id ON resolution_records(paper_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON resolution_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_strategy ON resolution_records(winning_strategy);

CREATE TABLE IF NOT EXISTS metrics_snapshots (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    paper_ids_json TEXT NOT NULL,
    total_forms INTEGER NOT NULL,
    resolved_count INTEGER NOT NULL,
    unresolved_count INTEGER NOT NULL,
    resolution_rate REAL NOT NULL,
    per_strategy_counts_json TEXT NOT NULL,
    entity_type_metrics_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_snapshots(timestamp);
"""


# ---------------------------------------------------------------------------
# Public schema accessors
# ---------------------------------------------------------------------------

def get_canonical_registry_schema() -> str:
    """Return the DDL SQL for canonical_registry.db as a string."""
    return _CANONICAL_REGISTRY_SCHEMA


def get_resolution_cache_schema() -> str:
    """Return the DDL SQL for resolution_cache.db as a string."""
    return _RESOLUTION_CACHE_SCHEMA


def get_resolution_audit_schema() -> str:
    """Return the DDL SQL for resolution_audit.db as a string."""
    return _RESOLUTION_AUDIT_SCHEMA


# ---------------------------------------------------------------------------
# Schema application helper
# ---------------------------------------------------------------------------

def create_schema_in_connection(conn: sqlite3.Connection, schema_sql: str) -> None:
    """
    Apply a schema SQL string to an existing SQLite connection.

    Executes each statement in *schema_sql* separated by semicolons.
    Useful for in-memory test databases so tests do not touch the filesystem.

    Args:
        conn: An open :class:`sqlite3.Connection`.
        schema_sql: One or more DDL statements separated by semicolons.

    Raises:
        sqlite3.Error: If any DDL statement fails.
    """
    conn.executescript(schema_sql)
    conn.commit()


# ---------------------------------------------------------------------------
# Main initialisation function
# ---------------------------------------------------------------------------

def create_all_schemas(base_path: str) -> Dict[str, str]:
    """
    Create the three SQLite database files and apply all DDL.

    Creates *base_path* (and any missing parent directories) if it does not
    already exist, then opens (or creates) the three database files and
    applies the full schema DDL to each one.

    Args:
        base_path: Directory in which the ``.db`` files will be created.

    Returns:
        A dict mapping logical database names to their absolute file paths::

            {
                "canonical_registry": "/abs/path/canonical_registry.db",
                "resolution_cache":   "/abs/path/resolution_cache.db",
                "resolution_audit":   "/abs/path/resolution_audit.db",
            }

    Raises:
        OSError: If the directory cannot be created.
        sqlite3.Error: If any DDL statement fails.
    """
    # Resolve to an absolute path so callers always get absolute paths back.
    base_path = os.path.abspath(base_path)
    os.makedirs(base_path, exist_ok=True)

    db_configs = [
        ("canonical_registry", "canonical_registry.db", _CANONICAL_REGISTRY_SCHEMA),
        ("resolution_cache",   "resolution_cache.db",   _RESOLUTION_CACHE_SCHEMA),
        ("resolution_audit",   "resolution_audit.db",   _RESOLUTION_AUDIT_SCHEMA),
    ]

    paths: Dict[str, str] = {}

    for logical_name, filename, schema_sql in db_configs:
        db_path = os.path.join(base_path, filename)
        conn = sqlite3.connect(db_path)
        try:
            create_schema_in_connection(conn, schema_sql)
        finally:
            conn.close()
        paths[logical_name] = db_path

    return paths

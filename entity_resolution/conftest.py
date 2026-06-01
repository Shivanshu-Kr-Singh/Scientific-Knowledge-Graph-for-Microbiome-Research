"""
Shared pytest configuration and fixtures for the entity_resolution package.

Sets up Hypothesis profiles and provides in-memory SQLite database fixtures
for all three databases used by the pipeline.
"""

from __future__ import annotations

import os
import sqlite3

import pytest
from hypothesis import HealthCheck, settings

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------

settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "dev",
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load the profile specified by the HYPOTHESIS_PROFILE env var, defaulting to "ci"
_profile = os.environ.get("HYPOTHESIS_PROFILE", "ci")
settings.load_profile(_profile)


# ---------------------------------------------------------------------------
# SQLite DDL helpers
# ---------------------------------------------------------------------------

_CANONICAL_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS canonical_entities (
    canonical_id    TEXT PRIMARY KEY,
    primary_name    TEXT NOT NULL,
    entity_type     TEXT NOT NULL CHECK(entity_type IN ('taxon','disease','method')),
    ontology_source TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS synonyms (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id            TEXT NOT NULL REFERENCES canonical_entities(canonical_id),
    surface_form            TEXT NOT NULL,
    surface_form_normalized TEXT NOT NULL,
    provenance              TEXT NOT NULL CHECK(provenance IN ('ontology','paper_text','curator')),
    added_by                TEXT,
    added_at                TEXT NOT NULL,
    UNIQUE(surface_form_normalized)
);

CREATE INDEX IF NOT EXISTS idx_synonyms_normalized   ON synonyms(surface_form_normalized);
CREATE INDEX IF NOT EXISTS idx_synonyms_canonical_id ON synonyms(canonical_id);

CREATE TABLE IF NOT EXISTS manual_overrides (
    surface_form TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    entity_type  TEXT NOT NULL,
    curator_id   TEXT NOT NULL,
    justification TEXT CHECK(length(justification) <= 500),
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS abbreviation_table (
    abbreviated_form TEXT NOT NULL,
    full_form        TEXT NOT NULL,
    added_by         TEXT NOT NULL,
    added_at         TEXT NOT NULL,
    PRIMARY KEY (abbreviated_form, full_form)
);

CREATE TABLE IF NOT EXISTS synonym_conflicts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    duplicate_form TEXT NOT NULL,
    entity_a_id    TEXT NOT NULL,
    entity_b_id    TEXT NOT NULL,
    conflict_at    TEXT NOT NULL,
    provenance_src TEXT
);

CREATE TABLE IF NOT EXISTS registry_version (
    id      INTEGER PRIMARY KEY CHECK(id = 1),
    version INTEGER NOT NULL DEFAULT 1
);

INSERT OR IGNORE INTO registry_version (id, version) VALUES (1, 1);
"""

_RESOLUTION_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS cache_entries (
    surface_form_normalized TEXT PRIMARY KEY,
    resolution_result_json  TEXT NOT NULL,
    cache_timestamp         TEXT NOT NULL,
    registry_version        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cache_version ON cache_entries(registry_version);
"""

_RESOLUTION_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS resolution_records (
    record_id            TEXT PRIMARY KEY,
    surface_form         TEXT NOT NULL,
    entity_type          TEXT NOT NULL,
    timestamp            TEXT NOT NULL,
    winning_strategy     TEXT NOT NULL,
    canonical_id         TEXT,
    grounding_confidence REAL NOT NULL,
    conflict_set_json    TEXT NOT NULL,
    paper_id             TEXT NOT NULL,
    high_conflict        INTEGER NOT NULL DEFAULT 0,
    hierarchy_match      INTEGER NOT NULL DEFAULT 0,
    hierarchy_level      INTEGER,
    curator_override     TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_surface_form ON resolution_records(surface_form);
CREATE INDEX IF NOT EXISTS idx_audit_canonical_id ON resolution_records(canonical_id);
CREATE INDEX IF NOT EXISTS idx_audit_strategy     ON resolution_records(winning_strategy);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp    ON resolution_records(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_paper_id     ON resolution_records(paper_id);

CREATE TABLE IF NOT EXISTS metrics_snapshots (
    run_id                   TEXT PRIMARY KEY,
    timestamp                TEXT NOT NULL,
    paper_ids_json           TEXT NOT NULL,
    total_forms              INTEGER NOT NULL,
    resolved_count           INTEGER NOT NULL,
    unresolved_count         INTEGER NOT NULL,
    resolution_rate          REAL NOT NULL,
    per_strategy_counts_json TEXT NOT NULL,
    entity_type_metrics_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_snapshots(timestamp);
"""


def _apply_ddl(conn: sqlite3.Connection, ddl: str) -> None:
    """Execute a multi-statement DDL script on the given connection."""
    conn.executescript(ddl)
    conn.commit()


# ---------------------------------------------------------------------------
# Shared pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_registry_db() -> sqlite3.Connection:
    """
    Return a sqlite3 connection to an in-memory database with the
    canonical_registry schema applied.

    The connection is closed automatically after the test.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _apply_ddl(conn, _CANONICAL_REGISTRY_DDL)
    yield conn
    conn.close()


@pytest.fixture
def in_memory_cache_db() -> sqlite3.Connection:
    """
    Return a sqlite3 connection to an in-memory database with the
    resolution_cache schema applied.

    The connection is closed automatically after the test.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _apply_ddl(conn, _RESOLUTION_CACHE_DDL)
    yield conn
    conn.close()


@pytest.fixture
def in_memory_audit_db() -> sqlite3.Connection:
    """
    Return a sqlite3 connection to an in-memory database with the
    resolution_audit schema applied.

    The connection is closed automatically after the test.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _apply_ddl(conn, _RESOLUTION_AUDIT_DDL)
    yield conn
    conn.close()

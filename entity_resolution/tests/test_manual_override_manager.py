"""
Unit tests for ManualOverrideManager (task 11.1).

Tests cover:
- get_override(): returns None on miss, returns ManualOverride on hit
- set_override(): validates canonical_id format, validates justification length,
                  INSERT/REPLACE semantics, cache invalidation
- remove_override(): DELETE row, cache invalidation
- bulk_import_csv(): valid rows imported, malformed rows skipped with logging,
                     BulkImportResult counts are correct

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.7, 9.8
"""

from __future__ import annotations

import sqlite3
import textwrap
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest

from entity_resolution.db_schema import (
    create_schema_in_connection,
    get_canonical_registry_schema,
)
from entity_resolution.manual_override_manager import ManualOverrideManager
from entity_resolution.models import ManualOverride


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_conn() -> sqlite3.Connection:
    """In-memory SQLite connection with the canonical_registry schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_schema_in_connection(conn, get_canonical_registry_schema())
    yield conn
    conn.close()


@pytest.fixture
def manager(registry_conn: sqlite3.Connection) -> ManualOverrideManager:
    """Fresh ManualOverrideManager backed by an in-memory database."""
    return ManualOverrideManager(conn=registry_conn)


@pytest.fixture
def mock_cache() -> MagicMock:
    """A mock ResolutionCache for verifying cache invalidation calls."""
    cache = MagicMock()
    cache.invalidate_version = MagicMock(return_value=0)
    return cache


@pytest.fixture
def manager_with_cache(
    registry_conn: sqlite3.Connection, mock_cache: MagicMock
) -> ManualOverrideManager:
    """ManualOverrideManager wired with a mock ResolutionCache."""
    return ManualOverrideManager(conn=registry_conn, resolution_cache=mock_cache)


# ---------------------------------------------------------------------------
# get_override() — Requirements 9.1, 9.2
# ---------------------------------------------------------------------------


def test_get_override_miss_returns_none(manager: ManualOverrideManager) -> None:
    """get_override() returns None when no override exists for the surface form."""
    result = manager.get_override("E. coli")
    assert result is None


def test_get_override_empty_string_returns_none(manager: ManualOverrideManager) -> None:
    """get_override() returns None for an empty surface form."""
    result = manager.get_override("")
    assert result is None


def test_get_override_returns_override_after_set(manager: ManualOverrideManager) -> None:
    """get_override() returns the ManualOverride after set_override() is called."""
    ok, err = manager.set_override(
        surface_form="E. coli",
        canonical_id="562",
        entity_type="taxon",
        curator_id="curator1",
        justification="Confirmed by expert",
    )
    assert ok is True, f"set_override failed: {err}"

    result = manager.get_override("E. coli")
    assert result is not None
    assert isinstance(result, ManualOverride)
    assert result.surface_form == "E. coli"
    assert result.canonical_id == "562"
    assert result.entity_type == "taxon"
    assert result.curator_id == "curator1"
    assert result.justification == "Confirmed by expert"


def test_get_override_case_sensitive(manager: ManualOverrideManager) -> None:
    """get_override() lookup is case-sensitive (surface forms stored as-is)."""
    manager.set_override("E. coli", "562", "taxon", "curator1")
    # Different case should not match
    assert manager.get_override("e. coli") is None
    assert manager.get_override("E. COLI") is None
    # Exact case should match
    assert manager.get_override("E. coli") is not None


def test_get_override_returns_all_fields(manager: ManualOverrideManager) -> None:
    """get_override() returns a ManualOverride with all fields populated."""
    manager.set_override(
        surface_form="Crohn's disease",
        canonical_id="D003424",
        entity_type="disease",
        curator_id="curator2",
        justification="Standard MeSH mapping",
    )
    result = manager.get_override("Crohn's disease")
    assert result is not None
    assert result.surface_form == "Crohn's disease"
    assert result.canonical_id == "D003424"
    assert result.entity_type == "disease"
    assert result.curator_id == "curator2"
    assert result.justification == "Standard MeSH mapping"
    assert isinstance(result.timestamp, datetime)


def test_get_override_no_justification(manager: ManualOverrideManager) -> None:
    """get_override() works when justification is None."""
    manager.set_override("16S rRNA", "METHOD-16S", "method", "curator3")
    result = manager.get_override("16S rRNA")
    assert result is not None
    assert result.justification is None


# ---------------------------------------------------------------------------
# set_override() — Requirements 9.3, 9.5
# ---------------------------------------------------------------------------


def test_set_override_valid_taxon(manager: ManualOverrideManager) -> None:
    """set_override() succeeds for a valid taxon canonical_id."""
    ok, err = manager.set_override("E. coli", "562", "taxon", "curator1")
    assert ok is True
    assert err is None


def test_set_override_valid_disease(manager: ManualOverrideManager) -> None:
    """set_override() succeeds for a valid disease canonical_id."""
    ok, err = manager.set_override("Crohn disease", "D003424", "disease", "curator1")
    assert ok is True
    assert err is None


def test_set_override_valid_method(manager: ManualOverrideManager) -> None:
    """set_override() succeeds for a valid method canonical_id."""
    ok, err = manager.set_override("16S sequencing", "METHOD-16S", "method", "curator1")
    assert ok is True
    assert err is None


@pytest.mark.parametrize(
    "bad_id,entity_type",
    [
        ("0", "taxon"),
        ("-1", "taxon"),
        ("01", "taxon"),
        ("abc", "taxon"),
        ("d006262", "disease"),
        ("D", "disease"),
        ("1D06262", "disease"),
        ("METHOD-", "method"),
        ("method-16S", "method"),
        ("METH-16S", "method"),
    ],
)
def test_set_override_invalid_canonical_id_rejected(
    manager: ManualOverrideManager, bad_id: str, entity_type: str
) -> None:
    """set_override() rejects invalid canonical_id formats."""
    ok, err = manager.set_override("some form", bad_id, entity_type, "curator1")
    assert ok is False
    assert err is not None
    # Confirm nothing was persisted
    assert manager.get_override("some form") is None


def test_set_override_justification_exactly_500_chars_accepted(
    manager: ManualOverrideManager,
) -> None:
    """set_override() accepts justification of exactly 500 characters."""
    justification = "x" * 500
    ok, err = manager.set_override("E. coli", "562", "taxon", "curator1", justification)
    assert ok is True
    assert err is None


def test_set_override_justification_501_chars_rejected(
    manager: ManualOverrideManager,
) -> None:
    """set_override() rejects justification exceeding 500 characters."""
    justification = "x" * 501
    ok, err = manager.set_override("E. coli", "562", "taxon", "curator1", justification)
    assert ok is False
    assert err is not None
    assert "500" in err


def test_set_override_empty_surface_form_rejected(manager: ManualOverrideManager) -> None:
    """set_override() rejects an empty surface_form."""
    ok, err = manager.set_override("", "562", "taxon", "curator1")
    assert ok is False
    assert err is not None


def test_set_override_empty_curator_id_rejected(manager: ManualOverrideManager) -> None:
    """set_override() rejects an empty curator_id."""
    ok, err = manager.set_override("E. coli", "562", "taxon", "")
    assert ok is False
    assert err is not None


def test_set_override_replace_existing(manager: ManualOverrideManager) -> None:
    """set_override() replaces an existing override for the same surface form."""
    manager.set_override("E. coli", "562", "taxon", "curator1")
    # Replace with a different canonical_id
    ok, err = manager.set_override("E. coli", "1301", "taxon", "curator2", "Updated")
    assert ok is True
    assert err is None

    result = manager.get_override("E. coli")
    assert result is not None
    assert result.canonical_id == "1301"
    assert result.curator_id == "curator2"
    assert result.justification == "Updated"


def test_set_override_records_timestamp(manager: ManualOverrideManager) -> None:
    """set_override() records a UTC timestamp (Requirement 9.3)."""
    before = datetime.now(timezone.utc)
    manager.set_override("E. coli", "562", "taxon", "curator1")
    after = datetime.now(timezone.utc)

    result = manager.get_override("E. coli")
    assert result is not None
    ts = result.timestamp
    # Make timezone-aware for comparison if needed
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    assert before <= ts <= after


def test_set_override_invalidates_cache(
    manager_with_cache: ManualOverrideManager, mock_cache: MagicMock
) -> None:
    """set_override() calls cache.invalidate_version() (Requirement 9.5)."""
    manager_with_cache.set_override("E. coli", "562", "taxon", "curator1")
    mock_cache.invalidate_version.assert_called()


# ---------------------------------------------------------------------------
# remove_override() — Requirement 9.5
# ---------------------------------------------------------------------------


def test_remove_override_deletes_existing(manager: ManualOverrideManager) -> None:
    """remove_override() deletes an existing override."""
    manager.set_override("E. coli", "562", "taxon", "curator1")
    assert manager.get_override("E. coli") is not None

    result = manager.remove_override("E. coli")
    assert result is True
    assert manager.get_override("E. coli") is None


def test_remove_override_nonexistent_is_noop(manager: ManualOverrideManager) -> None:
    """remove_override() is a no-op (returns True) when no override exists."""
    result = manager.remove_override("nonexistent surface form")
    assert result is True


def test_remove_override_empty_string_is_noop(manager: ManualOverrideManager) -> None:
    """remove_override() returns True for an empty surface form."""
    result = manager.remove_override("")
    assert result is True


def test_remove_override_invalidates_cache(
    manager_with_cache: ManualOverrideManager, mock_cache: MagicMock
) -> None:
    """remove_override() calls cache.invalidate_version() (Requirement 9.5)."""
    manager_with_cache.set_override("E. coli", "562", "taxon", "curator1")
    mock_cache.invalidate_version.reset_mock()

    manager_with_cache.remove_override("E. coli")
    mock_cache.invalidate_version.assert_called()


def test_remove_override_only_removes_target(manager: ManualOverrideManager) -> None:
    """remove_override() only removes the specified surface form."""
    manager.set_override("E. coli", "562", "taxon", "curator1")
    manager.set_override("Crohn disease", "D003424", "disease", "curator1")

    manager.remove_override("E. coli")

    assert manager.get_override("E. coli") is None
    assert manager.get_override("Crohn disease") is not None


# ---------------------------------------------------------------------------
# bulk_import_csv() — Requirements 9.7, 9.8
# ---------------------------------------------------------------------------


def _make_csv(*rows: dict) -> str:
    """Build a CSV string from a list of row dicts."""
    header = "surface_form,canonical_id,entity_type,curator_id,justification"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.get('surface_form', '')},"
            f"{row.get('canonical_id', '')},"
            f"{row.get('entity_type', '')},"
            f"{row.get('curator_id', '')},"
            f"{row.get('justification', '')}"
        )
    return "\n".join(lines)


def test_bulk_import_all_valid_rows(manager: ManualOverrideManager) -> None:
    """bulk_import_csv() imports all valid rows and reports correct counts."""
    csv_content = _make_csv(
        {"surface_form": "E. coli", "canonical_id": "562", "entity_type": "taxon",
         "curator_id": "curator1", "justification": "Confirmed"},
        {"surface_form": "Crohn disease", "canonical_id": "D003424", "entity_type": "disease",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "16S rRNA", "canonical_id": "METHOD-16S", "entity_type": "method",
         "curator_id": "curator2", "justification": "Standard method"},
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.total_rows == 3
    assert result.imported_count == 3
    assert result.skipped_count == 0
    assert result.skipped_rows == []

    # Verify overrides were persisted
    assert manager.get_override("E. coli") is not None
    assert manager.get_override("Crohn disease") is not None
    assert manager.get_override("16S rRNA") is not None


def test_bulk_import_skips_invalid_canonical_id(manager: ManualOverrideManager) -> None:
    """bulk_import_csv() skips rows with invalid canonical_id format (Req 9.8)."""
    csv_content = _make_csv(
        {"surface_form": "E. coli", "canonical_id": "INVALID", "entity_type": "taxon",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "Crohn disease", "canonical_id": "D003424", "entity_type": "disease",
         "curator_id": "curator1", "justification": ""},
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.total_rows == 2
    assert result.imported_count == 1
    assert result.skipped_count == 1
    assert len(result.skipped_rows) == 1
    assert result.skipped_rows[0]["row_number"] == 2  # first data row


def test_bulk_import_skips_missing_required_columns(manager: ManualOverrideManager) -> None:
    """bulk_import_csv() skips rows with missing required field values (Req 9.8)."""
    # Row with empty surface_form
    csv_content = _make_csv(
        {"surface_form": "", "canonical_id": "562", "entity_type": "taxon",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "E. coli", "canonical_id": "562", "entity_type": "taxon",
         "curator_id": "curator1", "justification": ""},
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.total_rows == 2
    assert result.imported_count == 1
    assert result.skipped_count == 1


def test_bulk_import_skips_duplicate_conflict(manager: ManualOverrideManager) -> None:
    """
    bulk_import_csv() skips rows where the surface_form already has an override
    for a different canonical_id (Req 9.8).
    """
    # Pre-set an override for "E. coli" -> "562"
    manager.set_override("E. coli", "562", "taxon", "curator1")

    # CSV tries to set "E. coli" -> "1301" (different canonical_id)
    csv_content = _make_csv(
        {"surface_form": "E. coli", "canonical_id": "1301", "entity_type": "taxon",
         "curator_id": "curator2", "justification": "Conflict"},
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.total_rows == 1
    assert result.imported_count == 0
    assert result.skipped_count == 1
    assert "conflict" in result.skipped_rows[0]["reason"].lower() or \
           "already has" in result.skipped_rows[0]["reason"].lower()

    # Original override should be unchanged
    override = manager.get_override("E. coli")
    assert override is not None
    assert override.canonical_id == "562"


def test_bulk_import_allows_same_canonical_id_duplicate(manager: ManualOverrideManager) -> None:
    """
    bulk_import_csv() allows re-importing the same surface_form → canonical_id
    mapping (idempotent, not a conflict).
    """
    manager.set_override("E. coli", "562", "taxon", "curator1")

    # CSV re-imports the same mapping
    csv_content = _make_csv(
        {"surface_form": "E. coli", "canonical_id": "562", "entity_type": "taxon",
         "curator_id": "curator2", "justification": "Re-confirmed"},
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.total_rows == 1
    assert result.imported_count == 1
    assert result.skipped_count == 0


def test_bulk_import_continues_after_skipped_row(manager: ManualOverrideManager) -> None:
    """
    bulk_import_csv() continues processing after a skipped row without aborting
    (Req 9.8).
    """
    csv_content = _make_csv(
        {"surface_form": "E. coli", "canonical_id": "INVALID", "entity_type": "taxon",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "Crohn disease", "canonical_id": "D003424", "entity_type": "disease",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "16S rRNA", "canonical_id": "METHOD-16S", "entity_type": "method",
         "curator_id": "curator2", "justification": ""},
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.total_rows == 3
    assert result.imported_count == 2
    assert result.skipped_count == 1

    # The two valid rows should be persisted
    assert manager.get_override("Crohn disease") is not None
    assert manager.get_override("16S rRNA") is not None
    # The invalid row should not be persisted
    assert manager.get_override("E. coli") is None


def test_bulk_import_reports_skipped_row_numbers(manager: ManualOverrideManager) -> None:
    """bulk_import_csv() reports the row number for each skipped row (Req 9.8)."""
    csv_content = _make_csv(
        {"surface_form": "E. coli", "canonical_id": "INVALID", "entity_type": "taxon",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "Crohn disease", "canonical_id": "D003424", "entity_type": "disease",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "bad method", "canonical_id": "METHOD-", "entity_type": "method",
         "curator_id": "curator2", "justification": ""},
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.skipped_count == 2
    skipped_row_numbers = {r["row_number"] for r in result.skipped_rows}
    assert 2 in skipped_row_numbers  # row 2 = first data row (header is row 1)
    assert 4 in skipped_row_numbers  # row 4 = third data row


def test_bulk_import_empty_csv(manager: ManualOverrideManager) -> None:
    """bulk_import_csv() handles an empty CSV (header only) gracefully."""
    csv_content = "surface_form,canonical_id,entity_type,curator_id,justification\n"
    result = manager.bulk_import_csv(csv_content)

    assert result.total_rows == 0
    assert result.imported_count == 0
    assert result.skipped_count == 0


def test_bulk_import_missing_header_column(manager: ManualOverrideManager) -> None:
    """bulk_import_csv() handles CSV with missing header columns gracefully."""
    # Missing 'justification' column in header
    csv_content = "surface_form,canonical_id,entity_type,curator_id\nE. coli,562,taxon,curator1"
    result = manager.bulk_import_csv(csv_content)

    # Should report the header issue
    assert result.imported_count == 0


def test_bulk_import_skips_justification_too_long(manager: ManualOverrideManager) -> None:
    """bulk_import_csv() skips rows where justification exceeds 500 characters."""
    long_justification = "x" * 501
    csv_content = (
        "surface_form,canonical_id,entity_type,curator_id,justification\n"
        f"E. coli,562,taxon,curator1,{long_justification}\n"
        "Crohn disease,D003424,disease,curator1,OK\n"
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.total_rows == 2
    assert result.imported_count == 1
    assert result.skipped_count == 1
    assert manager.get_override("E. coli") is None
    assert manager.get_override("Crohn disease") is not None


def test_bulk_import_result_counts_match(manager: ManualOverrideManager) -> None:
    """BulkImportResult: imported_count + skipped_count == total_rows (Req 9.7)."""
    csv_content = _make_csv(
        {"surface_form": "E. coli", "canonical_id": "562", "entity_type": "taxon",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "bad", "canonical_id": "INVALID", "entity_type": "taxon",
         "curator_id": "curator1", "justification": ""},
        {"surface_form": "Crohn disease", "canonical_id": "D003424", "entity_type": "disease",
         "curator_id": "curator1", "justification": ""},
    )
    result = manager.bulk_import_csv(csv_content)

    assert result.imported_count + result.skipped_count == result.total_rows


# ---------------------------------------------------------------------------
# Integration: set → get → remove cycle
# ---------------------------------------------------------------------------


def test_full_lifecycle(manager: ManualOverrideManager) -> None:
    """Full set → get → remove lifecycle works correctly."""
    # Set
    ok, err = manager.set_override("E. coli", "562", "taxon", "curator1", "Test")
    assert ok is True

    # Get
    override = manager.get_override("E. coli")
    assert override is not None
    assert override.canonical_id == "562"

    # Remove
    removed = manager.remove_override("E. coli")
    assert removed is True

    # Get after remove
    assert manager.get_override("E. coli") is None


def test_multiple_overrides_independent(manager: ManualOverrideManager) -> None:
    """Multiple overrides for different surface forms are independent."""
    manager.set_override("E. coli", "562", "taxon", "curator1")
    manager.set_override("Crohn disease", "D003424", "disease", "curator1")
    manager.set_override("16S rRNA", "METHOD-16S", "method", "curator2")

    assert manager.get_override("E. coli").canonical_id == "562"
    assert manager.get_override("Crohn disease").canonical_id == "D003424"
    assert manager.get_override("16S rRNA").canonical_id == "METHOD-16S"

    manager.remove_override("E. coli")
    assert manager.get_override("E. coli") is None
    assert manager.get_override("Crohn disease") is not None
    assert manager.get_override("16S rRNA") is not None

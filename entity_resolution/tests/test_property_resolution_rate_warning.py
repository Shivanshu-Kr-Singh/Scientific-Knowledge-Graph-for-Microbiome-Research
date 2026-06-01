"""
Property 14: Resolution Rate Warning Threshold

**Validates: Requirements 10.5**

For any pipeline run where at least one surface form was processed and any
entity type's resolution rate < 0.70, assert a warning is emitted to the
system log containing ``run_id``, entity type, observed rate, and the 0.70
threshold.

Requirements: 10.5
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import List

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from entity_resolution.conftest import _RESOLUTION_AUDIT_DDL, _apply_ddl
from entity_resolution.models import ResolutionResult
from entity_resolution.resolution_metrics import ResolutionMetrics

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-empty run_id strings (printable ASCII, no whitespace-only)
_run_id_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=40,
).filter(lambda s: len(s.strip()) >= 1)

# Entity types supported by the pipeline
_entity_type_st = st.sampled_from(["taxon", "disease", "method"])

# Total surface forms processed: at least 1
_total_count_st = st.integers(min_value=1, max_value=50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_in_memory_audit_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the resolution_audit schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _apply_ddl(conn, _RESOLUTION_AUDIT_DDL)
    return conn


def _make_resolution_result(
    entity_type: str,
    grounded: bool,
    confidence: float = 1.0,
) -> ResolutionResult:
    """Build a minimal ResolutionResult for metrics recording."""
    return ResolutionResult(
        surface_form="test_surface_form",
        entity_type=entity_type,
        canonical_id="1" if grounded else None,
        grounded=grounded,
        winning_strategy="exact" if grounded else "none",
        grounding_confidence=confidence if grounded else 0.0,
        conflict_set=[],
        paper_id="paper-1",
        timestamp=datetime.now(timezone.utc),
    )


class _ListHandler(logging.Handler):
    """A logging handler that collects formatted log messages into a list."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


# ---------------------------------------------------------------------------
# Property 14: Resolution Rate Warning Threshold
# **Validates: Requirements 10.5**
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    run_id=_run_id_st,
    entity_type=_entity_type_st,
    total_count=_total_count_st,
    resolved_count=st.integers(min_value=0, max_value=49),
)
def test_property_resolution_rate_warning_threshold(
    run_id: str,
    entity_type: str,
    total_count: int,
    resolved_count: int,
) -> None:
    """
    **Property 14: Resolution Rate Warning Threshold**

    **Validates: Requirements 10.5**

    For any pipeline run where at least one surface form was processed and
    any entity type's resolution rate < 0.70, assert a warning is emitted
    to the system log containing:
    - the run_id
    - the entity type
    - the observed rate (as a float)
    - the 0.70 threshold
    """
    # Constrain resolved_count to be within [0, total_count - 1] so that
    # resolved_count / total_count < 1.0, and use assume() to enforce < 0.70
    assume(resolved_count < total_count)
    observed_rate = resolved_count / total_count
    assume(observed_rate < 0.70)

    conn = _make_in_memory_audit_conn()
    try:
        metrics = ResolutionMetrics(conn=conn)

        # Record resolved_count grounded results
        for _ in range(resolved_count):
            result = _make_resolution_result(entity_type, grounded=True)
            metrics.record_resolution(result)

        # Record (total_count - resolved_count) ungrounded results
        unresolved_count = total_count - resolved_count
        for _ in range(unresolved_count):
            result = _make_resolution_result(entity_type, grounded=False)
            metrics.record_resolution(result)

        # Capture log warnings using a custom handler attached to the
        # resolution_metrics logger (avoids function-scoped fixture issues)
        metrics_logger = logging.getLogger("entity_resolution.resolution_metrics")
        handler = _ListHandler()
        handler.setLevel(logging.WARNING)
        metrics_logger.addHandler(handler)
        try:
            metrics.finalize_run(run_id=run_id, paper_ids=["paper-1"])
        finally:
            metrics_logger.removeHandler(handler)

        warning_messages = handler.messages

        # Assert that at least one warning was emitted
        assert len(warning_messages) >= 1, (
            f"Expected at least one WARNING log entry, but none were emitted. "
            f"run_id={run_id!r}, entity_type={entity_type!r}, "
            f"total_count={total_count}, resolved_count={resolved_count}, "
            f"observed_rate={observed_rate:.4f}"
        )

        # Find the warning that matches our entity type
        matching_warnings = [
            msg for msg in warning_messages
            if entity_type in msg
        ]
        assert len(matching_warnings) >= 1, (
            f"Expected a warning containing entity_type={entity_type!r}, "
            f"but none found. Warnings emitted: {warning_messages!r}. "
            f"run_id={run_id!r}, total_count={total_count}, "
            f"resolved_count={resolved_count}, observed_rate={observed_rate:.4f}"
        )

        # Check the matching warning contains all required fields
        warning_text = matching_warnings[0]

        # Assert run_id is present
        assert run_id in warning_text, (
            f"Warning does not contain run_id={run_id!r}. "
            f"Warning text: {warning_text!r}"
        )

        # Assert entity_type is present (already checked above, but be explicit)
        assert entity_type in warning_text, (
            f"Warning does not contain entity_type={entity_type!r}. "
            f"Warning text: {warning_text!r}"
        )

        # Assert the observed rate is present (formatted as a float)
        # The implementation formats it as "%.4f" so check for the rate value
        rate_str = f"{observed_rate:.4f}"
        assert rate_str in warning_text, (
            f"Warning does not contain observed_rate={rate_str!r}. "
            f"Warning text: {warning_text!r}. "
            f"run_id={run_id!r}, entity_type={entity_type!r}, "
            f"total_count={total_count}, resolved_count={resolved_count}"
        )

        # Assert the 0.70 threshold is present
        assert "0.70" in warning_text, (
            f"Warning does not contain the 0.70 threshold. "
            f"Warning text: {warning_text!r}. "
            f"run_id={run_id!r}, entity_type={entity_type!r}"
        )

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unit tests — explicit examples
# ---------------------------------------------------------------------------


def test_warning_emitted_for_zero_resolution_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Explicit example: 0% resolution rate triggers a warning.

    All 10 surface forms are unresolved → rate = 0.0 < 0.70.
    """
    conn = _make_in_memory_audit_conn()
    try:
        metrics = ResolutionMetrics(conn=conn)

        for _ in range(10):
            metrics.record_resolution(_make_resolution_result("taxon", grounded=False))

        with caplog.at_level(logging.WARNING, logger="entity_resolution.resolution_metrics"):
            metrics.finalize_run(run_id="run-zero", paper_ids=["paper-1"])

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("run-zero" in m for m in warning_messages), (
            f"Expected warning with run_id='run-zero'. Got: {warning_messages!r}"
        )
        assert any("taxon" in m for m in warning_messages)
        assert any("0.70" in m for m in warning_messages)

    finally:
        conn.close()


def test_no_warning_when_rate_at_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Explicit example: exactly 70% resolution rate does NOT trigger a warning.

    7 resolved out of 10 → rate = 0.70, which is NOT < 0.70.
    """
    conn = _make_in_memory_audit_conn()
    try:
        metrics = ResolutionMetrics(conn=conn)

        for _ in range(7):
            metrics.record_resolution(_make_resolution_result("disease", grounded=True))
        for _ in range(3):
            metrics.record_resolution(_make_resolution_result("disease", grounded=False))

        with caplog.at_level(logging.WARNING, logger="entity_resolution.resolution_metrics"):
            metrics.finalize_run(run_id="run-at-threshold", paper_ids=["paper-1"])

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("run-at-threshold" in m for m in warning_messages), (
            f"Unexpected warning at exactly 70% rate. Got: {warning_messages!r}"
        )

    finally:
        conn.close()


def test_no_warning_when_rate_above_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Explicit example: 80% resolution rate does NOT trigger a warning.

    8 resolved out of 10 → rate = 0.80 >= 0.70.
    """
    conn = _make_in_memory_audit_conn()
    try:
        metrics = ResolutionMetrics(conn=conn)

        for _ in range(8):
            metrics.record_resolution(_make_resolution_result("method", grounded=True))
        for _ in range(2):
            metrics.record_resolution(_make_resolution_result("method", grounded=False))

        with caplog.at_level(logging.WARNING, logger="entity_resolution.resolution_metrics"):
            metrics.finalize_run(run_id="run-above-threshold", paper_ids=["paper-1"])

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("run-above-threshold" in m for m in warning_messages), (
            f"Unexpected warning at 80% rate. Got: {warning_messages!r}"
        )

    finally:
        conn.close()


def test_warning_contains_all_required_fields(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Explicit example: verify warning contains run_id, entity_type, rate, and threshold.

    3 resolved out of 10 → rate = 0.30 < 0.70.
    """
    conn = _make_in_memory_audit_conn()
    try:
        metrics = ResolutionMetrics(conn=conn)

        for _ in range(3):
            metrics.record_resolution(_make_resolution_result("taxon", grounded=True))
        for _ in range(7):
            metrics.record_resolution(_make_resolution_result("taxon", grounded=False))

        with caplog.at_level(logging.WARNING, logger="entity_resolution.resolution_metrics"):
            metrics.finalize_run(run_id="run-explicit-001", paper_ids=["paper-1"])

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_messages) >= 1, f"No warnings emitted. Got: {warning_messages!r}"

        warning_text = warning_messages[0]

        assert "run-explicit-001" in warning_text, (
            f"run_id not in warning: {warning_text!r}"
        )
        assert "taxon" in warning_text, (
            f"entity_type not in warning: {warning_text!r}"
        )
        # rate = 3/10 = 0.3000
        assert "0.3000" in warning_text, (
            f"observed_rate not in warning: {warning_text!r}"
        )
        assert "0.70" in warning_text, (
            f"threshold not in warning: {warning_text!r}"
        )

    finally:
        conn.close()


def test_warning_emitted_per_entity_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Explicit example: warnings are emitted for each entity type below threshold.

    taxon: 2/10 = 0.20 < 0.70 → warning
    disease: 9/10 = 0.90 >= 0.70 → no warning
    method: 1/5 = 0.20 < 0.70 → warning
    """
    conn = _make_in_memory_audit_conn()
    try:
        metrics = ResolutionMetrics(conn=conn)

        # taxon: 2 resolved, 8 unresolved
        for _ in range(2):
            metrics.record_resolution(_make_resolution_result("taxon", grounded=True))
        for _ in range(8):
            metrics.record_resolution(_make_resolution_result("taxon", grounded=False))

        # disease: 9 resolved, 1 unresolved
        for _ in range(9):
            metrics.record_resolution(_make_resolution_result("disease", grounded=True))
        for _ in range(1):
            metrics.record_resolution(_make_resolution_result("disease", grounded=False))

        # method: 1 resolved, 4 unresolved
        for _ in range(1):
            metrics.record_resolution(_make_resolution_result("method", grounded=True))
        for _ in range(4):
            metrics.record_resolution(_make_resolution_result("method", grounded=False))

        with caplog.at_level(logging.WARNING, logger="entity_resolution.resolution_metrics"):
            metrics.finalize_run(run_id="run-multi-type", paper_ids=["paper-1"])

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        # taxon should have a warning
        assert any("taxon" in m for m in warning_messages), (
            f"Expected warning for 'taxon'. Got: {warning_messages!r}"
        )

        # method should have a warning
        assert any("method" in m for m in warning_messages), (
            f"Expected warning for 'method'. Got: {warning_messages!r}"
        )

        # disease should NOT have a warning (rate = 0.90 >= 0.70)
        disease_warnings = [m for m in warning_messages if "disease" in m]
        assert len(disease_warnings) == 0, (
            f"Unexpected warning for 'disease' at 90% rate. Got: {disease_warnings!r}"
        )

    finally:
        conn.close()

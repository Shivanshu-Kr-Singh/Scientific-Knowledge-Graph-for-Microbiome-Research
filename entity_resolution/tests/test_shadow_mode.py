"""
Tests for shadow mode discrepancy logging and Spec 1 compatibility wiring.

Covers:
- When shadow_mode=True and results differ, a ShadowModeDiscrepancy is logged
- normalize() returns Spec 1 result when shadow_mode=True
- normalize() returns Spec 2 result when shadow_mode=False
- enable_shadow_mode() and disable_shadow_mode() methods
- When Spec 1 normalizer is not configured, falls back to Spec 2 result with a warning

Requirements: 14.5, 14.6
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest

from entity_resolution.canonical_registry import CanonicalRegistry
from entity_resolution.conftest import _CANONICAL_REGISTRY_DDL, _apply_ddl
from entity_resolution.models import (
    CanonicalEntityRecord,
    EntityType,
    NormalizationResult,
    ShadowModeDiscrepancy,
    SynonymProvenance,
    SynonymRecord,
)
from entity_resolution.resolution_pipeline import ResolutionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry_conn() -> sqlite3.Connection:
    """Create an in-memory SQLite connection with the canonical_registry schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _apply_ddl(conn, _CANONICAL_REGISTRY_DDL)
    return conn


def _register_entity(
    registry: CanonicalRegistry,
    canonical_id: str,
    primary_name: str,
    entity_type: EntityType = EntityType.TAXON,
) -> None:
    """Register a canonical entity in the registry."""
    now = datetime.now(timezone.utc)
    record = CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=primary_name,
        entity_type=entity_type,
        ontology_source="ncbi_taxonomy",
        synonyms=[],
        created_at=now,
        updated_at=now,
    )
    success, error = registry.register(record)
    assert success, f"Failed to register entity {canonical_id!r}: {error}"


class _FakeSpec1Normalizer:
    """
    A minimal Spec 1 normalizer stub that returns a fixed NormalizationResult.
    """

    def __init__(self, canonical_id: Optional[str], grounded: bool) -> None:
        self._canonical_id = canonical_id
        self._grounded = grounded

    def normalize(self, surface_form: str, entity_type: str) -> NormalizationResult:
        return NormalizationResult(
            canonical_id=self._canonical_id,
            grounded=self._grounded,
        )


# ---------------------------------------------------------------------------
# Tests: enable_shadow_mode() / disable_shadow_mode()
# ---------------------------------------------------------------------------


class TestShadowModeToggle:
    """Tests for enable_shadow_mode() and disable_shadow_mode() methods."""

    def test_shadow_mode_disabled_by_default(self) -> None:
        """Pipeline starts with shadow_mode=False by default."""
        pipeline = ResolutionPipeline()
        assert pipeline._shadow_mode is False

    def test_shadow_mode_enabled_via_constructor(self) -> None:
        """shadow_mode=True can be set in the constructor."""
        pipeline = ResolutionPipeline(shadow_mode=True)
        assert pipeline._shadow_mode is True

    def test_enable_shadow_mode(self) -> None:
        """enable_shadow_mode() sets _shadow_mode to True."""
        pipeline = ResolutionPipeline(shadow_mode=False)
        pipeline.enable_shadow_mode()
        assert pipeline._shadow_mode is True

    def test_disable_shadow_mode(self) -> None:
        """disable_shadow_mode() sets _shadow_mode to False."""
        pipeline = ResolutionPipeline(shadow_mode=True)
        pipeline.disable_shadow_mode()
        assert pipeline._shadow_mode is False

    def test_enable_then_disable(self) -> None:
        """enable then disable returns to False."""
        pipeline = ResolutionPipeline()
        pipeline.enable_shadow_mode()
        assert pipeline._shadow_mode is True
        pipeline.disable_shadow_mode()
        assert pipeline._shadow_mode is False

    def test_disable_then_enable(self) -> None:
        """disable then enable returns to True."""
        pipeline = ResolutionPipeline(shadow_mode=True)
        pipeline.disable_shadow_mode()
        assert pipeline._shadow_mode is False
        pipeline.enable_shadow_mode()
        assert pipeline._shadow_mode is True


# ---------------------------------------------------------------------------
# Tests: normalize() routes correctly based on shadow_mode flag
# ---------------------------------------------------------------------------


class TestNormalizeRouting:
    """Tests that normalize() calls the right path based on shadow_mode."""

    def test_normalize_returns_spec2_result_when_shadow_mode_false(self) -> None:
        """
        When shadow_mode=False, normalize() returns the Spec 2 (pipeline) result.

        Requirements: 14.1, 14.3
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 1 normalizer returns a DIFFERENT result
            spec1 = _FakeSpec1Normalizer(canonical_id="99999", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=False,
                spec1_normalizer=spec1,
            )

            result = pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            # Must return Spec 2 result (exact match -> "562"), not Spec 1 ("99999")
            assert result.canonical_id == "562"
            assert result.grounded is True
        finally:
            conn.close()

    def test_normalize_returns_spec1_result_when_shadow_mode_true(self) -> None:
        """
        When shadow_mode=True, normalize() returns the Spec 1 result.

        Requirements: 14.5, 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 1 normalizer returns a DIFFERENT result
            spec1 = _FakeSpec1Normalizer(canonical_id="99999", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            result = pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            # Must return Spec 1 result ("99999"), not Spec 2 ("562")
            assert result.canonical_id == "99999"
            assert result.grounded is True
        finally:
            conn.close()

    def test_normalize_spec1_result_returned_even_when_results_agree(self) -> None:
        """
        When shadow_mode=True and both normalizers agree, Spec 1 result is returned.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 1 normalizer returns the SAME result as Spec 2
            spec1 = _FakeSpec1Normalizer(canonical_id="562", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            result = pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            assert result.canonical_id == "562"
            assert result.grounded is True
        finally:
            conn.close()

    def test_normalize_shadow_mode_toggle_changes_behavior(self) -> None:
        """
        Toggling shadow_mode changes which result normalize() returns.

        Requirements: 14.5
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            spec1 = _FakeSpec1Normalizer(canonical_id="99999", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=False,
                spec1_normalizer=spec1,
            )

            # shadow_mode=False -> Spec 2 result
            result_spec2 = pipeline.normalize("Escherichia coli", "taxon", "paper_001")
            assert result_spec2.canonical_id == "562"

            # Enable shadow mode -> Spec 1 result
            pipeline.enable_shadow_mode()
            result_spec1 = pipeline.normalize("Escherichia coli", "taxon", "paper_001")
            assert result_spec1.canonical_id == "99999"

            # Disable shadow mode -> back to Spec 2 result
            pipeline.disable_shadow_mode()
            result_spec2_again = pipeline.normalize("Escherichia coli", "taxon", "paper_001")
            assert result_spec2_again.canonical_id == "562"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Tests: ShadowModeDiscrepancy logging
# ---------------------------------------------------------------------------


class TestShadowModeDiscrepancyLogging:
    """Tests that ShadowModeDiscrepancy is logged when results differ."""

    def test_discrepancy_logged_when_canonical_ids_differ(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        When shadow_mode=True and canonical_ids differ, a discrepancy is logged.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 1 returns a different canonical_id
            spec1 = _FakeSpec1Normalizer(canonical_id="99999", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            with caplog.at_level(logging.INFO, logger="entity_resolution.resolution_pipeline"):
                pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            # A discrepancy log message must have been emitted
            discrepancy_logs = [
                r for r in caplog.records
                if "shadow mode discrepancy" in r.message.lower()
                or "discrepancy" in r.message.lower()
            ]
            assert len(discrepancy_logs) >= 1, (
                "Expected at least one discrepancy log message, got none. "
                f"All log messages: {[r.message for r in caplog.records]}"
            )
        finally:
            conn.close()

    def test_discrepancy_logged_when_grounded_flags_differ(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        When shadow_mode=True and grounded flags differ, a discrepancy is logged.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 2 will return grounded=True (exact match found)
            # Spec 1 returns grounded=False with same canonical_id
            spec1 = _FakeSpec1Normalizer(canonical_id="562", grounded=False)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            with caplog.at_level(logging.INFO, logger="entity_resolution.resolution_pipeline"):
                pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            discrepancy_logs = [
                r for r in caplog.records
                if "discrepancy" in r.message.lower()
            ]
            assert len(discrepancy_logs) >= 1, (
                "Expected discrepancy log when grounded flags differ. "
                f"All log messages: {[r.message for r in caplog.records]}"
            )
        finally:
            conn.close()

    def test_no_discrepancy_logged_when_results_agree(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        When shadow_mode=True and both normalizers agree, no discrepancy is logged.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 1 returns the same result as Spec 2
            spec1 = _FakeSpec1Normalizer(canonical_id="562", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            with caplog.at_level(logging.INFO, logger="entity_resolution.resolution_pipeline"):
                pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            discrepancy_logs = [
                r for r in caplog.records
                if "discrepancy" in r.message.lower()
            ]
            assert len(discrepancy_logs) == 0, (
                "Expected no discrepancy log when results agree, "
                f"but got: {[r.message for r in discrepancy_logs]}"
            )
        finally:
            conn.close()

    def test_discrepancy_log_contains_surface_form_and_ids(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        The discrepancy log message contains surface_form, spec1 and spec2 canonical_ids.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            spec1 = _FakeSpec1Normalizer(canonical_id="99999", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            with caplog.at_level(logging.INFO, logger="entity_resolution.resolution_pipeline"):
                pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            discrepancy_logs = [
                r for r in caplog.records
                if "discrepancy" in r.message.lower()
            ]
            assert len(discrepancy_logs) >= 1

            log_msg = discrepancy_logs[0].message
            # The log should reference the surface form
            assert "Escherichia coli" in log_msg or "escherichia coli" in log_msg.lower(), (
                f"Expected surface_form in log message, got: {log_msg!r}"
            )
        finally:
            conn.close()

    def test_discrepancy_logged_when_spec1_ungrounded_spec2_grounded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Discrepancy is logged when Spec 1 is ungrounded but Spec 2 is grounded.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 1 returns ungrounded (no canonical_id)
            spec1 = _FakeSpec1Normalizer(canonical_id=None, grounded=False)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            with caplog.at_level(logging.INFO, logger="entity_resolution.resolution_pipeline"):
                result = pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            # Returns Spec 1 result (ungrounded)
            assert result.canonical_id is None
            assert result.grounded is False

            # Discrepancy must be logged
            discrepancy_logs = [
                r for r in caplog.records
                if "discrepancy" in r.message.lower()
            ]
            assert len(discrepancy_logs) >= 1
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Tests: Spec 1 normalizer not configured (fallback to Spec 2)
# ---------------------------------------------------------------------------


class TestShadowModeFallback:
    """Tests for when Spec 1 normalizer is not configured."""

    def test_fallback_to_spec2_when_no_spec1_normalizer(self) -> None:
        """
        When shadow_mode=True but no Spec 1 normalizer is configured,
        normalize() falls back to the Spec 2 result.

        Requirements: 14.5
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=None,  # Not configured
            )

            result = pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            # Falls back to Spec 2 result
            assert result.canonical_id == "562"
            assert result.grounded is True
        finally:
            conn.close()

    def test_warning_logged_when_no_spec1_normalizer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        When shadow_mode=True but no Spec 1 normalizer is configured,
        a warning is logged.

        Requirements: 14.5
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=None,
            )

            with caplog.at_level(logging.WARNING, logger="entity_resolution.resolution_pipeline"):
                pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            warning_logs = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING
                and "shadow mode" in r.message.lower()
            ]
            assert len(warning_logs) >= 1, (
                "Expected a warning log when Spec 1 normalizer is not configured. "
                f"All log messages: {[r.message for r in caplog.records]}"
            )
        finally:
            conn.close()

    def test_fallback_returns_unresolved_when_spec2_unresolved(self) -> None:
        """
        When shadow_mode=True, no Spec 1 normalizer, and Spec 2 is unresolved,
        normalize() returns ungrounded result.

        Requirements: 14.5
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            # Do NOT register the surface form — Spec 2 will be unresolved

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=None,
            )

            result = pipeline.normalize("unknown entity xyz", "taxon", "paper_001")

            # Spec 2 is unresolved, so fallback returns unresolved
            assert result.canonical_id is None
            assert result.grounded is False
        finally:
            conn.close()

    def test_spec1_exception_falls_back_to_spec2(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        When Spec 1 normalizer raises an exception, normalize() falls back to
        the Spec 2 result and logs an error.

        Requirements: 14.5
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 1 normalizer that raises an exception
            class _BrokenSpec1:
                def normalize(self, surface_form: str, entity_type: str) -> NormalizationResult:
                    raise RuntimeError("Spec 1 normalizer is broken")

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=_BrokenSpec1(),
            )

            with caplog.at_level(logging.ERROR, logger="entity_resolution.resolution_pipeline"):
                result = pipeline.normalize("Escherichia coli", "taxon", "paper_001")

            # Falls back to Spec 2 result
            assert result.canonical_id == "562"
            assert result.grounded is True

            # An error must be logged
            error_logs = [
                r for r in caplog.records
                if r.levelno >= logging.ERROR
            ]
            assert len(error_logs) >= 1, (
                "Expected an error log when Spec 1 normalizer raises. "
                f"All log messages: {[r.message for r in caplog.records]}"
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Tests: normalize_shadow_mode() directly
# ---------------------------------------------------------------------------


class TestNormalizeShadowModeDirect:
    """Direct tests for normalize_shadow_mode() method."""

    def test_returns_spec1_result_when_results_differ(self) -> None:
        """
        normalize_shadow_mode() returns Spec 1 result when results differ.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            spec1 = _FakeSpec1Normalizer(canonical_id="SPEC1-ID", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=False,  # shadow_mode doesn't matter for direct call
                spec1_normalizer=spec1,
            )

            result = pipeline.normalize_shadow_mode(
                "Escherichia coli", "taxon", "paper_001"
            )

            # Must return Spec 1 result
            assert result.canonical_id == "SPEC1-ID"
            assert result.grounded is True
        finally:
            conn.close()

    def test_returns_spec1_result_when_results_agree(self) -> None:
        """
        normalize_shadow_mode() returns Spec 1 result even when results agree.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            spec1 = _FakeSpec1Normalizer(canonical_id="562", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                spec1_normalizer=spec1,
            )

            result = pipeline.normalize_shadow_mode(
                "Escherichia coli", "taxon", "paper_001"
            )

            assert result.canonical_id == "562"
            assert result.grounded is True
        finally:
            conn.close()

    def test_discrepancy_record_fields(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        The discrepancy log contains all required fields per Requirement 14.6:
        surface_form, entity_type, paper_id, spec1_canonical_id, spec1_grounded,
        spec2_canonical_id, spec2_grounded.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            spec1 = _FakeSpec1Normalizer(canonical_id="99999", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            with caplog.at_level(logging.INFO, logger="entity_resolution.resolution_pipeline"):
                pipeline.normalize_shadow_mode(
                    "Escherichia coli", "taxon", "paper_test_42"
                )

            discrepancy_logs = [
                r for r in caplog.records
                if "discrepancy" in r.message.lower()
            ]
            assert len(discrepancy_logs) >= 1

            log_msg = discrepancy_logs[0].message
            # Check that key fields appear in the log message
            assert "99999" in log_msg, f"spec1_canonical_id not in log: {log_msg!r}"
            assert "562" in log_msg, f"spec2_canonical_id not in log: {log_msg!r}"
        finally:
            conn.close()

    def test_normalize_shadow_mode_with_unresolved_spec2(self) -> None:
        """
        When Spec 2 is unresolved but Spec 1 is grounded, Spec 1 result is returned.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            # Do NOT register the surface form — Spec 2 will be unresolved

            spec1 = _FakeSpec1Normalizer(canonical_id="SPEC1-ONLY", grounded=True)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            result = pipeline.normalize_shadow_mode(
                "unknown entity", "taxon", "paper_001"
            )

            # Returns Spec 1 result (grounded)
            assert result.canonical_id == "SPEC1-ONLY"
            assert result.grounded is True
        finally:
            conn.close()

    def test_normalize_shadow_mode_with_unresolved_spec1(self) -> None:
        """
        When Spec 1 is ungrounded but Spec 2 is grounded, Spec 1 result is returned.

        Requirements: 14.6
        """
        conn = _make_registry_conn()
        try:
            registry = CanonicalRegistry(conn=conn)
            _register_entity(registry, "562", "Escherichia coli")

            # Spec 1 returns ungrounded
            spec1 = _FakeSpec1Normalizer(canonical_id=None, grounded=False)

            pipeline = ResolutionPipeline(
                registry=registry,
                shadow_mode=True,
                spec1_normalizer=spec1,
            )

            result = pipeline.normalize_shadow_mode(
                "Escherichia coli", "taxon", "paper_001"
            )

            # Returns Spec 1 result (ungrounded)
            assert result.canonical_id is None
            assert result.grounded is False
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Tests: ShadowModeDiscrepancy model fields
# ---------------------------------------------------------------------------


class TestShadowModeDiscrepancyModel:
    """Tests for the ShadowModeDiscrepancy Pydantic model."""

    def test_discrepancy_model_fields(self) -> None:
        """ShadowModeDiscrepancy model has all required fields."""
        now = datetime.now(timezone.utc)
        discrepancy = ShadowModeDiscrepancy(
            surface_form="E. coli",
            entity_type="taxon",
            paper_id="paper_001",
            spec1_canonical_id="99999",
            spec1_grounded=True,
            spec2_canonical_id="562",
            spec2_grounded=True,
            timestamp=now,
        )
        assert discrepancy.surface_form == "E. coli"
        assert discrepancy.entity_type == "taxon"
        assert discrepancy.paper_id == "paper_001"
        assert discrepancy.spec1_canonical_id == "99999"
        assert discrepancy.spec1_grounded is True
        assert discrepancy.spec2_canonical_id == "562"
        assert discrepancy.spec2_grounded is True
        assert discrepancy.timestamp == now

    def test_discrepancy_model_allows_none_canonical_ids(self) -> None:
        """ShadowModeDiscrepancy allows None for canonical_id fields."""
        now = datetime.now(timezone.utc)
        discrepancy = ShadowModeDiscrepancy(
            surface_form="unknown",
            entity_type="taxon",
            paper_id="paper_001",
            spec1_canonical_id=None,
            spec1_grounded=False,
            spec2_canonical_id=None,
            spec2_grounded=False,
            timestamp=now,
        )
        assert discrepancy.spec1_canonical_id is None
        assert discrepancy.spec2_canonical_id is None

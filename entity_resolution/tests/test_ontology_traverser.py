"""
Tests for OntologyTraverser — NCBI Taxonomy / MeSH hierarchy traversal.

Covers:
  - Unit tests: confidence formula, graceful degradation, 3-level limit,
    returns [] when no ancestor in registry.
  - Integration-style tests using mocked HTTP responses.
  - Property 8: Ontology Traversal Confidence Formula
    **Validates: Requirements 13.3**

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import requests
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.canonical_registry import CanonicalRegistry
from entity_resolution.models import (
    CanonicalEntityRecord,
    EntityType,
    SynonymProvenance,
    SynonymRecord,
)
from entity_resolution.ontology_traverser import OntologyCandidate, OntologyTraverser


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_registry(conn: sqlite3.Connection) -> CanonicalRegistry:
    """Return a CanonicalRegistry backed by the given in-memory connection."""
    return CanonicalRegistry(conn=conn)


def _register_taxon(registry: CanonicalRegistry, canonical_id: str, name: str) -> None:
    """Register a minimal taxon entity in the registry."""
    record = CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=name,
        entity_type=EntityType.TAXON,
        ontology_source="ncbi_taxonomy",
        synonyms=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    success, err = registry.register(record)
    assert success, f"Failed to register taxon '{canonical_id}': {err}"


def _register_disease(registry: CanonicalRegistry, canonical_id: str, name: str) -> None:
    """Register a minimal disease entity in the registry."""
    record = CanonicalEntityRecord(
        canonical_id=canonical_id,
        primary_name=name,
        entity_type=EntityType.DISEASE,
        ontology_source="mesh",
        synonyms=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    success, err = registry.register(record)
    assert success, f"Failed to register disease '{canonical_id}': {err}"


@pytest.fixture
def in_memory_registry_db() -> sqlite3.Connection:
    """In-memory SQLite connection with canonical_registry schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from entity_resolution.db_schema import (
        create_schema_in_connection,
        get_canonical_registry_schema,
    )
    create_schema_in_connection(conn, get_canonical_registry_schema())
    yield conn
    conn.close()


@pytest.fixture
def registry(in_memory_registry_db: sqlite3.Connection) -> CanonicalRegistry:
    """A fresh CanonicalRegistry backed by an in-memory database."""
    return _make_registry(in_memory_registry_db)


@pytest.fixture
def traverser() -> OntologyTraverser:
    """A fresh OntologyTraverser instance."""
    return OntologyTraverser()


# ---------------------------------------------------------------------------
# Unit tests — compute_confidence()
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    """Tests for OntologyTraverser.compute_confidence() static method."""

    def test_level_1_parent(self) -> None:
        """Level 1 (parent) yields confidence 0.50."""
        assert OntologyTraverser.compute_confidence(1) == pytest.approx(0.50)

    def test_level_2_grandparent(self) -> None:
        """Level 2 (grandparent) yields confidence 0.40."""
        assert OntologyTraverser.compute_confidence(2) == pytest.approx(0.40)

    def test_level_3_great_grandparent(self) -> None:
        """Level 3 (great-grandparent) yields confidence 0.30."""
        assert OntologyTraverser.compute_confidence(3) == pytest.approx(0.30)

    def test_formula_correctness(self) -> None:
        """Formula: 0.50 - (level - 1) * 0.10 for all valid levels."""
        for level in (1, 2, 3):
            expected = 0.50 - (level - 1) * 0.10
            result = OntologyTraverser.compute_confidence(level)
            assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12), (
                f"compute_confidence({level}) = {result}, expected {expected}"
            )

    def test_confidence_decreases_with_level(self) -> None:
        """Confidence decreases as hierarchy level increases."""
        c1 = OntologyTraverser.compute_confidence(1)
        c2 = OntologyTraverser.compute_confidence(2)
        c3 = OntologyTraverser.compute_confidence(3)
        assert c1 > c2 > c3


# ---------------------------------------------------------------------------
# Unit tests — graceful degradation (Requirement 13.1)
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Tests that traverse() returns [] and logs a warning when the service is unavailable."""

    def test_connection_error_returns_empty(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """ConnectionError from requests → returns [] without raising."""
        with patch("entity_resolution.ontology_traverser.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("refused")
            result = traverser.traverse("Escherichia coli", "taxon", registry)
        assert result == []

    def test_timeout_error_returns_empty(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """Timeout from requests → returns [] without raising."""
        with patch("entity_resolution.ontology_traverser.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("timed out")
            result = traverser.traverse("Crohn disease", "disease", registry)
        assert result == []

    def test_http_error_returns_empty(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """HTTP 500 error → returns [] without raising."""
        with patch("entity_resolution.ontology_traverser.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "500 Server Error"
            )
            mock_get.return_value = mock_response
            result = traverser.traverse("Escherichia coli", "taxon", registry)
        assert result == []

    def test_graceful_degradation_logs_warning(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry, caplog
    ) -> None:
        """A warning is logged when the service is unavailable."""
        import logging

        with patch("entity_resolution.ontology_traverser.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("refused")
            with caplog.at_level(logging.WARNING, logger="entity_resolution.ontology_traverser"):
                result = traverser.traverse("Escherichia coli", "taxon", registry)

        assert result == []
        assert any("unavailable" in record.message or "OntologyTraverser" in record.message
                   for record in caplog.records)

    def test_unsupported_entity_type_returns_empty(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """Unsupported entity_type (e.g. 'method') returns [] without calling HTTP."""
        with patch("entity_resolution.ontology_traverser.requests.get") as mock_get:
            result = traverser.traverse("PCR", "method", registry)
        assert result == []
        mock_get.assert_not_called()

    def test_empty_surface_form_returns_empty(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """Empty surface_form returns [] immediately."""
        result = traverser.traverse("", "taxon", registry)
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests — 3-level traversal limit (Requirement 13.2)
# ---------------------------------------------------------------------------


class TestThreeLevelLimit:
    """Tests that traversal stops at 3 levels."""

    def test_returns_empty_when_no_ancestor_in_registry_within_3_levels(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """
        When the registry has no ancestor within 3 levels, returns [].

        Requirement 13.6
        """
        # Registry is empty — no ancestors will match
        with patch.object(
            traverser,
            "_ncbi_esearch",
            return_value="562",
        ), patch.object(
            traverser,
            "_ncbi_fetch_lineage",
            return_value=["1224", "1236", "91347"],  # 3 ancestors, none in registry
        ):
            result = traverser.traverse("Escherichia coli", "taxon", registry)

        assert result == []

    def test_only_checks_up_to_3_levels(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """
        Even if more than 3 ancestors are returned, only the first 3 are checked.

        Requirement 13.2
        """
        # Register the 4th ancestor — should NOT be found
        _register_taxon(registry, "1", "root")

        checked_ids: List[str] = []
        original_lookup = registry.lookup_by_canonical_id

        def tracking_lookup(cid: str):
            checked_ids.append(cid)
            return original_lookup(cid)

        registry.lookup_by_canonical_id = tracking_lookup  # type: ignore[method-assign]

        with patch.object(
            traverser,
            "_ncbi_esearch",
            return_value="562",
        ), patch.object(
            traverser,
            "_ncbi_fetch_lineage",
            # _ncbi_fetch_lineage already limits to _MAX_LEVELS=3
            return_value=["1224", "1236", "91347"],
        ):
            result = traverser.traverse("Escherichia coli", "taxon", registry)

        # At most 3 IDs should have been checked
        assert len(checked_ids) <= 3
        assert result == []

    def test_returns_match_at_level_1(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """
        When the parent (level 1) is in the registry, returns it with
        hierarchy_level=1 and confidence=0.50.

        Requirements: 13.3, 13.4
        """
        # Register the parent taxon
        _register_taxon(registry, "561", "Escherichia")

        with patch.object(
            traverser,
            "_ncbi_esearch",
            return_value="562",
        ), patch.object(
            traverser,
            "_ncbi_fetch_lineage",
            return_value=["561", "543", "91347"],  # parent=561 is in registry
        ):
            result = traverser.traverse("Escherichia coli", "taxon", registry)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.canonical_id == "561"
        assert candidate.hierarchy_level == 1
        assert candidate.grounding_confidence == pytest.approx(0.50)

    def test_returns_match_at_level_2(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """
        When only the grandparent (level 2) is in the registry, returns it
        with hierarchy_level=2 and confidence=0.40.

        Requirements: 13.3, 13.4
        """
        # Register only the grandparent
        _register_taxon(registry, "543", "Enterobacteriaceae")

        with patch.object(
            traverser,
            "_ncbi_esearch",
            return_value="562",
        ), patch.object(
            traverser,
            "_ncbi_fetch_lineage",
            return_value=["561", "543", "91347"],  # grandparent=543 is in registry
        ):
            result = traverser.traverse("Escherichia coli", "taxon", registry)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.canonical_id == "543"
        assert candidate.hierarchy_level == 2
        assert candidate.grounding_confidence == pytest.approx(0.40)

    def test_returns_match_at_level_3(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """
        When only the great-grandparent (level 3) is in the registry, returns
        it with hierarchy_level=3 and confidence=0.30.

        Requirements: 13.3, 13.4
        """
        # Register only the great-grandparent
        _register_taxon(registry, "91347", "Enterobacterales")

        with patch.object(
            traverser,
            "_ncbi_esearch",
            return_value="562",
        ), patch.object(
            traverser,
            "_ncbi_fetch_lineage",
            return_value=["561", "543", "91347"],  # great-grandparent=91347 is in registry
        ):
            result = traverser.traverse("Escherichia coli", "taxon", registry)

        assert len(result) == 1
        candidate = result[0]
        assert candidate.canonical_id == "91347"
        assert candidate.hierarchy_level == 3
        assert candidate.grounding_confidence == pytest.approx(0.30)

    def test_returns_nearest_ancestor_first(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """
        When multiple ancestors are in the registry, returns the nearest one
        (lowest hierarchy level).

        Requirements: 13.2, 13.4
        """
        # Register both parent and grandparent
        _register_taxon(registry, "561", "Escherichia")
        _register_taxon(registry, "543", "Enterobacteriaceae")

        with patch.object(
            traverser,
            "_ncbi_esearch",
            return_value="562",
        ), patch.object(
            traverser,
            "_ncbi_fetch_lineage",
            return_value=["561", "543", "91347"],
        ):
            result = traverser.traverse("Escherichia coli", "taxon", registry)

        # Should return the parent (level 1), not the grandparent
        assert len(result) == 1
        assert result[0].canonical_id == "561"
        assert result[0].hierarchy_level == 1

    def test_returns_at_most_one_candidate(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """
        traverse() returns at most one candidate (the nearest ancestor).

        Requirements: 13.4 (postcondition)
        """
        _register_taxon(registry, "561", "Escherichia")
        _register_taxon(registry, "543", "Enterobacteriaceae")
        _register_taxon(registry, "91347", "Enterobacterales")

        with patch.object(
            traverser,
            "_ncbi_esearch",
            return_value="562",
        ), patch.object(
            traverser,
            "_ncbi_fetch_lineage",
            return_value=["561", "543", "91347"],
        ):
            result = traverser.traverse("Escherichia coli", "taxon", registry)

        assert len(result) <= 1


# ---------------------------------------------------------------------------
# Unit tests — MeSH traversal
# ---------------------------------------------------------------------------


class TestMeshTraversal:
    """Tests for MeSH disease hierarchy traversal."""

    def test_mesh_returns_empty_when_no_descriptor_found(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """Returns [] when MeSH lookup finds no descriptor."""
        with patch.object(traverser, "_mesh_lookup_descriptor", return_value=None):
            result = traverser.traverse("unknown disease", "disease", registry)
        assert result == []

    def test_mesh_returns_empty_when_no_ancestors(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """Returns [] when MeSH returns no ancestors."""
        with patch.object(
            traverser, "_mesh_lookup_descriptor", return_value="D006262"
        ), patch.object(traverser, "_mesh_fetch_ancestors", return_value=[]):
            result = traverser.traverse("Hypertension", "disease", registry)
        assert result == []

    def test_mesh_returns_match_at_level_1(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """Returns parent match with hierarchy_level=1 and confidence=0.50."""
        _register_disease(registry, "D002318", "Cardiovascular Diseases")

        with patch.object(
            traverser, "_mesh_lookup_descriptor", return_value="D006262"
        ), patch.object(
            traverser,
            "_mesh_fetch_ancestors",
            return_value=["D002318", "D009422"],
        ):
            result = traverser.traverse("Hypertension", "disease", registry)

        assert len(result) == 1
        assert result[0].canonical_id == "D002318"
        assert result[0].hierarchy_level == 1
        assert result[0].grounding_confidence == pytest.approx(0.50)

    def test_mesh_connection_error_returns_empty(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """ConnectionError during MeSH lookup → returns [] gracefully."""
        with patch("entity_resolution.ontology_traverser.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("refused")
            result = traverser.traverse("Hypertension", "disease", registry)
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests — OntologyCandidate model
# ---------------------------------------------------------------------------


class TestOntologyCandidate:
    """Tests for the OntologyCandidate Pydantic model."""

    def test_valid_candidate(self) -> None:
        """OntologyCandidate accepts valid fields."""
        candidate = OntologyCandidate(
            canonical_id="561",
            hierarchy_level=1,
            grounding_confidence=0.50,
        )
        assert candidate.canonical_id == "561"
        assert candidate.hierarchy_level == 1
        assert candidate.grounding_confidence == pytest.approx(0.50)

    def test_confidence_bounds(self) -> None:
        """grounding_confidence must be in [0.0, 1.0]."""
        # Valid boundary values
        OntologyCandidate(canonical_id="1", hierarchy_level=1, grounding_confidence=0.0)
        OntologyCandidate(canonical_id="1", hierarchy_level=1, grounding_confidence=1.0)

        # Invalid: above 1.0
        with pytest.raises(Exception):
            OntologyCandidate(canonical_id="1", hierarchy_level=1, grounding_confidence=1.1)

        # Invalid: below 0.0
        with pytest.raises(Exception):
            OntologyCandidate(canonical_id="1", hierarchy_level=1, grounding_confidence=-0.1)


# ---------------------------------------------------------------------------
# Unit tests — NCBI esearch returns no ID
# ---------------------------------------------------------------------------


class TestNcbiNoId:
    """Tests for the case where NCBI esearch returns no taxonomy ID."""

    def test_no_ncbi_id_returns_empty(
        self, traverser: OntologyTraverser, registry: CanonicalRegistry
    ) -> None:
        """Returns [] when NCBI esearch finds no taxonomy ID."""
        with patch.object(traverser, "_ncbi_esearch", return_value=None):
            result = traverser.traverse("unknown organism xyz", "taxon", registry)
        assert result == []


# ---------------------------------------------------------------------------
# Property 8: Ontology Traversal Confidence Formula
# **Validates: Requirements 13.3**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(hierarchy_level=st.sampled_from([1, 2, 3]))
def test_property_ontology_traversal_confidence_formula(hierarchy_level: int) -> None:
    """
    **Property 8: Ontology Traversal Confidence Formula**

    **Validates: Requirements 13.3**

    For each hierarchy_level ∈ {1, 2, 3}:
    - OntologyTraverser.compute_confidence(N) equals 0.50 - (N-1) * 0.10
      within floating-point tolerance.
    """
    result = OntologyTraverser.compute_confidence(hierarchy_level)

    # Compute expected value using the formula from Requirement 13.3
    expected = 0.50 - (hierarchy_level - 1) * 0.10

    # Assert formula correctness within floating-point tolerance
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"compute_confidence({hierarchy_level}) = {result}, expected {expected}"
    )

"""
graph/test_rollback_manager.py
-------------------------------
Unit tests for rollback manager.

Tests Requirements: 10.5, 19.4
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch

from graph.rollback_manager import RollbackManager, rollback_extraction_method
from graph.audit_log import AuditLog


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def audit_log(temp_db):
    """Create an AuditLog instance with temporary database."""
    return AuditLog(db_path=temp_db)


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = Mock()
    session = Mock()
    driver.session.return_value.__enter__ = Mock(return_value=session)
    driver.session.return_value.__exit__ = Mock(return_value=False)
    return driver


@pytest.fixture
def rollback_manager(mock_neo4j_driver, audit_log):
    """Create a RollbackManager instance with mocked Neo4j."""
    with patch('graph.rollback_manager.GraphDatabase.driver', return_value=mock_neo4j_driver):
        manager = RollbackManager(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            audit_log=audit_log,
        )
        manager.driver = mock_neo4j_driver
        yield manager


def test_get_relationships_to_rollback(rollback_manager, audit_log):
    """
    Test getting relationships that would be rolled back.
    
    Requirement 19.3: Query relationships by extraction method version
    """
    # Log some relationships
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={"direction": "increased"},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_1",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_2",
        target_id="taxon_2",
        properties={"direction": "decreased"},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_2",
    )
    
    # Get relationships to rollback
    relationships = rollback_manager.get_relationships_to_rollback(
        "llm_extractor_v1.2", "1.2"
    )
    
    assert len(relationships) == 2
    assert relationships[0]["source_id"] == "paper_2"  # Most recent first
    assert relationships[1]["source_id"] == "paper_1"


def test_rollback_dry_run(rollback_manager, audit_log):
    """
    Test rollback in dry-run mode (no actual deletion).
    
    Requirement 19.4: Support rollback of extractions by method version
    """
    # Log relationships
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
    )
    
    # Dry run rollback
    result = rollback_manager.rollback_by_method_version(
        "llm_extractor_v1.2", "1.2", dry_run=True
    )
    
    assert result["dry_run"] is True
    assert result["relationships_removed"] == 1
    assert result["extraction_method"] == "llm_extractor_v1.2"
    assert result["extractor_version"] == "1.2"
    
    # Verify no actual deletion occurred (driver.session not called)
    # In dry run, we don't interact with Neo4j


def test_rollback_by_method_version(rollback_manager, audit_log):
    """
    Test actual rollback by extraction method version.
    
    Requirements:
    - 10.5: Support rollback of extractions by method version
    - 19.4: Support rollback of extractions by method version
    """
    # Log relationships
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={"direction": "increased"},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_1",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_2",
        target_id="taxon_2",
        properties={"direction": "decreased"},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_2",
    )
    
    # Mock Neo4j session
    mock_session = Mock()
    rollback_manager.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
    
    # Perform rollback
    result = rollback_manager.rollback_by_method_version(
        "llm_extractor_v1.2", "1.2", dry_run=False
    )
    
    assert result["dry_run"] is False
    assert result["relationships_removed"] == 2
    assert result["extraction_method"] == "llm_extractor_v1.2"
    
    # Verify Neo4j session was called
    assert mock_session.run.called


def test_rollback_no_relationships_found(rollback_manager, audit_log):
    """
    Test rollback when no relationships are found for the method version.
    """
    # Attempt rollback with no matching relationships
    result = rollback_manager.rollback_by_method_version(
        "nonexistent_method", "1.0", dry_run=False
    )
    
    assert result["relationships_removed"] == 0
    assert result["extraction_method"] == "nonexistent_method"


def test_rollback_by_paper(rollback_manager, audit_log):
    """
    Test rollback of all extractions from a specific paper.
    """
    # Log relationships from different papers
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_1",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_2",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_1",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_2",
        target_id="taxon_3",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_2",
    )
    
    # Mock Neo4j session
    mock_session = Mock()
    rollback_manager.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
    
    # Rollback by paper
    result = rollback_manager.rollback_by_paper("paper_1", dry_run=False)
    
    assert result["relationships_removed"] == 2
    assert result["paper_id"] == "paper_1"


def test_get_extraction_method_statistics(rollback_manager, audit_log):
    """
    Test getting statistics about extractions by method version.
    """
    # Log relationships with different types
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_1",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_2",
        target_id="taxon_2",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_2",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_INTERVENTION_EFFECT",
        source_id="paper_1",
        target_id="taxon_3",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_1",
    )
    
    # Get statistics
    stats = rollback_manager.get_extraction_method_statistics(
        "llm_extractor_v1.2", "1.2"
    )
    
    assert stats["total_relationships"] == 3
    assert stats["relationships_by_type"]["REPORTS_ASSOCIATION"] == 2
    assert stats["relationships_by_type"]["REPORTS_INTERVENTION_EFFECT"] == 1
    assert stats["unique_papers"] == 2
    assert stats["relationships_by_paper"]["paper_1"] == 2
    assert stats["relationships_by_paper"]["paper_2"] == 1


def test_rollback_logs_deletion_to_audit_log(rollback_manager, audit_log):
    """
    Test that rollback operations are logged to the audit log.
    
    Requirement 19.5: Log all graph modifications
    """
    # Log a relationship
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
    )
    
    # Mock Neo4j session
    mock_session = Mock()
    rollback_manager.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
    
    # Perform rollback
    rollback_manager.rollback_by_method_version(
        "llm_extractor_v1.2", "1.2", dry_run=False
    )
    
    # Check that deletion was logged
    stats = audit_log.get_statistics()
    assert "delete_edge" in stats["operations_by_type"]
    assert stats["operations_by_type"]["delete_edge"] >= 1


def test_convenience_function():
    """
    Test the convenience function for rollback.
    """
    with patch('graph.rollback_manager.RollbackManager') as MockManager:
        mock_instance = Mock()
        MockManager.return_value.__enter__ = Mock(return_value=mock_instance)
        MockManager.return_value.__exit__ = Mock(return_value=False)
        
        mock_instance.rollback_by_method_version.return_value = {
            "relationships_removed": 5,
            "dry_run": True,
        }
        
        result = rollback_extraction_method(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            extraction_method="test_method",
            extractor_version="1.0",
            dry_run=True,
        )
        
        assert result["relationships_removed"] == 5
        assert result["dry_run"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

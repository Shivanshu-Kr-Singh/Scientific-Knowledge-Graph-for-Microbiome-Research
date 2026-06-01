"""
graph/test_audit_log.py
-----------------------
Unit tests for audit logging system.

Tests Requirements: 18.5, 19.1, 19.2, 19.3, 19.4, 19.5
"""

import pytest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from graph.audit_log import AuditLog, AuditLogEntry, get_audit_log


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


def test_audit_log_initialization(temp_db):
    """
    Test that audit log initializes correctly and creates database.
    
    Requirement 19.5: Create audit log table
    """
    audit_log = AuditLog(db_path=temp_db)
    
    # Check that database file was created
    assert os.path.exists(temp_db)
    
    # Check that we can query the database
    stats = audit_log.get_statistics()
    assert stats["total_entries"] == 0


def test_log_node_creation(audit_log):
    """
    Test logging of node creation.
    
    Requirement 19.5: Log all graph modifications with timestamp and user ID
    """
    # Log a node creation
    log_id = audit_log.log_node_creation(
        node_type="Paper",
        node_id="paper_123",
        properties={"title": "Test Paper", "year": 2024},
        user_id="test_user",
    )
    
    assert log_id > 0
    
    # Verify the log entry
    stats = audit_log.get_statistics()
    assert stats["total_entries"] == 1
    assert stats["operations_by_type"]["create_node"] == 1


def test_log_edge_creation_with_extraction_metadata(audit_log):
    """
    Test logging of edge creation with extraction metadata.
    
    Requirements:
    - 19.1: Store extraction method source code hash
    - 19.2: Store LLM prompt hash for LLM-based extractions
    - 19.5: Log with timestamp and user ID
    """
    # Log an edge creation with full metadata
    log_id = audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_123",
        target_id="taxon_456",
        properties={
            "direction": "increased",
            "p_value": 0.001,
            "confidence": 0.85,
        },
        user_id="system",
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        source_code_hash="abc123def456",
        llm_prompt_hash="prompt_hash_789",
        paper_id="paper_123",
        section="results",
    )
    
    assert log_id > 0
    
    # Query the log entry
    entries = audit_log.query_by_extraction_method("llm_extractor_v1.2", "1.2")
    assert len(entries) == 1
    
    entry = entries[0]
    assert entry.operation_type == "create_edge"
    assert entry.entity_type == "REPORTS_ASSOCIATION"
    assert entry.extraction_method == "llm_extractor_v1.2"
    assert entry.extractor_version == "1.2"
    assert entry.source_code_hash == "abc123def456"
    assert entry.llm_prompt_hash == "prompt_hash_789"
    assert entry.paper_id == "paper_123"
    assert entry.section == "results"


def test_query_by_extraction_method(audit_log):
    """
    Test querying relationships by extraction method version.
    
    Requirement 19.3: Support querying relationships by extraction method version
    """
    # Log multiple edges with different extraction methods
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={},
        extraction_method="regex_ner",
        extractor_version="1.0",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_2",
        target_id="taxon_2",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_3",
        target_id="taxon_3",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
    )
    
    # Query by extraction method
    entries = audit_log.query_by_extraction_method("llm_extractor_v1.2", "1.2")
    assert len(entries) == 2
    
    entries = audit_log.query_by_extraction_method("regex_ner", "1.0")
    assert len(entries) == 1


def test_get_relationships_by_method_version(audit_log):
    """
    Test getting relationship details by extraction method version.
    
    Requirement 19.3: Support querying relationships by extraction method version
    """
    # Log edges
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={"direction": "increased"},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
        paper_id="paper_1",
        section="results",
    )
    
    # Get relationships
    relationships = audit_log.get_relationships_by_method_version(
        "llm_extractor_v1.2", "1.2"
    )
    
    assert len(relationships) == 1
    rel = relationships[0]
    assert rel["relationship_type"] == "REPORTS_ASSOCIATION"
    assert rel["source_id"] == "paper_1"
    assert rel["target_id"] == "taxon_1"
    assert rel["properties"]["direction"] == "increased"
    assert rel["paper_id"] == "paper_1"
    assert rel["section"] == "results"


def test_query_by_user(audit_log):
    """
    Test querying modifications by user ID.
    
    Requirement 19.5: Audit log with user ID
    """
    # Log modifications by different users
    audit_log.log_node_creation(
        node_type="Paper",
        node_id="paper_1",
        properties={},
        user_id="user_alice",
    )
    
    audit_log.log_node_creation(
        node_type="Paper",
        node_id="paper_2",
        properties={},
        user_id="user_bob",
    )
    
    audit_log.log_node_creation(
        node_type="Paper",
        node_id="paper_3",
        properties={},
        user_id="user_alice",
    )
    
    # Query by user
    alice_entries = audit_log.query_by_user("user_alice")
    assert len(alice_entries) == 2
    
    bob_entries = audit_log.query_by_user("user_bob")
    assert len(bob_entries) == 1


def test_query_by_time_range(audit_log):
    """
    Test querying modifications by time range.
    
    Requirement 19.5: Audit log with timestamp
    """
    # Log modifications
    now = datetime.now(timezone.utc)
    
    audit_log.log_node_creation(
        node_type="Paper",
        node_id="paper_1",
        properties={},
    )
    
    # Query by time range
    start_time = now - timedelta(minutes=1)
    end_time = now + timedelta(minutes=1)
    
    entries = audit_log.query_by_time_range(start_time, end_time)
    assert len(entries) >= 1


def test_get_statistics(audit_log):
    """
    Test getting audit log statistics.
    """
    # Log various modifications
    audit_log.log_node_creation(
        node_type="Paper",
        node_id="paper_1",
        properties={},
        user_id="user_alice",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={},
        extraction_method="regex_ner",
        extractor_version="1.0",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_2",
        target_id="taxon_2",
        properties={},
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
    )
    
    # Get statistics
    stats = audit_log.get_statistics()
    
    assert stats["total_entries"] == 3
    assert stats["operations_by_type"]["create_node"] == 1
    assert stats["operations_by_type"]["create_edge"] == 2
    assert stats["extractions_by_method"]["regex_ner"] == 1
    assert stats["extractions_by_method"]["llm_extractor_v1.2"] == 1
    assert stats["modifications_by_user"]["user_alice"] == 1
    assert stats["modifications_by_user"]["system"] == 2


def test_audit_log_entry_model():
    """
    Test AuditLogEntry Pydantic model validation.
    """
    # Valid entry
    entry = AuditLogEntry(
        operation_type="create_edge",
        entity_type="REPORTS_ASSOCIATION",
        entity_id="paper_1->taxon_1",
        user_id="system",
        extraction_method="llm_extractor_v1.2",
        extractor_version="1.2",
    )
    
    assert entry.operation_type == "create_edge"
    assert entry.user_id == "system"
    assert entry.timestamp is not None


def test_multiple_versions_same_method(audit_log):
    """
    Test querying different versions of the same extraction method.
    
    Requirement 19.3: Support querying by extraction method version
    """
    # Log edges with different versions
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_1",
        target_id="taxon_1",
        properties={},
        extraction_method="llm_extractor",
        extractor_version="1.0",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_2",
        target_id="taxon_2",
        properties={},
        extraction_method="llm_extractor",
        extractor_version="1.1",
    )
    
    audit_log.log_edge_creation(
        relationship_type="REPORTS_ASSOCIATION",
        source_id="paper_3",
        target_id="taxon_3",
        properties={},
        extraction_method="llm_extractor",
        extractor_version="1.2",
    )
    
    # Query specific version
    v1_0_entries = audit_log.query_by_extraction_method("llm_extractor", "1.0")
    assert len(v1_0_entries) == 1
    
    v1_2_entries = audit_log.query_by_extraction_method("llm_extractor", "1.2")
    assert len(v1_2_entries) == 1
    
    # Query all versions (no version specified)
    all_entries = audit_log.query_by_extraction_method("llm_extractor")
    assert len(all_entries) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

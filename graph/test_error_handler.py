"""
graph/test_error_handler.py
----------------------------
Unit tests for error handling and recovery mechanisms.

Tests Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from graph.error_handler import ErrorHandler
from graph.semantic_relationships import SemanticRelationship, RelationType
from graph.reified_claims import ScientificClaim, EvidenceStrength
from graph.provenance import ProvenanceMetadata


@pytest.fixture
def temp_queues():
    """Create temporary queue files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        yield {
            "incomplete_extraction": str(tmpdir_path / "incomplete_extraction.json"),
            "curator_review": str(tmpdir_path / "curator_review.json"),
            "query_log": str(tmpdir_path / "query_log.json")
        }


@pytest.fixture
def error_handler(temp_queues):
    """Create an ErrorHandler with temporary queues."""
    return ErrorHandler(
        incomplete_extraction_queue_path=temp_queues["incomplete_extraction"],
        curator_review_queue_path=temp_queues["curator_review"],
        query_log_path=temp_queues["query_log"]
    )


@pytest.fixture
def sample_provenance():
    """Create sample provenance metadata."""
    return ProvenanceMetadata(
        paper_id="PMC123456",
        section_type="results",
        source_sentence="Bacteroides fragilis was significantly increased in T2D patients.",
        sentence_offset=100,
        extraction_method="regex_ner",
        extraction_timestamp=datetime.now(timezone.utc),
        extractor_version="1.0",
        confidence_score=0.85,
        validation_status="unvalidated"
    )


@pytest.fixture
def sample_relationship(sample_provenance):
    """Create sample semantic relationship."""
    return SemanticRelationship(
        source_entity="Bacteroides fragilis",
        target_entity="Type 2 Diabetes",
        relation_type=RelationType.REPORTS_ASSOCIATION,
        properties={
            "direction": "increased",
            "comparison": "T2D vs healthy",
            "statistical_measure": "LDA score",
            "effect_size": 3.2,
            "p_value": 0.001
        },
        provenance=sample_provenance,
        evidence_strength="strong",
        extraction_confidence=0.85
    )


@pytest.fixture
def sample_claim():
    """Create sample scientific claim."""
    return ScientificClaim(
        claim_id="claim-001",
        claim_type="association",
        subject_entity="Bacteroides fragilis",
        predicate="associated_with_increased",
        object_entity="Type 2 Diabetes",
        supporting_papers=["PMC123456", "PMC789012"],
        contradicting_papers=[],
        total_sample_size=500,
        evidence_strength=EvidenceStrength.STRONG,
        consensus_confidence=0.85,
        effect_direction_consistency=1.0,
        first_reported="2024-01-01T00:00:00Z",
        last_updated="2024-01-15T00:00:00Z"
    )


# ========== Requirement 15.1: Extraction Failure Handling ==========

class TestExtractionFailureHandling:
    """
    Test extraction failure handling.
    
    Requirement 15.1: WHEN extraction fails to capture provenance data,
    THE System SHALL log a warning and add the paper to an "incomplete_extraction"
    queue without creating a graph edge.
    """
    
    def test_handle_extraction_failure(self, error_handler, temp_queues):
        """Test handling extraction failure adds paper to queue."""
        # Handle extraction failure
        error_handler.handle_extraction_failure(
            paper_id="PMC123456",
            paper_title="Test Paper",
            failure_reason="Missing provenance data",
            details={"missing_fields": ["source_sentence", "section_type"]}
        )
        
        # Verify queue file exists
        queue_path = Path(temp_queues["incomplete_extraction"])
        assert queue_path.exists()
        
        # Verify entry in queue
        with open(queue_path, 'r') as f:
            queue = json.load(f)
        
        assert len(queue) == 1
        assert queue[0]["paper_id"] == "PMC123456"
        assert queue[0]["paper_title"] == "Test Paper"
        assert queue[0]["failure_reason"] == "Missing provenance data"
        assert queue[0]["status"] == "pending_review"
        assert "queued_at" in queue[0]
    
    def test_multiple_extraction_failures(self, error_handler, temp_queues):
        """Test multiple extraction failures are appended to queue."""
        # Add multiple failures
        error_handler.handle_extraction_failure("PMC111111", failure_reason="Missing section")
        error_handler.handle_extraction_failure("PMC222222", failure_reason="Missing sentence")
        error_handler.handle_extraction_failure("PMC333333", failure_reason="Invalid confidence")
        
        # Verify all in queue
        queue_path = Path(temp_queues["incomplete_extraction"])
        with open(queue_path, 'r') as f:
            queue = json.load(f)
        
        assert len(queue) == 3
        paper_ids = [entry["paper_id"] for entry in queue]
        assert "PMC111111" in paper_ids
        assert "PMC222222" in paper_ids
        assert "PMC333333" in paper_ids
    
    def test_validate_provenance_completeness_valid(self, error_handler, sample_provenance):
        """Test provenance validation with valid provenance."""
        is_valid, error_msg = error_handler.validate_provenance_completeness(sample_provenance)
        
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_provenance_completeness_missing_paper_id(self, error_handler, sample_provenance):
        """Test provenance validation with missing paper_id."""
        sample_provenance.paper_id = ""
        
        is_valid, error_msg = error_handler.validate_provenance_completeness(sample_provenance)
        
        assert is_valid is False
        assert error_msg == "Missing paper_id"
    
    def test_validate_provenance_completeness_missing_section(self, error_handler, sample_provenance):
        """Test provenance validation with missing section_type."""
        sample_provenance.section_type = ""
        
        is_valid, error_msg = error_handler.validate_provenance_completeness(sample_provenance)
        
        assert is_valid is False
        assert error_msg == "Missing section_type"
    
    def test_validate_provenance_completeness_missing_sentence(self, error_handler, sample_provenance):
        """Test provenance validation with missing source_sentence."""
        sample_provenance.source_sentence = ""
        
        is_valid, error_msg = error_handler.validate_provenance_completeness(sample_provenance)
        
        assert is_valid is False
        assert error_msg == "Missing source_sentence"
    
    def test_validate_provenance_completeness_invalid_confidence(self, error_handler, sample_provenance):
        """Test provenance validation with invalid confidence score."""
        sample_provenance.confidence_score = 1.5
        
        is_valid, error_msg = error_handler.validate_provenance_completeness(sample_provenance)
        
        assert is_valid is False
        assert "Invalid confidence_score" in error_msg
    
    def test_get_incomplete_extraction_stats(self, error_handler):
        """Test getting statistics about incomplete extraction queue."""
        # Add some failures
        error_handler.handle_extraction_failure("PMC111", failure_reason="Missing provenance")
        error_handler.handle_extraction_failure("PMC222", failure_reason="Missing provenance")
        error_handler.handle_extraction_failure("PMC333", failure_reason="Invalid data")
        
        # Get stats
        stats = error_handler.get_incomplete_extraction_stats()
        
        assert stats["queue_size"] == 3
        assert stats["exists"] is True
        assert stats["failure_reason_counts"]["Missing provenance"] == 2
        assert stats["failure_reason_counts"]["Invalid data"] == 1


# ========== Requirement 15.2: Conflicting Statistics Handling ==========

class TestConflictingStatisticsHandling:
    """
    Test conflicting statistics handling.
    
    Requirement 15.2: WHEN multiple conflicting statistical measures are found
    in the same paper, THE System SHALL create separate edges for each distinct
    claim and flag the paper with "conflicting_statistics".
    """
    
    def test_handle_conflicting_statistics(self, error_handler, sample_relationship):
        """Test handling conflicting statistics flags relationships."""
        # Create multiple relationships with different statistics
        rel1 = sample_relationship
        rel2 = SemanticRelationship(
            source_entity=sample_relationship.source_entity,
            target_entity=sample_relationship.target_entity,
            relation_type=sample_relationship.relation_type,
            properties={
                "direction": "increased",
                "comparison": "T2D vs healthy",
                "statistical_measure": "fold change",
                "effect_size": 2.5,
                "p_value": 0.01
            },
            provenance=sample_relationship.provenance,
            evidence_strength="moderate",
            extraction_confidence=0.75
        )
        
        relationships = [rel1, rel2]
        
        # Handle conflicting statistics
        result = error_handler.handle_conflicting_statistics(
            paper_id="PMC123456",
            relationships=relationships,
            conflict_details={"measures": ["LDA score", "fold change"]}
        )
        
        # Verify all relationships returned
        assert len(result) == 2
        
        # Verify relationships are flagged
        for rel in result:
            assert rel.properties["conflicting_statistics"] is True
            assert rel.properties["conflict_group_size"] == 2
    
    def test_detect_conflicting_statistics(self, error_handler, sample_relationship):
        """Test detecting conflicting statistics in relationships."""
        # Create relationships with same source/target but different statistics
        rel1 = sample_relationship
        rel2 = SemanticRelationship(
            source_entity=sample_relationship.source_entity,
            target_entity=sample_relationship.target_entity,
            relation_type=sample_relationship.relation_type,
            properties={
                "direction": "increased",
                "comparison": "T2D vs healthy",
                "statistical_measure": "fold change",
                "effect_size": 2.5,  # Different effect size
                "p_value": 0.01  # Different p-value
            },
            provenance=sample_relationship.provenance,
            evidence_strength="moderate",
            extraction_confidence=0.75
        )
        
        relationships = [rel1, rel2]
        
        # Detect conflicts
        conflicts = error_handler.detect_conflicting_statistics(relationships)
        
        # Verify conflict detected
        assert "PMC123456" in conflicts
        assert len(conflicts["PMC123456"]) == 2
    
    def test_no_conflict_when_same_statistics(self, error_handler, sample_relationship):
        """Test no conflict detected when statistics are the same."""
        # Create relationships with identical statistics
        rel1 = sample_relationship
        rel2 = SemanticRelationship(
            source_entity=sample_relationship.source_entity,
            target_entity=sample_relationship.target_entity,
            relation_type=sample_relationship.relation_type,
            properties=sample_relationship.properties.copy(),
            provenance=sample_relationship.provenance,
            evidence_strength=sample_relationship.evidence_strength,
            extraction_confidence=sample_relationship.extraction_confidence
        )
        
        relationships = [rel1, rel2]
        
        # Detect conflicts
        conflicts = error_handler.detect_conflicting_statistics(relationships)
        
        # Verify no conflict
        assert len(conflicts) == 0


# ========== Requirement 15.3: Entity Normalization Failure Handling ==========

class TestEntityNormalizationFailureHandling:
    """
    Test entity normalization failure handling.
    
    Requirement 15.3: WHEN entity normalization fails, THE System SHALL
    create an "ungrounded" node with temporary ID and add to curator review queue.
    """
    
    def test_handle_entity_normalization_failure(self, error_handler, temp_queues):
        """Test handling entity normalization failure adds to curator queue."""
        # Handle normalization failure
        error_handler.handle_entity_normalization_failure(
            entity_text="Unknown Bacterium XYZ",
            entity_type="taxon",
            failure_reason="No match in NCBI Taxonomy",
            temporary_id="ungrounded:unknown_bacterium_xyz"
        )
        
        # Verify queue file exists
        queue_path = Path(temp_queues["curator_review"])
        assert queue_path.exists()
        
        # Verify entry in queue
        with open(queue_path, 'r') as f:
            queue = json.load(f)
        
        assert len(queue) == 1
        assert queue[0]["entity_text"] == "Unknown Bacterium XYZ"
        assert queue[0]["entity_type"] == "taxon"
        assert queue[0]["temporary_id"] == "ungrounded:unknown_bacterium_xyz"
        assert queue[0]["failure_reason"] == "No match in NCBI Taxonomy"
        assert queue[0]["status"] == "pending_review"
    
    def test_duplicate_entity_not_added_twice(self, error_handler):
        """Test that duplicate entities are not added to queue twice."""
        # Add same entity twice
        error_handler.handle_entity_normalization_failure(
            entity_text="Unknown Bacterium",
            entity_type="taxon",
            failure_reason="No match",
            temporary_id="ungrounded:unknown_bacterium"
        )
        error_handler.handle_entity_normalization_failure(
            entity_text="Unknown Bacterium",
            entity_type="taxon",
            failure_reason="No match",
            temporary_id="ungrounded:unknown_bacterium"
        )
        
        # Verify only one entry
        stats = error_handler.get_curator_review_stats()
        assert stats["queue_size"] == 1
    
    def test_get_curator_review_stats(self, error_handler):
        """Test getting statistics about curator review queue."""
        # Add some entities
        error_handler.handle_entity_normalization_failure(
            "Unknown Taxon 1", "taxon", "No match", "ungrounded:taxon1"
        )
        error_handler.handle_entity_normalization_failure(
            "Unknown Taxon 2", "taxon", "No match", "ungrounded:taxon2"
        )
        error_handler.handle_entity_normalization_failure(
            "Unknown Disease", "disease", "No match", "ungrounded:disease1"
        )
        
        # Get stats
        stats = error_handler.get_curator_review_stats()
        
        assert stats["queue_size"] == 3
        assert stats["exists"] is True
        assert stats["entity_type_counts"]["taxon"] == 2
        assert stats["entity_type_counts"]["disease"] == 1


# ========== Requirement 15.4: Query Timeout Handling ==========

class TestQueryTimeoutHandling:
    """
    Test query timeout handling.
    
    Requirement 15.4: WHEN a query times out, THE System SHALL cancel execution,
    return partial results with timeout flag, and log the query pattern for optimization.
    """
    
    def test_handle_query_timeout(self, error_handler, temp_queues):
        """Test handling query timeout returns partial results with timeout flag."""
        # Simulate query timeout
        partial_results = [
            {"taxon": "Bacteroides fragilis", "paper_count": 5},
            {"taxon": "Faecalibacterium prausnitzii", "paper_count": 3}
        ]
        
        result = error_handler.handle_query_timeout(
            query_description="Cross-study associations for Type 2 Diabetes",
            query_params={"disease": "Type 2 Diabetes", "min_papers": 3},
            partial_results=partial_results,
            execution_time_ms=35000,
            timeout_threshold_ms=30000
        )
        
        # Verify result structure
        assert result["timeout"] is True
        assert result["result_count"] == 2
        assert len(result["results"]) == 2
        assert result["execution_time_ms"] == 35000
        assert result["timeout_threshold_ms"] == 30000
        assert "message" in result
        
        # Verify query logged
        log_path = Path(temp_queues["query_log"])
        assert log_path.exists()
        
        with open(log_path, 'r') as f:
            log = json.load(f)
        
        assert len(log) == 1
        assert log[0]["query_description"] == "Cross-study associations for Type 2 Diabetes"
        assert log[0]["execution_time_ms"] == 35000
        assert log[0]["partial_result_count"] == 2
        assert log[0]["status"] == "timeout"
    
    def test_multiple_query_timeouts_logged(self, error_handler):
        """Test multiple query timeouts are logged."""
        # Simulate multiple timeouts
        error_handler.handle_query_timeout(
            "Query 1", {}, [], 31000, 30000
        )
        error_handler.handle_query_timeout(
            "Query 2", {}, [], 32000, 30000
        )
        error_handler.handle_query_timeout(
            "Query 3", {}, [], 33000, 30000
        )
        
        # Verify all logged
        stats = error_handler.get_query_timeout_stats()
        assert stats["timeout_count"] == 3
        assert stats["avg_execution_time_ms"] == 32000
    
    def test_get_query_timeout_stats(self, error_handler):
        """Test getting statistics about query timeouts."""
        # Add some timeouts
        error_handler.handle_query_timeout(
            "Cross-study associations", {}, [], 31000, 30000
        )
        error_handler.handle_query_timeout(
            "Cross-study associations", {}, [], 32000, 30000
        )
        error_handler.handle_query_timeout(
            "Intervention evidence", {}, [], 35000, 30000
        )
        
        # Get stats
        stats = error_handler.get_query_timeout_stats()
        
        assert stats["timeout_count"] == 3
        assert stats["exists"] is True
        assert stats["query_description_counts"]["Cross-study associations"] == 2
        assert stats["query_description_counts"]["Intervention evidence"] == 1
        assert stats["avg_execution_time_ms"] == (31000 + 32000 + 35000) / 3


# ========== Requirement 15.5: Conflicting Claims Handling ==========

class TestConflictingClaimsHandling:
    """
    Test conflicting claims handling.
    
    Requirement 15.5: WHEN attempting to create a reified claim with opposite
    predicate to an existing claim, THE System SHALL create separate claims
    and link them with CONFLICTS_WITH relationship.
    """
    
    def test_handle_conflicting_claims(self, error_handler, sample_claim):
        """Test handling conflicting claims creates CONFLICTS_WITH relationship."""
        # Create conflicting claim
        conflicting_claim = ScientificClaim(
            claim_id="claim-002",
            claim_type="association",
            subject_entity="Bacteroides fragilis",
            predicate="associated_with_decreased",  # Opposite predicate
            object_entity="Type 2 Diabetes",
            supporting_papers=["PMC345678"],
            contradicting_papers=[],
            total_sample_size=200,
            evidence_strength=EvidenceStrength.MODERATE,
            consensus_confidence=0.75,
            effect_direction_consistency=1.0,
            first_reported="2024-02-01T00:00:00Z",
            last_updated="2024-02-01T00:00:00Z"
        )
        
        # Handle conflicting claims
        existing, new, conflict_rel = error_handler.handle_conflicting_claims(
            sample_claim,
            conflicting_claim
        )
        
        # Verify claims returned
        assert existing.claim_id == "claim-001"
        assert new.claim_id == "claim-002"
        
        # Verify conflict relationship
        assert conflict_rel["relationship_type"] == "CONFLICTS_WITH"
        assert conflict_rel["source_claim_id"] == "claim-001"
        assert conflict_rel["target_claim_id"] == "claim-002"
        assert conflict_rel["conflict_type"] == "opposite_predicate"
        assert conflict_rel["existing_predicate"] == "associated_with_increased"
        assert conflict_rel["new_predicate"] == "associated_with_decreased"
    
    def test_detect_opposite_predicates_increased_decreased(self, error_handler):
        """Test detecting increased/decreased opposition."""
        assert error_handler.detect_opposite_predicates(
            "associated_with_increased",
            "associated_with_decreased"
        ) is True
        
        assert error_handler.detect_opposite_predicates(
            "decreased_abundance",
            "increased_abundance"
        ) is True
    
    def test_detect_opposite_predicates_positive_negative(self, error_handler):
        """Test detecting positive/negative opposition."""
        assert error_handler.detect_opposite_predicates(
            "positive_effect",
            "negative_effect"
        ) is True
    
    def test_detect_opposite_predicates_up_down(self, error_handler):
        """Test detecting up/down opposition."""
        assert error_handler.detect_opposite_predicates(
            "upregulated",
            "downregulated"
        ) is True
    
    def test_detect_opposite_predicates_higher_lower(self, error_handler):
        """Test detecting higher/lower opposition."""
        assert error_handler.detect_opposite_predicates(
            "higher_abundance",
            "lower_abundance"
        ) is True
    
    def test_detect_opposite_predicates_no_opposition(self, error_handler):
        """Test no opposition detected for similar predicates."""
        assert error_handler.detect_opposite_predicates(
            "associated_with_increased",
            "associated_with_increased"
        ) is False
        
        assert error_handler.detect_opposite_predicates(
            "uses_methodology",
            "reports_finding"
        ) is False


# ========== Integration Tests ==========

class TestErrorHandlerIntegration:
    """Integration tests for error handler."""
    
    def test_complete_error_handling_workflow(self, error_handler):
        """Test complete error handling workflow with all error types."""
        # 1. Handle extraction failure
        error_handler.handle_extraction_failure(
            "PMC111111",
            failure_reason="Missing provenance"
        )
        
        # 2. Handle entity normalization failure
        error_handler.handle_entity_normalization_failure(
            "Unknown Taxon",
            "taxon",
            "No match",
            "ungrounded:unknown"
        )
        
        # 3. Handle query timeout
        error_handler.handle_query_timeout(
            "Test Query",
            {},
            [],
            31000,
            30000
        )
        
        # Verify all queues populated
        extraction_stats = error_handler.get_incomplete_extraction_stats()
        curator_stats = error_handler.get_curator_review_stats()
        timeout_stats = error_handler.get_query_timeout_stats()
        
        assert extraction_stats["queue_size"] == 1
        assert curator_stats["queue_size"] == 1
        assert timeout_stats["timeout_count"] == 1

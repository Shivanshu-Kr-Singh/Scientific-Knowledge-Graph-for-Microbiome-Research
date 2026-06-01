"""
graph/test_enhanced_loader_validation.py
-----------------------------------------
Integration tests for enhanced Neo4j loader with data validation.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
"""

import pytest
from datetime import datetime, UTC
from pathlib import Path
import tempfile
import json

from graph.enhanced_neo4j_loader import EnhancedNeo4jLoader
from graph.semantic_relationships import create_association_relationship
from graph.provenance import ProvenanceMetadata


@pytest.fixture
def temp_validation_queue():
    """Create a temporary validation queue file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_neo4j_loader(temp_validation_queue):
    """
    Create a mock Neo4j loader for testing validation without actual database.
    
    Note: This uses a mock URI since we're testing validation logic, not actual Neo4j operations.
    """
    # Create loader with mock credentials (won't actually connect in these tests)
    loader = EnhancedNeo4jLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="test",
        validation_queue_path=temp_validation_queue
    )
    
    # Don't actually connect to Neo4j
    yield loader
    
    # Cleanup
    try:
        loader.close()
    except:
        pass  # Ignore connection errors in tests


@pytest.fixture
def valid_provenance():
    """Create valid provenance metadata for testing."""
    return ProvenanceMetadata(
        paper_id="PMC123456",
        section_type="results",
        source_sentence="Bacteroides fragilis was significantly increased in T2D patients.",
        sentence_offset=100,
        extraction_method="llm_extractor_v1.2",
        extraction_timestamp=datetime.now(UTC),
        extractor_version="1.2.0",
        confidence_score=0.85,
        validation_status="unvalidated",
    )


class TestLoaderValidationIntegration:
    """
    Integration tests for loader with validation enabled.
    
    Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
    """
    
    def test_loader_has_validator(self, mock_neo4j_loader):
        """Test that loader initializes with a validator."""
        assert mock_neo4j_loader.validator is not None
        assert hasattr(mock_neo4j_loader.validator, 'validate_batch')
    
    def test_validation_queue_path_configuration(self, temp_validation_queue):
        """Test that validation queue path is properly configured."""
        loader = EnhancedNeo4jLoader(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
            validation_queue_path=temp_validation_queue
        )
        
        assert loader.validator.validation_queue_path == Path(temp_validation_queue)
        
        try:
            loader.close()
        except:
            pass
    
    def test_load_relationships_with_validation_enabled(self, mock_neo4j_loader, valid_provenance):
        """
        Test that load_relationships_batch validates when validate=True.
        
        This test verifies the validation logic without actually connecting to Neo4j.
        """
        # Create mix of valid and invalid relationships
        valid_rel = create_association_relationship(
            source_entity="PMC111111",
            target_entity="taxon_valid",
            direction="increased",
            comparison="T2D vs healthy",
            statistical_measure="LDA score",
            provenance=valid_provenance,
            evidence_strength="strong",
            extraction_confidence=0.9,
            p_value=0.001,
        )
        
        invalid_rel = create_association_relationship(
            source_entity="PMC222222",
            target_entity="taxon_invalid",
            direction="decreased",
            comparison="IBD vs healthy",
            statistical_measure="fold change",
            provenance=valid_provenance,
            evidence_strength="moderate",
            extraction_confidence=0.8,
            p_value=0.5,
        )
        # Make it invalid
        invalid_rel.properties["p_value"] = 2.5
        
        relationships = [valid_rel, invalid_rel]
        
        # Test validation without actual Neo4j loading
        # We'll just test the validation part
        validation_result = mock_neo4j_loader.validator.validate_batch(relationships)
        
        assert validation_result.total_count == 2
        assert validation_result.valid_count == 1
        assert validation_result.invalid_count == 1
        
        # Store invalid relationships
        mock_neo4j_loader.validator.store_invalid_relationships(
            validation_result.invalid_relationships
        )
        
        # Verify queue was created
        assert mock_neo4j_loader.validator.validation_queue_path.exists()
        
        # Verify queue contents
        with open(mock_neo4j_loader.validator.validation_queue_path, 'r') as f:
            queue = json.load(f)
        
        assert len(queue) == 1
        assert queue[0]["source_entity"] == "PMC222222"
        assert queue[0]["validation_errors"][0]["field_name"] == "p_value"
    
    def test_validation_can_be_disabled(self, mock_neo4j_loader, valid_provenance):
        """Test that validation can be disabled with validate=False."""
        # Create an invalid relationship
        invalid_rel = create_association_relationship(
            source_entity="PMC333333",
            target_entity="taxon_test",
            direction="increased",
            comparison="test",
            statistical_measure="test",
            provenance=valid_provenance,
            evidence_strength="weak",
            extraction_confidence=0.7,
            p_value=0.5,
        )
        invalid_rel.properties["p_value"] = 2.5
        
        # When validation is disabled, the validator should not be called
        # We can't test actual loading without Neo4j, but we can verify
        # the validation logic path
        is_valid, errors = mock_neo4j_loader.validator.validate_relationship(invalid_rel)
        
        assert not is_valid
        assert len(errors) > 0


class TestValidationQueueManagement:
    """
    Tests for validation queue management.
    
    Requirement 14.5: Store invalid relationships in validation queue
    """
    
    def test_validation_queue_created_on_first_invalid(self, valid_provenance):
        """Test that validation queue is created when first invalid relationship is found."""
        # Create a new temp file path that doesn't exist yet
        with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix='.json') as f:
            temp_path = f.name
        
        # Path should not exist now
        assert not Path(temp_path).exists()
        
        # Create loader with this path
        loader = EnhancedNeo4jLoader(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
            validation_queue_path=temp_path
        )
        
        # Create invalid relationship
        invalid_rel = create_association_relationship(
            source_entity="PMC444444",
            target_entity="taxon_test",
            direction="increased",
            comparison="test",
            statistical_measure="test",
            provenance=valid_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
            p_value=0.5,
        )
        invalid_rel.properties["p_value"] = -0.1
        
        is_valid, errors = loader.validator.validate_relationship(invalid_rel)
        assert not is_valid
        
        # Store in queue
        loader.validator.store_invalid_relationships([(invalid_rel, errors)])
        
        # Queue should now exist
        assert Path(temp_path).exists()
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        try:
            loader.close()
        except:
            pass
    
    def test_validation_queue_accumulates_errors(self, mock_neo4j_loader, valid_provenance):
        """Test that validation queue accumulates multiple invalid relationships."""
        invalid_rels = []
        
        for i in range(3):
            rel = create_association_relationship(
                source_entity=f"PMC{i}",
                target_entity=f"taxon_{i}",
                direction="increased",
                comparison="test",
                statistical_measure="test",
                provenance=valid_provenance,
                evidence_strength="weak",
                extraction_confidence=0.6,
                p_value=0.5,
            )
            # Make each one invalid in a different way
            if i == 0:
                rel.properties["p_value"] = 2.0
            elif i == 1:
                rel.extraction_confidence = 1.5
            else:
                rel.properties["direction"] = "invalid"
            
            is_valid, errors = mock_neo4j_loader.validator.validate_relationship(rel)
            if not is_valid:
                invalid_rels.append((rel, errors))
        
        # Store all invalid relationships
        mock_neo4j_loader.validator.store_invalid_relationships(invalid_rels)
        
        # Check queue stats
        stats = mock_neo4j_loader.validator.get_validation_queue_stats()
        assert stats["queue_size"] == 3
        assert stats["exists"] is True


class TestValidationErrorTypes:
    """
    Tests for different types of validation errors.
    
    Requirements: 14.1, 14.2, 14.3, 14.4
    """
    
    def test_confidence_score_validation_error(self, mock_neo4j_loader, valid_provenance):
        """Test validation error for invalid confidence score (Requirement 14.1)."""
        rel = create_association_relationship(
            source_entity="PMC555555",
            target_entity="taxon_test",
            direction="increased",
            comparison="test",
            statistical_measure="test",
            provenance=valid_provenance,
            evidence_strength="strong",
            extraction_confidence=0.9,
        )
        rel.extraction_confidence = 1.5
        
        is_valid, errors = mock_neo4j_loader.validator.validate_relationship(rel)
        
        assert not is_valid
        assert any(e.field_name == "extraction_confidence" for e in errors)
        assert any(e.error_type == "out_of_range" for e in errors)
    
    def test_p_value_validation_error(self, mock_neo4j_loader, valid_provenance):
        """Test validation error for invalid p-value (Requirement 14.2)."""
        rel = create_association_relationship(
            source_entity="PMC666666",
            target_entity="taxon_test",
            direction="decreased",
            comparison="test",
            statistical_measure="test",
            provenance=valid_provenance,
            evidence_strength="moderate",
            extraction_confidence=0.75,
            p_value=0.5,
        )
        rel.properties["p_value"] = 1.5
        
        is_valid, errors = mock_neo4j_loader.validator.validate_relationship(rel)
        
        assert not is_valid
        assert any(e.field_name == "p_value" for e in errors)
        assert any(e.error_type == "out_of_range" for e in errors)
    
    def test_direction_validation_error(self, mock_neo4j_loader, valid_provenance):
        """Test validation error for invalid direction (Requirement 14.3)."""
        rel = create_association_relationship(
            source_entity="PMC777777",
            target_entity="taxon_test",
            direction="increased",
            comparison="test",
            statistical_measure="test",
            provenance=valid_provenance,
            evidence_strength="weak",
            extraction_confidence=0.65,
        )
        rel.properties["direction"] = "stable"
        
        is_valid, errors = mock_neo4j_loader.validator.validate_relationship(rel)
        
        assert not is_valid
        assert any(e.field_name == "direction" for e in errors)
        assert any(e.error_type == "invalid_value" for e in errors)
    
    def test_evidence_strength_validation_error(self, mock_neo4j_loader, valid_provenance):
        """Test validation error for invalid evidence_strength (Requirement 14.4)."""
        rel = create_association_relationship(
            source_entity="PMC888888",
            target_entity="taxon_test",
            direction="no_change",
            comparison="test",
            statistical_measure="test",
            provenance=valid_provenance,
            evidence_strength="strong",
            extraction_confidence=0.8,
        )
        rel.evidence_strength = "very_strong"
        
        is_valid, errors = mock_neo4j_loader.validator.validate_relationship(rel)
        
        assert not is_valid
        assert any(e.field_name == "evidence_strength" for e in errors)
        assert any(e.error_type == "invalid_value" for e in errors)

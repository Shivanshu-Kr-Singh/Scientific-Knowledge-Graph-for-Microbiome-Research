"""
graph/test_data_validator.py
-----------------------------
Unit tests for data validation before loading into Neo4j.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
"""

import pytest
from datetime import datetime, UTC
from pathlib import Path
import json
import tempfile

from graph.data_validator import DataValidator, ValidationError, ValidationResult
from graph.semantic_relationships import (
    SemanticRelationship,
    RelationType,
    create_association_relationship,
    create_intervention_relationship,
)
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
def validator(temp_validation_queue):
    """Create a DataValidator with temporary queue."""
    return DataValidator(validation_queue_path=temp_validation_queue)


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


@pytest.fixture
def valid_association(valid_provenance):
    """Create a valid association relationship for testing."""
    return create_association_relationship(
        source_entity="PMC123456",
        target_entity="taxon_bacteroides_fragilis",
        direction="increased",
        comparison="T2D vs healthy",
        statistical_measure="LDA score",
        provenance=valid_provenance,
        evidence_strength="strong",
        extraction_confidence=0.85,
        effect_size=3.2,
        p_value=0.001,
    )


class TestConfidenceScoreValidation:
    """
    Test confidence score validation.
    
    Requirement 14.1: System SHALL validate that all confidence scores
    are in the range [0.0, 1.0]
    """
    
    def test_valid_confidence_score(self, validator, valid_association):
        """Test that valid confidence scores pass validation."""
        errors = validator.validate_confidence_score(valid_association)
        assert len(errors) == 0
    
    def test_confidence_score_at_lower_bound(self, validator, valid_association):
        """Test confidence score at lower bound (0.5) is valid."""
        valid_association.extraction_confidence = 0.5
        errors = validator.validate_confidence_score(valid_association)
        assert len(errors) == 0
    
    def test_confidence_score_at_upper_bound(self, validator, valid_association):
        """Test confidence score at upper bound (1.0) is valid."""
        valid_association.extraction_confidence = 1.0
        errors = validator.validate_confidence_score(valid_association)
        assert len(errors) == 0
    
    def test_confidence_score_below_range(self, validator, valid_association):
        """Test that confidence score below 0.0 fails validation."""
        valid_association.extraction_confidence = -0.1
        errors = validator.validate_confidence_score(valid_association)
        assert len(errors) == 1
        assert errors[0].error_type == "out_of_range"
        assert errors[0].field_name == "extraction_confidence"
    
    def test_confidence_score_above_range(self, validator, valid_association):
        """Test that confidence score above 1.0 fails validation."""
        valid_association.extraction_confidence = 1.5
        errors = validator.validate_confidence_score(valid_association)
        assert len(errors) == 1
        assert errors[0].error_type == "out_of_range"
        assert errors[0].field_name == "extraction_confidence"
    
    def test_confidence_score_invalid_type(self, validator, valid_association):
        """Test that non-numeric confidence score fails validation."""
        valid_association.extraction_confidence = "high"
        errors = validator.validate_confidence_score(valid_association)
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_type"


class TestPValueValidation:
    """
    Test p-value validation.
    
    Requirement 14.2: System SHALL validate that all p_values (when present)
    are in the range [0.0, 1.0]
    """
    
    def test_valid_p_value(self, validator, valid_association):
        """Test that valid p-value passes validation."""
        errors = validator.validate_p_value(valid_association)
        assert len(errors) == 0
    
    def test_p_value_at_lower_bound(self, validator, valid_association):
        """Test p-value at lower bound (0.0) is valid."""
        valid_association.properties["p_value"] = 0.0
        errors = validator.validate_p_value(valid_association)
        assert len(errors) == 0
    
    def test_p_value_at_upper_bound(self, validator, valid_association):
        """Test p-value at upper bound (1.0) is valid."""
        valid_association.properties["p_value"] = 1.0
        errors = validator.validate_p_value(valid_association)
        assert len(errors) == 0
    
    def test_p_value_below_range(self, validator, valid_association):
        """Test that p-value below 0.0 fails validation."""
        valid_association.properties["p_value"] = -0.01
        errors = validator.validate_p_value(valid_association)
        assert len(errors) == 1
        assert errors[0].error_type == "out_of_range"
        assert errors[0].field_name == "p_value"
    
    def test_p_value_above_range(self, validator, valid_association):
        """Test that p-value above 1.0 fails validation."""
        valid_association.properties["p_value"] = 1.5
        errors = validator.validate_p_value(valid_association)
        assert len(errors) == 1
        assert errors[0].error_type == "out_of_range"
        assert errors[0].field_name == "p_value"
    
    def test_p_value_missing(self, validator, valid_association):
        """Test that missing p-value is acceptable (optional field)."""
        del valid_association.properties["p_value"]
        errors = validator.validate_p_value(valid_association)
        assert len(errors) == 0
    
    def test_p_value_none(self, validator, valid_association):
        """Test that None p-value is acceptable."""
        valid_association.properties["p_value"] = None
        errors = validator.validate_p_value(valid_association)
        assert len(errors) == 0
    
    def test_adjusted_p_value_validation(self, validator, valid_association):
        """Test that adjusted_p_value is also validated."""
        valid_association.properties["adjusted_p_value"] = 1.5
        errors = validator.validate_p_value(valid_association)
        assert len(errors) == 1
        assert errors[0].field_name == "adjusted_p_value"
        assert errors[0].error_type == "out_of_range"


class TestDirectionValidation:
    """
    Test direction value validation.
    
    Requirement 14.3: System SHALL validate that direction values are in
    the set {"increased", "decreased", "no_change"}
    """
    
    def test_valid_direction_increased(self, validator, valid_association):
        """Test that 'increased' direction is valid."""
        valid_association.properties["direction"] = "increased"
        errors = validator.validate_direction(valid_association)
        assert len(errors) == 0
    
    def test_valid_direction_decreased(self, validator, valid_association):
        """Test that 'decreased' direction is valid."""
        valid_association.properties["direction"] = "decreased"
        errors = validator.validate_direction(valid_association)
        assert len(errors) == 0
    
    def test_valid_direction_no_change(self, validator, valid_association):
        """Test that 'no_change' direction is valid."""
        valid_association.properties["direction"] = "no_change"
        errors = validator.validate_direction(valid_association)
        assert len(errors) == 0
    
    def test_invalid_direction(self, validator, valid_association):
        """Test that invalid direction value fails validation."""
        valid_association.properties["direction"] = "stable"
        errors = validator.validate_direction(valid_association)
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_value"
        assert errors[0].field_name == "direction"
        assert errors[0].invalid_value == "stable"
    
    def test_effect_direction_validation(self, validator, valid_provenance):
        """Test that effect_direction is also validated for interventions."""
        # Create a valid intervention first, then modify it to bypass Pydantic validation
        intervention = create_intervention_relationship(
            source_entity="PMC123456",
            target_entity="taxon_lactobacillus",
            intervention_type="probiotic",
            effect_direction="increased",
            provenance=valid_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
        
        # Modify to invalid value after creation
        intervention.properties["effect_direction"] = "invalid_direction"
        
        errors = validator.validate_direction(intervention)
        assert len(errors) == 1
        assert errors[0].field_name == "effect_direction"


class TestEvidenceStrengthValidation:
    """
    Test evidence strength validation.
    
    Requirement 14.4: System SHALL validate that evidence_strength values
    are in the set {"strong", "moderate", "weak", "conflicting"}
    """
    
    def test_valid_evidence_strength_strong(self, validator, valid_association):
        """Test that 'strong' evidence strength is valid."""
        valid_association.evidence_strength = "strong"
        errors = validator.validate_evidence_strength(valid_association)
        assert len(errors) == 0
    
    def test_valid_evidence_strength_moderate(self, validator, valid_association):
        """Test that 'moderate' evidence strength is valid."""
        valid_association.evidence_strength = "moderate"
        errors = validator.validate_evidence_strength(valid_association)
        assert len(errors) == 0
    
    def test_valid_evidence_strength_weak(self, validator, valid_association):
        """Test that 'weak' evidence strength is valid."""
        valid_association.evidence_strength = "weak"
        errors = validator.validate_evidence_strength(valid_association)
        assert len(errors) == 0
    
    def test_valid_evidence_strength_conflicting(self, validator, valid_association):
        """Test that 'conflicting' evidence strength is valid."""
        valid_association.evidence_strength = "conflicting"
        errors = validator.validate_evidence_strength(valid_association)
        assert len(errors) == 0
    
    def test_invalid_evidence_strength(self, validator, valid_association):
        """Test that invalid evidence strength fails validation."""
        valid_association.evidence_strength = "very_strong"
        errors = validator.validate_evidence_strength(valid_association)
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_value"
        assert errors[0].field_name == "evidence_strength"
        assert errors[0].invalid_value == "very_strong"


class TestBatchValidation:
    """
    Test batch validation of multiple relationships.
    
    Requirements: 14.1, 14.2, 14.3, 14.4
    """
    
    def test_all_valid_relationships(self, validator, valid_association, valid_provenance):
        """Test batch validation with all valid relationships."""
        relationships = [
            valid_association,
            create_association_relationship(
                source_entity="PMC789012",
                target_entity="taxon_escherichia_coli",
                direction="decreased",
                comparison="IBD vs healthy",
                statistical_measure="fold change",
                provenance=valid_provenance,
                evidence_strength="moderate",
                extraction_confidence=0.75,
                p_value=0.03,
            ),
        ]
        
        result = validator.validate_batch(relationships)
        
        assert result.total_count == 2
        assert result.valid_count == 2
        assert result.invalid_count == 0
        assert len(result.valid_relationships) == 2
        assert len(result.invalid_relationships) == 0
    
    def test_mixed_valid_invalid_relationships(self, validator, valid_association, valid_provenance):
        """Test batch validation with mix of valid and invalid relationships."""
        # Create valid relationship first, then modify to make it invalid
        invalid_rel = create_association_relationship(
            source_entity="PMC999999",
            target_entity="taxon_invalid",
            direction="increased",
            comparison="test",
            statistical_measure="test",
            provenance=valid_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
            p_value=0.5,  # Valid initially
        )
        
        # Modify to invalid p-value after creation
        invalid_rel.properties["p_value"] = 2.5
        
        relationships = [valid_association, invalid_rel]
        
        result = validator.validate_batch(relationships)
        
        assert result.total_count == 2
        assert result.valid_count == 1
        assert result.invalid_count == 1
        assert len(result.valid_relationships) == 1
        assert len(result.invalid_relationships) == 1
    
    def test_multiple_validation_errors(self, validator, valid_association):
        """Test relationship with multiple validation errors."""
        # Create relationship with multiple errors
        valid_association.extraction_confidence = 1.5  # Invalid
        valid_association.properties["p_value"] = -0.1  # Invalid
        valid_association.properties["direction"] = "invalid"  # Invalid
        valid_association.evidence_strength = "super_strong"  # Invalid
        
        is_valid, errors = validator.validate_relationship(valid_association)
        
        assert not is_valid
        assert len(errors) == 4  # All four validations should fail


class TestValidationQueue:
    """
    Test validation queue for storing invalid relationships.
    
    Requirement 14.5: System SHALL store relationships that fail validation
    in a separate validation queue for manual review
    """
    
    def test_store_invalid_relationships(self, validator, valid_association, temp_validation_queue):
        """Test storing invalid relationships in validation queue."""
        # Make relationship invalid
        valid_association.properties["p_value"] = 2.0
        
        is_valid, errors = validator.validate_relationship(valid_association)
        assert not is_valid
        
        # Store in queue
        validator.store_invalid_relationships([(valid_association, errors)])
        
        # Verify queue file exists and contains data
        assert Path(temp_validation_queue).exists()
        
        with open(temp_validation_queue, 'r') as f:
            queue = json.load(f)
        
        assert len(queue) == 1
        assert queue[0]["source_entity"] == "PMC123456"
        assert queue[0]["target_entity"] == "taxon_bacteroides_fragilis"
        assert len(queue[0]["validation_errors"]) == 1
        assert queue[0]["validation_errors"][0]["field_name"] == "p_value"
    
    def test_append_to_existing_queue(self, validator, valid_association, temp_validation_queue):
        """Test appending to existing validation queue."""
        # Create initial queue entry
        valid_association.properties["p_value"] = 2.0
        is_valid, errors = validator.validate_relationship(valid_association)
        validator.store_invalid_relationships([(valid_association, errors)])
        
        # Create second invalid relationship
        valid_association.properties["p_value"] = -0.5
        is_valid, errors = validator.validate_relationship(valid_association)
        validator.store_invalid_relationships([(valid_association, errors)])
        
        # Verify both entries in queue
        with open(temp_validation_queue, 'r') as f:
            queue = json.load(f)
        
        assert len(queue) == 2
    
    def test_validation_queue_stats(self, validator, valid_association, temp_validation_queue):
        """Test getting validation queue statistics."""
        # Initially empty
        stats = validator.get_validation_queue_stats()
        assert stats["queue_size"] == 0
        
        # Add invalid relationship
        valid_association.properties["p_value"] = 2.0
        is_valid, errors = validator.validate_relationship(valid_association)
        validator.store_invalid_relationships([(valid_association, errors)])
        
        # Check stats
        stats = validator.get_validation_queue_stats()
        assert stats["queue_size"] == 1
        assert stats["exists"] is True
        assert "out_of_range" in stats["error_type_counts"]
    
    def test_empty_invalid_relationships_list(self, validator):
        """Test storing empty list of invalid relationships."""
        validator.store_invalid_relationships([])
        # Should not raise error, just log info


class TestIntegrationScenarios:
    """
    Integration tests for realistic validation scenarios.
    """
    
    def test_realistic_batch_validation(self, validator, valid_provenance):
        """Test realistic batch with various validation issues."""
        # Create valid relationships first, then modify to make them invalid
        rel1 = create_association_relationship(
            source_entity="PMC111111",
            target_entity="taxon_valid",
            direction="increased",
            comparison="test",
            statistical_measure="LDA",
            provenance=valid_provenance,
            evidence_strength="strong",
            extraction_confidence=0.9,
            p_value=0.001,
        )
        
        rel2 = create_association_relationship(
            source_entity="PMC222222",
            target_entity="taxon_invalid_conf",
            direction="decreased",
            comparison="test",
            statistical_measure="LDA",
            provenance=valid_provenance,
            evidence_strength="moderate",
            extraction_confidence=0.9,  # Valid initially
        )
        # Modify to invalid confidence
        rel2.extraction_confidence = 1.5
        
        rel3 = create_association_relationship(
            source_entity="PMC333333",
            target_entity="taxon_invalid_p",
            direction="increased",
            comparison="test",
            statistical_measure="LDA",
            provenance=valid_provenance,
            evidence_strength="weak",
            extraction_confidence=0.7,
            p_value=0.5,  # Valid initially
        )
        # Modify to invalid p-value
        rel3.properties["p_value"] = 1.2
        
        relationships = [rel1, rel2, rel3]
        
        result = validator.validate_batch(relationships)
        
        assert result.total_count == 3
        assert result.valid_count == 1
        assert result.invalid_count == 2
        
        # Store invalid ones
        validator.store_invalid_relationships(result.invalid_relationships)
        
        stats = validator.get_validation_queue_stats()
        assert stats["queue_size"] == 2

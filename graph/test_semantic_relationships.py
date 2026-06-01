"""
graph/test_semantic_relationships.py
------------------------------------
Unit tests for semantic relationship data models.

Tests validate:
- Relationship type enums
- Property validation for each relationship type
- Confidence threshold enforcement (>= 0.5)
- Factory functions for creating relationships

Requirements: 2.1, 2.2, 2.3, 2.4
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from graph.semantic_relationships import (
    SemanticRelationship,
    RelationType,
    create_association_relationship,
    create_intervention_relationship,
    create_methodology_relationship,
)
from graph.provenance import ProvenanceMetadata


# Fixtures

@pytest.fixture
def sample_provenance() -> ProvenanceMetadata:
    """Create a sample provenance metadata for testing."""
    return ProvenanceMetadata(
        paper_id="PMC123456",
        section_type="results",
        source_sentence="Bacteroides fragilis was significantly increased in T2D patients (p=0.001).",
        sentence_offset=150,
        extraction_method="regex_ner",
        extraction_timestamp=datetime.now(timezone.utc),
        extractor_version="1.0.0",
        confidence_score=0.85,
        validation_status="unvalidated",
    )


# Test RelationType Enum

def test_relation_type_enum():
    """Test that RelationType enum has all required values."""
    assert RelationType.REPORTS_ASSOCIATION == "REPORTS_ASSOCIATION"
    assert RelationType.REPORTS_INTERVENTION_EFFECT == "REPORTS_INTERVENTION_EFFECT"
    assert RelationType.USES_METHODOLOGY == "USES_METHODOLOGY"
    
    # Test enum membership
    assert "REPORTS_ASSOCIATION" in [rt.value for rt in RelationType]
    assert "REPORTS_INTERVENTION_EFFECT" in [rt.value for rt in RelationType]
    assert "USES_METHODOLOGY" in [rt.value for rt in RelationType]


# Test REPORTS_ASSOCIATION Relationships

def test_create_association_relationship_valid(sample_provenance):
    """Test creating a valid REPORTS_ASSOCIATION relationship."""
    rel = create_association_relationship(
        source_entity="paper_123",
        target_entity="taxon_456",
        direction="increased",
        comparison="T2D vs healthy",
        statistical_measure="LDA score",
        provenance=sample_provenance,
        evidence_strength="strong",
        extraction_confidence=0.85,
        effect_size=3.2,
        p_value=0.001,
    )
    
    assert rel.source_entity == "paper_123"
    assert rel.target_entity == "taxon_456"
    assert rel.relation_type == RelationType.REPORTS_ASSOCIATION
    assert rel.properties["direction"] == "increased"
    assert rel.properties["comparison"] == "T2D vs healthy"
    assert rel.properties["statistical_measure"] == "LDA score"
    assert rel.properties["effect_size"] == 3.2
    assert rel.properties["p_value"] == 0.001
    assert rel.evidence_strength == "strong"
    assert rel.extraction_confidence == 0.85


def test_association_missing_required_properties(sample_provenance):
    """Test that REPORTS_ASSOCIATION fails without required properties."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            relation_type=RelationType.REPORTS_ASSOCIATION,
            properties={"direction": "increased"},  # Missing comparison, statistical_measure
            provenance=sample_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
    
    assert "missing" in str(exc_info.value).lower()


def test_association_invalid_direction(sample_provenance):
    """Test that REPORTS_ASSOCIATION rejects invalid direction values."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            relation_type=RelationType.REPORTS_ASSOCIATION,
            properties={
                "direction": "invalid_direction",
                "comparison": "T2D vs healthy",
                "statistical_measure": "LDA score",
            },
            provenance=sample_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
    
    assert "direction" in str(exc_info.value).lower()


def test_association_invalid_p_value(sample_provenance):
    """Test that REPORTS_ASSOCIATION rejects p_values outside [0.0, 1.0]."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            relation_type=RelationType.REPORTS_ASSOCIATION,
            properties={
                "direction": "increased",
                "comparison": "T2D vs healthy",
                "statistical_measure": "LDA score",
                "p_value": 1.5,  # Invalid: > 1.0
            },
            provenance=sample_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
    
    assert "p_value" in str(exc_info.value).lower()


def test_association_all_directions(sample_provenance):
    """Test that all valid direction values are accepted."""
    for direction in ["increased", "decreased", "no_change"]:
        rel = create_association_relationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            direction=direction,
            comparison="T2D vs healthy",
            statistical_measure="LDA score",
            provenance=sample_provenance,
            evidence_strength="moderate",
            extraction_confidence=0.75,
        )
        assert rel.properties["direction"] == direction


# Test REPORTS_INTERVENTION_EFFECT Relationships

def test_create_intervention_relationship_valid(sample_provenance):
    """Test creating a valid REPORTS_INTERVENTION_EFFECT relationship."""
    rel = create_intervention_relationship(
        source_entity="paper_123",
        target_entity="taxon_456",
        intervention_type="probiotic",
        effect_direction="increased",
        provenance=sample_provenance,
        evidence_strength="strong",
        extraction_confidence=0.90,
        duration="4 weeks",
        dosage="10^9 CFU/day",
        sample_size=120,
    )
    
    assert rel.source_entity == "paper_123"
    assert rel.target_entity == "taxon_456"
    assert rel.relation_type == RelationType.REPORTS_INTERVENTION_EFFECT
    assert rel.properties["intervention_type"] == "probiotic"
    assert rel.properties["effect_direction"] == "increased"
    assert rel.properties["duration"] == "4 weeks"
    assert rel.properties["dosage"] == "10^9 CFU/day"
    assert rel.properties["sample_size"] == 120
    assert rel.evidence_strength == "strong"
    assert rel.extraction_confidence == 0.90


def test_intervention_missing_required_properties(sample_provenance):
    """Test that REPORTS_INTERVENTION_EFFECT fails without required properties."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            relation_type=RelationType.REPORTS_INTERVENTION_EFFECT,
            properties={"intervention_type": "probiotic"},  # Missing effect_direction
            provenance=sample_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
    
    assert "missing" in str(exc_info.value).lower()


def test_intervention_invalid_type(sample_provenance):
    """Test that REPORTS_INTERVENTION_EFFECT rejects invalid intervention types."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            relation_type=RelationType.REPORTS_INTERVENTION_EFFECT,
            properties={
                "intervention_type": "invalid_type",
                "effect_direction": "increased",
            },
            provenance=sample_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
    
    assert "intervention_type" in str(exc_info.value).lower()


def test_intervention_all_types(sample_provenance):
    """Test that all valid intervention types are accepted."""
    for intervention_type in ["probiotic", "FMT", "diet", "antibiotic", "prebiotic", "synbiotic", "other"]:
        rel = create_intervention_relationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            intervention_type=intervention_type,
            effect_direction="increased",
            provenance=sample_provenance,
            evidence_strength="moderate",
            extraction_confidence=0.75,
        )
        assert rel.properties["intervention_type"] == intervention_type


def test_intervention_invalid_sample_size(sample_provenance):
    """Test that REPORTS_INTERVENTION_EFFECT rejects invalid sample sizes."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            relation_type=RelationType.REPORTS_INTERVENTION_EFFECT,
            properties={
                "intervention_type": "probiotic",
                "effect_direction": "increased",
                "sample_size": -10,  # Invalid: negative
            },
            provenance=sample_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
    
    assert "sample_size" in str(exc_info.value).lower()


# Test USES_METHODOLOGY Relationships

def test_create_methodology_relationship_valid(sample_provenance):
    """Test creating a valid USES_METHODOLOGY relationship."""
    rel = create_methodology_relationship(
        source_entity="paper_123",
        target_entity="method_789",
        method_name="16S rRNA sequencing",
        provenance=sample_provenance,
        evidence_strength="strong",
        extraction_confidence=0.95,
        sequencing_platform="Illumina",
        sample_size=50,
        data_availability="open",
    )
    
    assert rel.source_entity == "paper_123"
    assert rel.target_entity == "method_789"
    assert rel.relation_type == RelationType.USES_METHODOLOGY
    assert rel.properties["method_name"] == "16S rRNA sequencing"
    assert rel.properties["sequencing_platform"] == "Illumina"
    assert rel.properties["sample_size"] == 50
    assert rel.properties["data_availability"] == "open"
    assert rel.evidence_strength == "strong"
    assert rel.extraction_confidence == 0.95


def test_methodology_missing_required_properties(sample_provenance):
    """Test that USES_METHODOLOGY fails without required properties."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="method_789",
            relation_type=RelationType.USES_METHODOLOGY,
            properties={},  # Missing method_name
            provenance=sample_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
    
    assert "missing" in str(exc_info.value).lower()


def test_methodology_invalid_sample_size(sample_provenance):
    """Test that USES_METHODOLOGY rejects invalid sample sizes."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="method_789",
            relation_type=RelationType.USES_METHODOLOGY,
            properties={
                "method_name": "16S rRNA",
                "sample_size": 0,  # Invalid: must be positive
            },
            provenance=sample_provenance,
            evidence_strength="strong",
            extraction_confidence=0.85,
        )
    
    assert "sample_size" in str(exc_info.value).lower()


# Test Confidence Threshold Enforcement (Requirement 2.4)

def test_confidence_threshold_enforcement(sample_provenance):
    """
    Test that relationships with confidence < 0.5 are rejected.
    
    Requirement 2.4: System SHALL only create relationships with
    extraction confidence >= 0.5
    """
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            relation_type=RelationType.REPORTS_ASSOCIATION,
            properties={
                "direction": "increased",
                "comparison": "T2D vs healthy",
                "statistical_measure": "LDA score",
            },
            provenance=sample_provenance,
            evidence_strength="weak",
            extraction_confidence=0.4,  # Below threshold
        )
    
    assert "0.5" in str(exc_info.value)
    assert "2.4" in str(exc_info.value)  # Requirement reference


def test_confidence_at_threshold(sample_provenance):
    """Test that confidence exactly at 0.5 is accepted."""
    rel = create_association_relationship(
        source_entity="paper_123",
        target_entity="taxon_456",
        direction="increased",
        comparison="T2D vs healthy",
        statistical_measure="LDA score",
        provenance=sample_provenance,
        evidence_strength="weak",
        extraction_confidence=0.5,  # Exactly at threshold
    )
    assert rel.extraction_confidence == 0.5


def test_confidence_above_threshold(sample_provenance):
    """Test that confidence > 0.5 is accepted."""
    rel = create_association_relationship(
        source_entity="paper_123",
        target_entity="taxon_456",
        direction="increased",
        comparison="T2D vs healthy",
        statistical_measure="LDA score",
        provenance=sample_provenance,
        evidence_strength="strong",
        extraction_confidence=0.95,
    )
    assert rel.extraction_confidence == 0.95


# Test Evidence Strength Validation

def test_invalid_evidence_strength(sample_provenance):
    """Test that invalid evidence_strength values are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        SemanticRelationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            relation_type=RelationType.REPORTS_ASSOCIATION,
            properties={
                "direction": "increased",
                "comparison": "T2D vs healthy",
                "statistical_measure": "LDA score",
            },
            provenance=sample_provenance,
            evidence_strength="invalid_strength",
            extraction_confidence=0.85,
        )
    
    assert "evidence_strength" in str(exc_info.value).lower()


def test_all_evidence_strengths(sample_provenance):
    """Test that all valid evidence_strength values are accepted."""
    for strength in ["strong", "moderate", "weak", "conflicting"]:
        rel = create_association_relationship(
            source_entity="paper_123",
            target_entity="taxon_456",
            direction="increased",
            comparison="T2D vs healthy",
            statistical_measure="LDA score",
            provenance=sample_provenance,
            evidence_strength=strength,
            extraction_confidence=0.75,
        )
        assert rel.evidence_strength == strength


# Test Provenance Integration

def test_provenance_embedded_in_relationship(sample_provenance):
    """Test that provenance metadata is correctly embedded in relationships."""
    rel = create_association_relationship(
        source_entity="paper_123",
        target_entity="taxon_456",
        direction="increased",
        comparison="T2D vs healthy",
        statistical_measure="LDA score",
        provenance=sample_provenance,
        evidence_strength="strong",
        extraction_confidence=0.85,
    )
    
    assert rel.provenance.paper_id == "PMC123456"
    assert rel.provenance.section_type == "results"
    assert rel.provenance.extraction_method == "regex_ner"
    assert rel.provenance.confidence_score == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

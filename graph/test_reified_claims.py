"""
Unit tests for reified claim data models.

Tests Requirements:
- 4.1: Reified claim node creation
- 4.2: Separate supporting/contradicting paper lists with no overlap
- 4.3: Consensus confidence validation
- 4.4: Effect direction consistency validation
- 4.5: Temporal evolution tracking
- 4.6: Conflicting evidence handling
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from graph.reified_claims import (
    EvidenceStrength,
    ScientificClaim,
    ReifiedClaimNode
)


class TestEvidenceStrength:
    """Test EvidenceStrength enum."""
    
    def test_evidence_strength_values(self):
        """Test that all expected evidence strength values exist."""
        assert EvidenceStrength.STRONG == "strong"
        assert EvidenceStrength.MODERATE == "moderate"
        assert EvidenceStrength.WEAK == "weak"
        assert EvidenceStrength.CONFLICTING == "conflicting"


class TestScientificClaim:
    """Test ScientificClaim model."""
    
    def test_valid_scientific_claim(self):
        """Test creating a valid scientific claim (Requirement 4.1)."""
        claim = ScientificClaim(
            claim_id="claim-001",
            claim_type="association",
            subject_entity="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_papers=["paper1", "paper2"],
            contradicting_papers=[],
            total_sample_size=150,
            evidence_strength=EvidenceStrength.STRONG,
            consensus_confidence=0.85,
            effect_direction_consistency=0.90,
            first_reported="2024-01-01",
            last_updated="2024-06-01"
        )
        
        assert claim.claim_id == "claim-001"
        assert claim.claim_type == "association"
        assert len(claim.supporting_papers) == 2
        assert claim.consensus_confidence == 0.85
    
    def test_consensus_confidence_validation_lower_bound(self):
        """Test consensus_confidence must be >= 0.0 (Requirement 4.3)."""
        with pytest.raises(ValidationError) as exc_info:
            ScientificClaim(
                claim_id="claim-002",
                claim_type="association",
                subject_entity="Taxon A",
                predicate="increases_in",
                object_entity="Disease B",
                supporting_papers=["paper1"],
                contradicting_papers=[],
                total_sample_size=50,
                evidence_strength=EvidenceStrength.WEAK,
                consensus_confidence=-0.1,  # Invalid: below 0.0
                effect_direction_consistency=0.75,
                first_reported="2024-01-01",
                last_updated="2024-01-01"
            )
        
        assert "consensus_confidence" in str(exc_info.value)
    
    def test_consensus_confidence_validation_upper_bound(self):
        """Test consensus_confidence must be <= 1.0 (Requirement 4.3)."""
        with pytest.raises(ValidationError) as exc_info:
            ScientificClaim(
                claim_id="claim-003",
                claim_type="association",
                subject_entity="Taxon A",
                predicate="increases_in",
                object_entity="Disease B",
                supporting_papers=["paper1"],
                contradicting_papers=[],
                total_sample_size=50,
                evidence_strength=EvidenceStrength.WEAK,
                consensus_confidence=1.5,  # Invalid: above 1.0
                effect_direction_consistency=0.75,
                first_reported="2024-01-01",
                last_updated="2024-01-01"
            )
        
        assert "consensus_confidence" in str(exc_info.value)
    
    def test_effect_direction_consistency_validation_lower_bound(self):
        """Test effect_direction_consistency must be >= 0.0 (Requirement 4.4)."""
        with pytest.raises(ValidationError) as exc_info:
            ScientificClaim(
                claim_id="claim-004",
                claim_type="association",
                subject_entity="Taxon A",
                predicate="increases_in",
                object_entity="Disease B",
                supporting_papers=["paper1"],
                contradicting_papers=[],
                total_sample_size=50,
                evidence_strength=EvidenceStrength.WEAK,
                consensus_confidence=0.75,
                effect_direction_consistency=-0.1,  # Invalid: below 0.0
                first_reported="2024-01-01",
                last_updated="2024-01-01"
            )
        
        assert "effect_direction_consistency" in str(exc_info.value)
    
    def test_effect_direction_consistency_validation_upper_bound(self):
        """Test effect_direction_consistency must be <= 1.0 (Requirement 4.4)."""
        with pytest.raises(ValidationError) as exc_info:
            ScientificClaim(
                claim_id="claim-005",
                claim_type="association",
                subject_entity="Taxon A",
                predicate="increases_in",
                object_entity="Disease B",
                supporting_papers=["paper1"],
                contradicting_papers=[],
                total_sample_size=50,
                evidence_strength=EvidenceStrength.WEAK,
                consensus_confidence=0.75,
                effect_direction_consistency=1.1,  # Invalid: above 1.0
                first_reported="2024-01-01",
                last_updated="2024-01-01"
            )
        
        assert "effect_direction_consistency" in str(exc_info.value)
    
    def test_no_overlap_between_paper_lists(self):
        """Test supporting and contradicting papers must not overlap (Requirement 4.2)."""
        with pytest.raises(ValidationError) as exc_info:
            ScientificClaim(
                claim_id="claim-006",
                claim_type="association",
                subject_entity="Taxon A",
                predicate="increases_in",
                object_entity="Disease B",
                supporting_papers=["paper1", "paper2"],
                contradicting_papers=["paper2", "paper3"],  # paper2 overlaps
                total_sample_size=50,
                evidence_strength=EvidenceStrength.CONFLICTING,
                consensus_confidence=0.60,
                effect_direction_consistency=0.50,
                first_reported="2024-01-01",
                last_updated="2024-01-01"
            )
        
        assert "cannot appear in both" in str(exc_info.value).lower()
    
    def test_temporal_ordering_valid(self):
        """Test first_reported <= last_updated is valid (Requirement 4.5)."""
        claim = ScientificClaim(
            claim_id="claim-007",
            claim_type="association",
            subject_entity="Taxon A",
            predicate="increases_in",
            object_entity="Disease B",
            supporting_papers=["paper1"],
            contradicting_papers=[],
            total_sample_size=50,
            evidence_strength=EvidenceStrength.WEAK,
            consensus_confidence=0.75,
            effect_direction_consistency=0.80,
            first_reported="2024-01-01",
            last_updated="2024-06-01"
        )
        
        assert claim.first_reported == "2024-01-01"
        assert claim.last_updated == "2024-06-01"
    
    def test_temporal_ordering_equal_dates(self):
        """Test first_reported = last_updated is valid (Requirement 4.5)."""
        claim = ScientificClaim(
            claim_id="claim-008",
            claim_type="association",
            subject_entity="Taxon A",
            predicate="increases_in",
            object_entity="Disease B",
            supporting_papers=["paper1"],
            contradicting_papers=[],
            total_sample_size=50,
            evidence_strength=EvidenceStrength.WEAK,
            consensus_confidence=0.75,
            effect_direction_consistency=0.80,
            first_reported="2024-01-01",
            last_updated="2024-01-01"
        )
        
        assert claim.first_reported == claim.last_updated
    
    def test_temporal_ordering_invalid(self):
        """Test first_reported > last_updated is invalid (Requirement 4.5)."""
        with pytest.raises(ValidationError) as exc_info:
            ScientificClaim(
                claim_id="claim-009",
                claim_type="association",
                subject_entity="Taxon A",
                predicate="increases_in",
                object_entity="Disease B",
                supporting_papers=["paper1"],
                contradicting_papers=[],
                total_sample_size=50,
                evidence_strength=EvidenceStrength.WEAK,
                consensus_confidence=0.75,
                effect_direction_consistency=0.80,
                first_reported="2024-06-01",
                last_updated="2024-01-01"  # Invalid: before first_reported
            )
        
        assert "first_reported" in str(exc_info.value).lower()
    
    def test_claim_type_validation(self):
        """Test claim_type must be valid."""
        with pytest.raises(ValidationError) as exc_info:
            ScientificClaim(
                claim_id="claim-010",
                claim_type="invalid_type",  # Invalid claim type
                subject_entity="Taxon A",
                predicate="increases_in",
                object_entity="Disease B",
                supporting_papers=["paper1"],
                contradicting_papers=[],
                total_sample_size=50,
                evidence_strength=EvidenceStrength.WEAK,
                consensus_confidence=0.75,
                effect_direction_consistency=0.80,
                first_reported="2024-01-01",
                last_updated="2024-01-01"
            )
        
        assert "claim_type" in str(exc_info.value)
    
    def test_duplicate_paper_ids_in_supporting(self):
        """Test supporting_papers must not contain duplicates (Requirement 4.2)."""
        with pytest.raises(ValidationError) as exc_info:
            ScientificClaim(
                claim_id="claim-011",
                claim_type="association",
                subject_entity="Taxon A",
                predicate="increases_in",
                object_entity="Disease B",
                supporting_papers=["paper1", "paper1"],  # Duplicate
                contradicting_papers=[],
                total_sample_size=50,
                evidence_strength=EvidenceStrength.WEAK,
                consensus_confidence=0.75,
                effect_direction_consistency=0.80,
                first_reported="2024-01-01",
                last_updated="2024-01-01"
            )
        
        assert "duplicates" in str(exc_info.value).lower()


class TestReifiedClaimNode:
    """Test ReifiedClaimNode model."""
    
    def test_valid_reified_claim_node(self):
        """Test creating a valid reified claim node (Requirement 4.1)."""
        node = ReifiedClaimNode(
            node_id="node-001",
            node_type="ScientificClaim",
            claim_type="association",
            subject_entity="taxon-123",
            predicate="increases_in",
            object_entity="disease-456",
            supporting_paper_ids=["paper1", "paper2"],
            contradicting_paper_ids=[],
            total_sample_size=150,
            evidence_strength="strong",
            consensus_confidence=0.85,
            effect_direction_consistency=0.90,
            first_reported=datetime(2024, 1, 1),
            last_updated=datetime(2024, 6, 1)
        )
        
        assert node.node_id == "node-001"
        assert node.node_type == "ScientificClaim"
        assert len(node.supporting_paper_ids) == 2
    
    def test_consensus_metrics_validation(self):
        """Test consensus metrics validation (Requirements 4.3, 4.4)."""
        # Valid at boundaries
        node = ReifiedClaimNode(
            node_id="node-002",
            claim_type="association",
            subject_entity="taxon-123",
            predicate="increases_in",
            object_entity="disease-456",
            supporting_paper_ids=["paper1"],
            contradicting_paper_ids=[],
            total_sample_size=50,
            evidence_strength="weak",
            consensus_confidence=0.0,  # Valid: at lower bound
            effect_direction_consistency=1.0,  # Valid: at upper bound
            first_reported=datetime(2024, 1, 1),
            last_updated=datetime(2024, 1, 1)
        )
        
        assert node.consensus_confidence == 0.0
        assert node.effect_direction_consistency == 1.0
    
    def test_no_overlap_between_paper_ids(self):
        """Test supporting and contradicting paper IDs must not overlap (Requirement 4.2)."""
        with pytest.raises(ValidationError) as exc_info:
            ReifiedClaimNode(
                node_id="node-003",
                claim_type="association",
                subject_entity="taxon-123",
                predicate="increases_in",
                object_entity="disease-456",
                supporting_paper_ids=["paper1", "paper2"],
                contradicting_paper_ids=["paper2", "paper3"],  # paper2 overlaps
                total_sample_size=50,
                evidence_strength="conflicting",
                consensus_confidence=0.60,
                effect_direction_consistency=0.50,
                first_reported=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1)
            )
        
        assert "cannot appear in both" in str(exc_info.value).lower()
    
    def test_temporal_ordering_with_datetime(self):
        """Test temporal ordering with datetime objects (Requirement 4.5)."""
        node = ReifiedClaimNode(
            node_id="node-004",
            claim_type="association",
            subject_entity="taxon-123",
            predicate="increases_in",
            object_entity="disease-456",
            supporting_paper_ids=["paper1"],
            contradicting_paper_ids=[],
            total_sample_size=50,
            evidence_strength="weak",
            consensus_confidence=0.75,
            effect_direction_consistency=0.80,
            first_reported=datetime(2024, 1, 1),
            last_updated=datetime(2024, 6, 1)
        )
        
        assert node.first_reported < node.last_updated
    
    def test_temporal_ordering_invalid_datetime(self):
        """Test first_reported > last_updated is invalid with datetime (Requirement 4.5)."""
        with pytest.raises(ValidationError) as exc_info:
            ReifiedClaimNode(
                node_id="node-005",
                claim_type="association",
                subject_entity="taxon-123",
                predicate="increases_in",
                object_entity="disease-456",
                supporting_paper_ids=["paper1"],
                contradicting_paper_ids=[],
                total_sample_size=50,
                evidence_strength="weak",
                consensus_confidence=0.75,
                effect_direction_consistency=0.80,
                first_reported=datetime(2024, 6, 1),
                last_updated=datetime(2024, 1, 1)  # Invalid: before first_reported
            )
        
        assert "first_reported" in str(exc_info.value).lower()
    
    def test_evidence_strength_validation(self):
        """Test evidence_strength validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReifiedClaimNode(
                node_id="node-006",
                claim_type="association",
                subject_entity="taxon-123",
                predicate="increases_in",
                object_entity="disease-456",
                supporting_paper_ids=["paper1"],
                contradicting_paper_ids=[],
                total_sample_size=50,
                evidence_strength="invalid_strength",  # Invalid
                consensus_confidence=0.75,
                effect_direction_consistency=0.80,
                first_reported=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1)
            )
        
        assert "evidence_strength" in str(exc_info.value)
    
    def test_meta_analysis_method_validation(self):
        """Test meta_analysis_method validation."""
        # Valid method
        node = ReifiedClaimNode(
            node_id="node-007",
            claim_type="association",
            subject_entity="taxon-123",
            predicate="increases_in",
            object_entity="disease-456",
            supporting_paper_ids=["paper1"],
            contradicting_paper_ids=[],
            total_sample_size=50,
            evidence_strength="strong",
            consensus_confidence=0.85,
            effect_direction_consistency=0.90,
            first_reported=datetime(2024, 1, 1),
            last_updated=datetime(2024, 1, 1),
            meta_analysis_performed=True,
            meta_analysis_method="random_effects"
        )
        
        assert node.meta_analysis_method == "random_effects"
        
        # Invalid method
        with pytest.raises(ValidationError) as exc_info:
            ReifiedClaimNode(
                node_id="node-008",
                claim_type="association",
                subject_entity="taxon-123",
                predicate="increases_in",
                object_entity="disease-456",
                supporting_paper_ids=["paper1"],
                contradicting_paper_ids=[],
                total_sample_size=50,
                evidence_strength="strong",
                consensus_confidence=0.85,
                effect_direction_consistency=0.90,
                first_reported=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1),
                meta_analysis_method="invalid_method"
            )
        
        assert "meta_analysis_method" in str(exc_info.value)
    
    def test_conflicting_evidence_handling(self):
        """Test handling of conflicting evidence (Requirement 4.6)."""
        node = ReifiedClaimNode(
            node_id="node-009",
            claim_type="association",
            subject_entity="taxon-123",
            predicate="increases_in",
            object_entity="disease-456",
            supporting_paper_ids=["paper1", "paper2"],
            contradicting_paper_ids=["paper3", "paper4"],
            total_sample_size=100,
            evidence_strength="conflicting",  # Requirement 4.6
            consensus_confidence=0.55,
            effect_direction_consistency=0.50,
            first_reported=datetime(2024, 1, 1),
            last_updated=datetime(2024, 6, 1)
        )
        
        assert node.evidence_strength == "conflicting"
        assert len(node.supporting_paper_ids) == 2
        assert len(node.contradicting_paper_ids) == 2

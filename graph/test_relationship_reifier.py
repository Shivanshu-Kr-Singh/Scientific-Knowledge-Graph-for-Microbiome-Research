"""
Unit tests for RelationshipReifier class.

Tests the reification of relationships into first-class claim entities,
including claim creation, evidence aggregation, and conflict detection.

Requirements: 4.1, 4.3, 4.5, 4.6, 9.1
"""

import pytest
from datetime import datetime, timezone, timedelta
import uuid

from graph.relationship_reifier import RelationshipReifier
from graph.reified_claims import ScientificClaim, EvidenceStrength
from graph.provenance import ProvenanceMetadata


@pytest.fixture
def reifier():
    """Create a RelationshipReifier instance for testing."""
    return RelationshipReifier()


@pytest.fixture
def sample_provenance():
    """Create sample provenance metadata for testing."""
    return ProvenanceMetadata(
        paper_id="10.1234/test.2024",
        section_type="results",
        source_sentence="Bacteroides fragilis was significantly increased in T2D patients.",
        extraction_method="regex_ner",
        extraction_timestamp=datetime.now(timezone.utc),
        extractor_version="1.0",
        confidence_score=0.85,
    )


@pytest.fixture
def multiple_provenance():
    """Create multiple provenance metadata instances for testing."""
    base_time = datetime.now(timezone.utc)
    return [
        ProvenanceMetadata(
            paper_id="10.1234/paper1.2024",
            section_type="results",
            source_sentence="Bacteroides fragilis increased in T2D.",
            extraction_method="regex_ner",
            extraction_timestamp=base_time,
            extractor_version="1.0",
            confidence_score=0.85,
        ),
        ProvenanceMetadata(
            paper_id="10.1234/paper2.2024",
            section_type="results",
            source_sentence="B. fragilis abundance was higher in diabetic patients.",
            extraction_method="biobert_ner",
            extraction_timestamp=base_time + timedelta(days=30),
            extractor_version="1.0",
            confidence_score=0.90,
        ),
        ProvenanceMetadata(
            paper_id="10.1234/paper3.2024",
            section_type="results",
            source_sentence="Elevated B. fragilis observed in T2D cohort.",
            extraction_method="llm_extractor_v1.2",
            extraction_timestamp=base_time + timedelta(days=60),
            extractor_version="1.2",
            confidence_score=0.80,
        ),
    ]


class TestReifyClaim:
    """Test the reify_claim method."""
    
    def test_reify_claim_basic(self, reifier, sample_provenance):
        """
        Test basic claim reification with single evidence.
        
        Requirement 4.1: Create reified claim node aggregating supporting evidence
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        # Verify claim structure
        assert claim.claim_id is not None
        assert claim.claim_type == "association"
        assert claim.subject_entity == "Bacteroides fragilis"
        assert claim.predicate == "associated_with_increased_abundance"
        assert claim.object_entity == "Type 2 Diabetes"
        
        # Verify evidence aggregation
        assert len(claim.supporting_papers) == 1
        assert claim.supporting_papers[0] == sample_provenance.paper_id
        assert len(claim.contradicting_papers) == 0
        
        # Verify consensus metrics
        assert 0.0 <= claim.consensus_confidence <= 1.0
        assert claim.effect_direction_consistency == 1.0
    
    def test_reify_claim_generates_unique_id(self, reifier, sample_provenance):
        """
        Test that each claim gets a unique UUID.
        
        Requirement 4.1: Generate unique claim_id using UUID
        """
        claim1 = reifier.reify_claim(
            subject="Taxon A",
            predicate="increases_in",
            object_entity="Disease X",
            supporting_evidence=[sample_provenance],
        )
        
        claim2 = reifier.reify_claim(
            subject="Taxon A",
            predicate="increases_in",
            object_entity="Disease X",
            supporting_evidence=[sample_provenance],
        )
        
        # Verify IDs are different
        assert claim1.claim_id != claim2.claim_id
        
        # Verify IDs are valid UUIDs
        uuid.UUID(claim1.claim_id)
        uuid.UUID(claim2.claim_id)
    
    def test_reify_claim_multiple_evidence(self, reifier, multiple_provenance):
        """
        Test claim reification with multiple pieces of evidence.
        
        Requirement 4.1: Aggregate evidence from multiple papers
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=multiple_provenance,
        )
        
        # Verify all papers are included
        assert len(claim.supporting_papers) == 3
        expected_papers = {prov.paper_id for prov in multiple_provenance}
        assert set(claim.supporting_papers) == expected_papers
    
    def test_reify_claim_consensus_confidence(self, reifier, multiple_provenance):
        """
        Test consensus confidence calculation as weighted average.
        
        Requirement 4.3: Calculate consensus_confidence as weighted average
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=multiple_provenance,
        )
        
        # Verify consensus confidence is in valid range
        assert 0.0 <= claim.consensus_confidence <= 1.0
        
        # Verify it's a weighted average (should be close to mean of confidences)
        confidences = [prov.confidence_score for prov in multiple_provenance]
        mean_confidence = sum(confidences) / len(confidences)
        
        # Consensus should be reasonably close to mean (within 0.1)
        assert abs(claim.consensus_confidence - mean_confidence) < 0.1
    
    def test_reify_claim_temporal_tracking(self, reifier, multiple_provenance):
        """
        Test temporal tracking of first_reported and last_updated.
        
        Requirement 4.5: Track temporal evolution with first_reported and last_updated
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=multiple_provenance,
        )
        
        # Verify temporal fields are set
        assert claim.first_reported is not None
        assert claim.last_updated is not None
        
        # Verify first_reported <= last_updated
        first = datetime.fromisoformat(claim.first_reported)
        last = datetime.fromisoformat(claim.last_updated)
        assert first <= last
        
        # Verify first_reported is earliest timestamp
        earliest = min(prov.extraction_timestamp for prov in multiple_provenance)
        assert first == earliest
        
        # Verify last_updated is latest timestamp
        latest = max(prov.extraction_timestamp for prov in multiple_provenance)
        assert last == latest
    
    def test_reify_claim_empty_subject_raises_error(self, reifier, sample_provenance):
        """Test that empty subject raises ValueError."""
        with pytest.raises(ValueError, match="subject must be a non-empty string"):
            reifier.reify_claim(
                subject="",
                predicate="increases_in",
                object_entity="Disease X",
                supporting_evidence=[sample_provenance],
            )
    
    def test_reify_claim_empty_predicate_raises_error(self, reifier, sample_provenance):
        """Test that empty predicate raises ValueError."""
        with pytest.raises(ValueError, match="predicate must be a non-empty string"):
            reifier.reify_claim(
                subject="Taxon A",
                predicate="",
                object_entity="Disease X",
                supporting_evidence=[sample_provenance],
            )
    
    def test_reify_claim_empty_object_raises_error(self, reifier, sample_provenance):
        """Test that empty object raises ValueError."""
        with pytest.raises(ValueError, match="object_entity must be a non-empty string"):
            reifier.reify_claim(
                subject="Taxon A",
                predicate="increases_in",
                object_entity="",
                supporting_evidence=[sample_provenance],
            )
    
    def test_reify_claim_empty_evidence_raises_error(self, reifier):
        """Test that empty evidence list raises ValueError."""
        with pytest.raises(ValueError, match="supporting_evidence must contain at least one"):
            reifier.reify_claim(
                subject="Taxon A",
                predicate="increases_in",
                object_entity="Disease X",
                supporting_evidence=[],
            )
    
    def test_reify_claim_low_confidence_raises_error(self, reifier):
        """Test that evidence with confidence < 0.5 raises ValueError."""
        low_confidence_provenance = ProvenanceMetadata(
            paper_id="10.1234/test.2024",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.3,  # Below threshold
        )
        
        with pytest.raises(ValueError, match="confidence_score >= 0.5"):
            reifier.reify_claim(
                subject="Taxon A",
                predicate="increases_in",
                object_entity="Disease X",
                supporting_evidence=[low_confidence_provenance],
            )


class TestUpdateClaimWithNewEvidence:
    """Test the update_claim_with_new_evidence method."""
    
    def test_update_claim_with_supporting_evidence(self, reifier, sample_provenance):
        """
        Test updating claim with new supporting evidence.
        
        Requirement 4.5: Update claim with new evidence
        """
        # Create initial claim
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        # Create new supporting evidence
        new_evidence = ProvenanceMetadata(
            paper_id="10.1234/new_paper.2024",
            section_type="results",
            source_sentence="B. fragilis elevated in T2D.",
            extraction_method="biobert_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.88,
        )
        
        # Update claim
        updated_claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=new_evidence,
            supports=True,
        )
        
        # Verify new paper is added to supporting_papers
        assert len(updated_claim.supporting_papers) == 2
        assert new_evidence.paper_id in updated_claim.supporting_papers
        assert len(updated_claim.contradicting_papers) == 0
    
    def test_update_claim_with_contradicting_evidence(self, reifier, sample_provenance):
        """
        Test updating claim with contradicting evidence.
        
        Requirement 4.6: Update evidence_strength when contradicting evidence is added
        """
        # Create initial claim
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        # Create contradicting evidence
        contradicting_evidence = ProvenanceMetadata(
            paper_id="10.1234/contradicting.2024",
            section_type="results",
            source_sentence="B. fragilis decreased in T2D.",
            extraction_method="biobert_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.82,
        )
        
        # Update claim
        updated_claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=contradicting_evidence,
            supports=False,
        )
        
        # Verify new paper is added to contradicting_papers
        assert len(updated_claim.contradicting_papers) == 1
        assert contradicting_evidence.paper_id in updated_claim.contradicting_papers
        
        # Verify evidence_strength changed to conflicting
        assert updated_claim.evidence_strength == EvidenceStrength.CONFLICTING
    
    def test_update_claim_recalculates_consensus_confidence(self, reifier, sample_provenance):
        """
        Test that consensus confidence is recalculated.
        
        Requirement 4.5: Recalculate consensus_confidence when new evidence is added
        """
        # Create initial claim
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        original_confidence = claim.consensus_confidence
        
        # Add new supporting evidence
        new_evidence = ProvenanceMetadata(
            paper_id="10.1234/new_paper.2024",
            section_type="results",
            source_sentence="B. fragilis elevated in T2D.",
            extraction_method="biobert_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.95,  # High confidence
        )
        
        updated_claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=new_evidence,
            supports=True,
        )
        
        # Verify consensus confidence changed
        assert updated_claim.consensus_confidence != original_confidence
        assert 0.0 <= updated_claim.consensus_confidence <= 1.0
    
    def test_update_claim_updates_last_updated_timestamp(self, reifier, sample_provenance):
        """
        Test that last_updated timestamp is updated.
        
        Requirement 4.5: Update last_updated timestamp to current time
        """
        # Create initial claim
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        original_last_updated = claim.last_updated
        
        # Wait a moment to ensure timestamp difference
        import time
        time.sleep(0.1)
        
        # Add new evidence
        new_evidence = ProvenanceMetadata(
            paper_id="10.1234/new_paper.2024",
            section_type="results",
            source_sentence="B. fragilis elevated in T2D.",
            extraction_method="biobert_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.88,
        )
        
        updated_claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=new_evidence,
            supports=True,
        )
        
        # Verify last_updated changed
        assert updated_claim.last_updated != original_last_updated
        
        # Verify last_updated is more recent
        original_time = datetime.fromisoformat(original_last_updated)
        updated_time = datetime.fromisoformat(updated_claim.last_updated)
        assert updated_time > original_time
    
    def test_update_claim_duplicate_paper_ignored(self, reifier, sample_provenance):
        """Test that duplicate paper IDs are ignored."""
        # Create initial claim
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        # Try to add the same paper again
        updated_claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=sample_provenance,
            supports=True,
        )
        
        # Verify claim unchanged
        assert len(updated_claim.supporting_papers) == len(claim.supporting_papers)
        assert updated_claim.claim_id == claim.claim_id


class TestDetectConflictingClaims:
    """Test the detect_conflicting_claims method."""
    
    def test_detect_conflicting_claims_increased_vs_decreased(self, reifier, sample_provenance):
        """
        Test detection of conflicting claims with opposite directions.
        
        Requirement 4.6: Detect conflicting claims
        Requirement 9.1: Support conflicting evidence detection
        """
        # Create claim with "increased" predicate
        claim1 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        # Create claim with "decreased" predicate
        claim2 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_decreased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        # Detect conflicts
        conflicts = reifier.detect_conflicting_claims([claim1, claim2])
        
        # Verify conflict detected
        assert len(conflicts) == 1
        assert (claim1, claim2) in conflicts or (claim2, claim1) in conflicts
    
    def test_detect_conflicting_claims_same_subject_object(self, reifier, sample_provenance):
        """
        Test that conflicts are only detected for same subject and object.
        
        Requirement 4.6: Only return pairs with same subject and object
        """
        # Create claims with different subjects
        claim1 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        claim2 = reifier.reify_claim(
            subject="Lactobacillus reuteri",  # Different subject
            predicate="associated_with_decreased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        # Detect conflicts
        conflicts = reifier.detect_conflicting_claims([claim1, claim2])
        
        # Verify no conflict detected (different subjects)
        assert len(conflicts) == 0
    
    def test_detect_conflicting_claims_no_conflicts(self, reifier, sample_provenance):
        """
        Test that no conflicts are detected when claims agree.
        
        Requirement 4.6: Empty list if no conflicts found
        """
        # Create two claims with same direction
        claim1 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        claim2 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
        )
        
        # Detect conflicts
        conflicts = reifier.detect_conflicting_claims([claim1, claim2])
        
        # Verify no conflicts
        assert len(conflicts) == 0
    
    def test_detect_conflicting_claims_empty_list(self, reifier):
        """Test that empty list returns no conflicts."""
        conflicts = reifier.detect_conflicting_claims([])
        assert len(conflicts) == 0
    
    def test_detect_conflicting_claims_positive_vs_negative(self, reifier, sample_provenance):
        """Test detection of positive vs negative effect conflicts."""
        claim1 = reifier.reify_claim(
            subject="Probiotic X",
            predicate="has_positive_effect_on",
            object_entity="Gut Health",
            supporting_evidence=[sample_provenance],
        )
        
        claim2 = reifier.reify_claim(
            subject="Probiotic X",
            predicate="has_negative_effect_on",
            object_entity="Gut Health",
            supporting_evidence=[sample_provenance],
        )
        
        conflicts = reifier.detect_conflicting_claims([claim1, claim2])
        
        # Verify conflict detected
        assert len(conflicts) == 1


class TestEvidenceStrengthClassification:
    """Test evidence strength classification logic."""
    
    def test_strong_evidence_rct_low_pvalue(self, reifier, sample_provenance):
        """
        Test strong evidence classification (p<0.01, RCT).
        
        Requirement 5.1: strong (p<0.01, RCT/meta-analysis)
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.005,
            article_type="original_research"
        )
        
        assert claim.evidence_strength == EvidenceStrength.STRONG
    
    def test_strong_evidence_meta_analysis_low_pvalue(self, reifier, sample_provenance):
        """
        Test strong evidence classification (p<0.01, meta-analysis).
        
        Requirement 5.1: strong (p<0.01, RCT/meta-analysis)
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.001,
            article_type="meta_analysis"
        )
        
        assert claim.evidence_strength == EvidenceStrength.STRONG
    
    def test_strong_evidence_pvalue_zero(self, reifier, sample_provenance):
        """
        Test strong evidence with p=0.0 (including p_value = 0.0).
        
        Requirement 5.1: p<0.01 includes p=0.0
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.0,
            article_type="original_research"
        )
        
        assert claim.evidence_strength == EvidenceStrength.STRONG
    
    def test_moderate_evidence_pvalue_below_005(self, reifier, sample_provenance):
        """
        Test moderate evidence classification (p<0.05).
        
        Requirement 5.2: moderate (p<0.05)
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.03,
            article_type="observational"
        )
        
        assert claim.evidence_strength == EvidenceStrength.MODERATE
    
    def test_moderate_evidence_rct_without_low_pvalue(self, reifier, sample_provenance):
        """
        Test moderate evidence when RCT but p>=0.01.
        
        Requirement 5.2: moderate (p<0.05) even for RCT if p>=0.01
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.02,
            article_type="original_research"
        )
        
        assert claim.evidence_strength == EvidenceStrength.MODERATE
    
    def test_weak_evidence_pvalue_below_01(self, reifier, sample_provenance):
        """
        Test weak evidence classification (p<0.1).
        
        Requirement 5.3: weak (p<0.1 or no p-value)
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.08,
            article_type="case_report"
        )
        
        assert claim.evidence_strength == EvidenceStrength.WEAK
    
    def test_weak_evidence_no_pvalue(self, reifier, sample_provenance):
        """
        Test weak evidence when no p-value provided.
        
        Requirement 5.3: weak (p<0.1 or no p-value)
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=None,
            article_type="original_research"
        )
        
        assert claim.evidence_strength == EvidenceStrength.WEAK
    
    def test_weak_evidence_high_pvalue(self, reifier, sample_provenance):
        """
        Test weak evidence when p>=0.1.
        
        Requirement 5.3: weak (p<0.1 or no p-value)
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.15,
            article_type="original_research"
        )
        
        assert claim.evidence_strength == EvidenceStrength.WEAK
    
    def test_conflicting_evidence_with_contradicting_papers(self, reifier, sample_provenance):
        """
        Test conflicting evidence when contradicting papers exist.
        
        Requirement 5.4: conflicting when both supporting and contradicting evidence
        """
        # Create initial claim
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.001,
            article_type="original_research"
        )
        
        # Initially should be strong
        assert claim.evidence_strength == EvidenceStrength.STRONG
        
        # Add contradicting evidence
        contradicting_evidence = ProvenanceMetadata(
            paper_id="10.1234/contradicting.2024",
            section_type="results",
            source_sentence="B. fragilis decreased in T2D.",
            extraction_method="biobert_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.82,
        )
        
        updated_claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=contradicting_evidence,
            supports=False,
            p_value=0.001,
            article_type="original_research"
        )
        
        # Should now be conflicting
        assert updated_claim.evidence_strength == EvidenceStrength.CONFLICTING
    
    def test_pvalue_validation_below_zero(self, reifier, sample_provenance):
        """
        Test that p-value < 0.0 raises ValueError.
        
        Requirement 5.5: validate p_values in range [0.0, 1.0]
        """
        with pytest.raises(ValueError, match="p_value must be in range"):
            reifier.reify_claim(
                subject="Bacteroides fragilis",
                predicate="associated_with_increased_abundance",
                object_entity="Type 2 Diabetes",
                supporting_evidence=[sample_provenance],
                p_value=-0.01,
                article_type="original_research"
            )
    
    def test_pvalue_validation_above_one(self, reifier, sample_provenance):
        """
        Test that p-value > 1.0 raises ValueError.
        
        Requirement 5.5: validate p_values in range [0.0, 1.0]
        """
        with pytest.raises(ValueError, match="p_value must be in range"):
            reifier.reify_claim(
                subject="Bacteroides fragilis",
                predicate="associated_with_increased_abundance",
                object_entity="Type 2 Diabetes",
                supporting_evidence=[sample_provenance],
                p_value=1.5,
                article_type="original_research"
            )
    
    def test_pvalue_validation_exactly_zero(self, reifier, sample_provenance):
        """
        Test that p-value = 0.0 is valid.
        
        Requirement 5.5: validate p_values in range [0.0, 1.0]
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=0.0,
            article_type="original_research"
        )
        
        assert claim.evidence_strength == EvidenceStrength.STRONG
    
    def test_pvalue_validation_exactly_one(self, reifier, sample_provenance):
        """
        Test that p-value = 1.0 is valid.
        
        Requirement 5.5: validate p_values in range [0.0, 1.0]
        """
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[sample_provenance],
            p_value=1.0,
            article_type="original_research"
        )
        
        assert claim.evidence_strength == EvidenceStrength.WEAK

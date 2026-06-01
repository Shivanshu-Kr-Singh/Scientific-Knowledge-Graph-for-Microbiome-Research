"""
graph/test_multi_paper_aggregation.py
--------------------------------------
Integration test for multi-paper aggregation and reified claim creation.

This test validates the relationship reifier's ability to aggregate evidence from
multiple papers into reified claims, including:
1. Creating a reified claim from 3+ papers reporting the same (subject, predicate, object) triple
2. Correct calculation of consensus_confidence as a weighted average
3. Detection of conflicting evidence when papers report opposite directions

Task: 15.3 Write integration test for multi-paper aggregation
Requirements: 20.3
"""

import pytest
from datetime import datetime, timezone

from graph.relationship_reifier import RelationshipReifier
from graph.provenance import ProvenanceMetadata
from graph.reified_claims import ScientificClaim, EvidenceStrength


# ========== Test Fixtures ==========

@pytest.fixture
def reifier():
    """Create a relationship reifier instance."""
    return RelationshipReifier()


@pytest.fixture
def paper1_evidence():
    """
    Create provenance metadata for paper 1 supporting Bacteroides-T2D association.
    
    Paper 1: High confidence (0.9), p=0.001, RCT study
    """
    return ProvenanceMetadata(
        paper_id="10.1234/paper1.2024",
        section_type="results",
        source_sentence=(
            "Bacteroides fragilis was significantly increased in Type 2 Diabetes patients "
            "compared to healthy controls (p = 0.001, fold change = 2.5)."
        ),
        sentence_offset=0,
        extraction_method="llm_extractor_v1.2",
        extraction_timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        extractor_version="1.2",
        llm_prompt_hash="abc123",
        confidence_score=0.9,
        validation_status="unvalidated",
        validator_id=None,
        surrounding_context=None,
        figure_table_ref=None
    )


@pytest.fixture
def paper2_evidence():
    """
    Create provenance metadata for paper 2 supporting Bacteroides-T2D association.
    
    Paper 2: Medium confidence (0.8), p=0.005, observational study
    """
    return ProvenanceMetadata(
        paper_id="10.1234/paper2.2024",
        section_type="results",
        source_sentence=(
            "We confirmed that Bacteroides fragilis abundance was elevated in T2D patients "
            "(p = 0.005, LDA score = 3.1)."
        ),
        sentence_offset=0,
        extraction_method="biobert_ner",
        extraction_timestamp=datetime(2024, 2, 20, 14, 30, 0, tzinfo=timezone.utc),
        extractor_version="2.0",
        llm_prompt_hash=None,
        confidence_score=0.8,
        validation_status="unvalidated",
        validator_id=None,
        surrounding_context=None,
        figure_table_ref=None
    )


@pytest.fixture
def paper3_evidence():
    """
    Create provenance metadata for paper 3 supporting Bacteroides-T2D association.
    
    Paper 3: Lower confidence (0.75), p=0.01, case-control study
    """
    return ProvenanceMetadata(
        paper_id="10.1234/paper3.2024",
        section_type="results",
        source_sentence=(
            "Bacteroides fragilis showed increased relative abundance in Type 2 Diabetes "
            "cohort (p = 0.01, effect size = 1.8)."
        ),
        sentence_offset=0,
        extraction_method="regex_ner",
        extraction_timestamp=datetime(2024, 3, 10, 9, 15, 0, tzinfo=timezone.utc),
        extractor_version="1.0",
        llm_prompt_hash=None,
        confidence_score=0.75,
        validation_status="unvalidated",
        validator_id=None,
        surrounding_context=None,
        figure_table_ref=None
    )


@pytest.fixture
def paper4_evidence():
    """
    Create provenance metadata for paper 4 supporting Bacteroides-T2D association.
    
    Paper 4: Good confidence (0.85), p=0.003, meta-analysis
    """
    return ProvenanceMetadata(
        paper_id="10.1234/paper4.2024",
        section_type="results",
        source_sentence=(
            "Meta-analysis revealed consistent increase in Bacteroides fragilis across studies "
            "(pooled p = 0.003, pooled effect size = 2.2)."
        ),
        sentence_offset=0,
        extraction_method="llm_extractor_v1.2",
        extraction_timestamp=datetime(2024, 4, 5, 16, 45, 0, tzinfo=timezone.utc),
        extractor_version="1.2",
        llm_prompt_hash="abc123",
        confidence_score=0.85,
        validation_status="unvalidated",
        validator_id=None,
        surrounding_context=None,
        figure_table_ref=None
    )


@pytest.fixture
def paper5_contradicting_evidence():
    """
    Create provenance metadata for paper 5 contradicting Bacteroides-T2D association.
    
    Paper 5: Medium confidence (0.8), p=0.02, reports DECREASED abundance
    """
    return ProvenanceMetadata(
        paper_id="10.1234/paper5.2024",
        section_type="results",
        source_sentence=(
            "Contrary to previous reports, we observed decreased Bacteroides fragilis "
            "in Type 2 Diabetes patients (p = 0.02, fold change = 0.6)."
        ),
        sentence_offset=0,
        extraction_method="biobert_ner",
        extraction_timestamp=datetime(2024, 5, 12, 11, 20, 0, tzinfo=timezone.utc),
        extractor_version="2.0",
        llm_prompt_hash=None,
        confidence_score=0.8,
        validation_status="unvalidated",
        validator_id=None,
        surrounding_context=None,
        figure_table_ref=None
    )


@pytest.fixture
def paper6_contradicting_evidence():
    """
    Create provenance metadata for paper 6 contradicting Bacteroides-T2D association.
    
    Paper 6: Good confidence (0.82), p=0.015, reports DECREASED abundance
    """
    return ProvenanceMetadata(
        paper_id="10.1234/paper6.2024",
        section_type="results",
        source_sentence=(
            "Bacteroides fragilis was significantly reduced in T2D patients compared to controls "
            "(p = 0.015, relative abundance ratio = 0.55)."
        ),
        sentence_offset=0,
        extraction_method="llm_extractor_v1.2",
        extraction_timestamp=datetime(2024, 6, 8, 13, 10, 0, tzinfo=timezone.utc),
        extractor_version="1.2",
        llm_prompt_hash="abc123",
        confidence_score=0.82,
        validation_status="unvalidated",
        validator_id=None,
        surrounding_context=None,
        figure_table_ref=None
    )


# ========== Integration Tests ==========

class TestMultiPaperAggregation:
    """
    Integration tests for multi-paper aggregation and reified claim creation.
    
    Requirements: 20.3
    """
    
    def test_create_reified_claim_from_three_papers(
        self,
        reifier,
        paper1_evidence,
        paper2_evidence,
        paper3_evidence
    ):
        """
        Test creating a reified claim from 3 papers with same (subject, predicate, object).
        
        This test verifies that:
        1. A reified claim can be created from 3+ papers
        2. All supporting papers are included in the claim
        3. The claim has a unique claim_id
        4. Basic claim structure is correct
        
        Requirements: 20.3, 4.1
        """
        # Create reified claim from 3 papers
        supporting_evidence = [paper1_evidence, paper2_evidence, paper3_evidence]
        
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=supporting_evidence,
            claim_type="association",
            p_value=0.001,  # Best p-value from the papers
            article_type="original_research"
        )
        
        # Verify claim was created
        assert isinstance(claim, ScientificClaim), \
            "Should return a ScientificClaim instance"
        
        # Verify claim has unique ID
        assert claim.claim_id is not None, "Claim should have a claim_id"
        assert len(claim.claim_id) > 0, "Claim ID should not be empty"
        
        # Verify claim structure
        assert claim.subject_entity == "Bacteroides fragilis", \
            "Subject should match input"
        assert claim.predicate == "associated_with_increased_abundance", \
            "Predicate should match input"
        assert claim.object_entity == "Type 2 Diabetes", \
            "Object should match input"
        assert claim.claim_type == "association", \
            "Claim type should match input"
        
        # Verify all 3 papers are included
        assert len(claim.supporting_papers) == 3, \
            "Should have 3 supporting papers"
        
        expected_paper_ids = {
            "10.1234/paper1.2024",
            "10.1234/paper2.2024",
            "10.1234/paper3.2024"
        }
        assert set(claim.supporting_papers) == expected_paper_ids, \
            "Supporting papers should match input evidence"
        
        # Verify no contradicting papers initially
        assert len(claim.contradicting_papers) == 0, \
            "Should have no contradicting papers initially"
    
    def test_consensus_confidence_calculation_weighted_average(
        self,
        reifier,
        paper1_evidence,
        paper2_evidence,
        paper3_evidence
    ):
        """
        Test that consensus_confidence is calculated as a weighted average.
        
        This test verifies that:
        1. Consensus confidence is calculated correctly
        2. The calculation uses confidence scores as weights
        3. The result is in the valid range [0.0, 1.0]
        
        Paper confidences:
        - Paper 1: 0.9
        - Paper 2: 0.8
        - Paper 3: 0.75
        
        Expected weighted average:
        weighted_sum = (0.9 * 0.9) + (0.8 * 0.8) + (0.75 * 0.75)
                     = 0.81 + 0.64 + 0.5625 = 2.0125
        total_weight = 0.9 + 0.8 + 0.75 = 2.45
        consensus = 2.0125 / 2.45 ≈ 0.8214
        
        Requirements: 20.3, 4.3
        """
        # Create reified claim
        supporting_evidence = [paper1_evidence, paper2_evidence, paper3_evidence]
        
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=supporting_evidence,
            claim_type="association",
            p_value=0.001,
            article_type="original_research"
        )
        
        # Calculate expected consensus confidence
        # Weighted average: sum(confidence^2) / sum(confidence)
        weighted_sum = (0.9 * 0.9) + (0.8 * 0.8) + (0.75 * 0.75)
        total_weight = 0.9 + 0.8 + 0.75
        expected_consensus = weighted_sum / total_weight
        
        # Verify consensus confidence
        assert claim.consensus_confidence is not None, \
            "Consensus confidence should be calculated"
        
        assert 0.0 <= claim.consensus_confidence <= 1.0, \
            f"Consensus confidence should be in [0.0, 1.0], got {claim.consensus_confidence}"
        
        # Allow small floating point tolerance
        assert abs(claim.consensus_confidence - expected_consensus) < 0.001, \
            f"Consensus confidence should be {expected_consensus:.4f}, got {claim.consensus_confidence:.4f}"
        
        # Verify it's approximately 0.8214
        assert abs(claim.consensus_confidence - 0.8214) < 0.001, \
            f"Consensus confidence should be approximately 0.8214, got {claim.consensus_confidence:.4f}"
    
    def test_consensus_confidence_with_four_papers(
        self,
        reifier,
        paper1_evidence,
        paper2_evidence,
        paper3_evidence,
        paper4_evidence
    ):
        """
        Test consensus confidence calculation with 4 papers.
        
        This test verifies that the weighted average calculation works correctly
        with more papers.
        
        Paper confidences:
        - Paper 1: 0.9
        - Paper 2: 0.8
        - Paper 3: 0.75
        - Paper 4: 0.85
        
        Expected weighted average:
        weighted_sum = (0.9^2) + (0.8^2) + (0.75^2) + (0.85^2)
                     = 0.81 + 0.64 + 0.5625 + 0.7225 = 2.735
        total_weight = 0.9 + 0.8 + 0.75 + 0.85 = 3.3
        consensus = 2.735 / 3.3 ≈ 0.8288
        
        Requirements: 20.3, 4.3
        """
        # Create reified claim with 4 papers
        supporting_evidence = [
            paper1_evidence,
            paper2_evidence,
            paper3_evidence,
            paper4_evidence
        ]
        
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=supporting_evidence,
            claim_type="association",
            p_value=0.001,
            article_type="meta_analysis"
        )
        
        # Calculate expected consensus confidence
        weighted_sum = (0.9 * 0.9) + (0.8 * 0.8) + (0.75 * 0.75) + (0.85 * 0.85)
        total_weight = 0.9 + 0.8 + 0.75 + 0.85
        expected_consensus = weighted_sum / total_weight
        
        # Verify consensus confidence
        assert abs(claim.consensus_confidence - expected_consensus) < 0.001, \
            f"Consensus confidence should be {expected_consensus:.4f}, got {claim.consensus_confidence:.4f}"
        
        # Verify it's approximately 0.8288
        assert abs(claim.consensus_confidence - 0.8288) < 0.001, \
            f"Consensus confidence should be approximately 0.8288, got {claim.consensus_confidence:.4f}"
        
        # Verify all 4 papers are included
        assert len(claim.supporting_papers) == 4, \
            "Should have 4 supporting papers"
    
    def test_conflicting_evidence_detection_opposite_directions(
        self,
        reifier,
        paper1_evidence,
        paper2_evidence,
        paper3_evidence,
        paper5_contradicting_evidence,
        paper6_contradicting_evidence
    ):
        """
        Test detection of conflicting evidence when papers have opposite directions.
        
        This test verifies that:
        1. A claim can be created with supporting evidence
        2. Contradicting evidence can be added to the claim
        3. Evidence strength is set to "conflicting" when contradicting evidence exists
        4. Both supporting and contradicting papers are tracked separately
        
        Scenario:
        - 3 papers report INCREASED Bacteroides in T2D
        - 2 papers report DECREASED Bacteroides in T2D
        - This should be detected as conflicting evidence
        
        Requirements: 20.3, 4.6, 5.4
        """
        # Create initial claim with 3 supporting papers
        supporting_evidence = [paper1_evidence, paper2_evidence, paper3_evidence]
        
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=supporting_evidence,
            claim_type="association",
            p_value=0.001,
            article_type="original_research"
        )
        
        # Verify initial state (no contradicting evidence)
        assert len(claim.supporting_papers) == 3, \
            "Should have 3 supporting papers initially"
        assert len(claim.contradicting_papers) == 0, \
            "Should have no contradicting papers initially"
        assert claim.evidence_strength != EvidenceStrength.CONFLICTING, \
            "Evidence strength should not be conflicting initially"
        
        # Add first contradicting evidence
        claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=paper5_contradicting_evidence,
            supports=False,  # This paper contradicts the claim
            p_value=0.02,
            article_type="original_research"
        )
        
        # Verify contradicting paper was added
        assert len(claim.supporting_papers) == 3, \
            "Should still have 3 supporting papers"
        assert len(claim.contradicting_papers) == 1, \
            "Should have 1 contradicting paper"
        assert "10.1234/paper5.2024" in claim.contradicting_papers, \
            "Paper 5 should be in contradicting papers"
        
        # Verify evidence strength is now conflicting (Requirement 5.4)
        assert claim.evidence_strength == EvidenceStrength.CONFLICTING, \
            "Evidence strength should be 'conflicting' when contradicting evidence exists"
        
        # Add second contradicting evidence
        claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=paper6_contradicting_evidence,
            supports=False,
            p_value=0.015,
            article_type="original_research"
        )
        
        # Verify both contradicting papers are tracked
        assert len(claim.supporting_papers) == 3, \
            "Should still have 3 supporting papers"
        assert len(claim.contradicting_papers) == 2, \
            "Should have 2 contradicting papers"
        
        expected_contradicting = {"10.1234/paper5.2024", "10.1234/paper6.2024"}
        assert set(claim.contradicting_papers) == expected_contradicting, \
            "Both contradicting papers should be tracked"
        
        # Verify evidence strength remains conflicting
        assert claim.evidence_strength == EvidenceStrength.CONFLICTING, \
            "Evidence strength should remain 'conflicting'"
        
        # Verify no overlap between supporting and contradicting papers (Requirement 4.2)
        supporting_set = set(claim.supporting_papers)
        contradicting_set = set(claim.contradicting_papers)
        overlap = supporting_set & contradicting_set
        assert len(overlap) == 0, \
            "There should be no overlap between supporting and contradicting papers"
    
    def test_effect_direction_consistency_with_conflicting_evidence(
        self,
        reifier,
        paper1_evidence,
        paper2_evidence,
        paper3_evidence,
        paper5_contradicting_evidence
    ):
        """
        Test that effect_direction_consistency is calculated correctly with conflicts.
        
        This test verifies that:
        1. Initial effect_direction_consistency is 1.0 (all papers agree)
        2. After adding contradicting evidence, consistency decreases
        3. Consistency reflects the ratio of supporting papers
        
        Scenario:
        - 3 supporting papers (increased)
        - 1 contradicting paper (decreased)
        - Expected consistency: 3/4 = 0.75
        
        Requirements: 20.3, 4.4
        """
        # Create initial claim
        supporting_evidence = [paper1_evidence, paper2_evidence, paper3_evidence]
        
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=supporting_evidence,
            claim_type="association",
            p_value=0.001,
            article_type="original_research"
        )
        
        # Verify initial effect_direction_consistency is 1.0
        assert claim.effect_direction_consistency == 1.0, \
            "Initial effect_direction_consistency should be 1.0 (all papers agree)"
        
        # Add contradicting evidence
        claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=paper5_contradicting_evidence,
            supports=False,
            p_value=0.02,
            article_type="original_research"
        )
        
        # Verify effect_direction_consistency is updated
        # 3 supporting / (3 supporting + 1 contradicting) = 3/4 = 0.75
        expected_consistency = 3.0 / 4.0
        
        assert abs(claim.effect_direction_consistency - expected_consistency) < 0.001, \
            f"Effect direction consistency should be {expected_consistency:.2f}, " \
            f"got {claim.effect_direction_consistency:.2f}"
    
    def test_temporal_tracking_across_multiple_papers(
        self,
        reifier,
        paper1_evidence,
        paper2_evidence,
        paper3_evidence,
        paper4_evidence
    ):
        """
        Test that temporal tracking works correctly across multiple papers.
        
        This test verifies that:
        1. first_reported is set to the earliest paper timestamp
        2. last_updated is set to the latest paper timestamp
        3. first_reported <= last_updated (Requirement 4.5)
        
        Paper timestamps:
        - Paper 1: 2024-01-15
        - Paper 2: 2024-02-20
        - Paper 3: 2024-03-10
        - Paper 4: 2024-04-05
        
        Expected:
        - first_reported: 2024-01-15
        - last_updated: 2024-04-05
        
        Requirements: 20.3, 4.5
        """
        # Create claim with papers in non-chronological order
        supporting_evidence = [
            paper3_evidence,  # 2024-03-10
            paper1_evidence,  # 2024-01-15 (earliest)
            paper4_evidence,  # 2024-04-05 (latest)
            paper2_evidence   # 2024-02-20
        ]
        
        claim = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=supporting_evidence,
            claim_type="association",
            p_value=0.001,
            article_type="original_research"
        )
        
        # Parse timestamps
        first_reported = datetime.fromisoformat(claim.first_reported)
        last_updated = datetime.fromisoformat(claim.last_updated)
        
        # Verify first_reported is the earliest timestamp
        expected_first = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert first_reported == expected_first, \
            f"first_reported should be {expected_first}, got {first_reported}"
        
        # Verify last_updated is the latest timestamp
        expected_last = datetime(2024, 4, 5, 16, 45, 0, tzinfo=timezone.utc)
        assert last_updated == expected_last, \
            f"last_updated should be {expected_last}, got {last_updated}"
        
        # Verify temporal ordering (Requirement 4.5)
        assert first_reported <= last_updated, \
            "first_reported should be <= last_updated"
    
    def test_detect_conflicting_claims_opposite_predicates(
        self,
        reifier,
        paper1_evidence,
        paper2_evidence,
        paper5_contradicting_evidence,
        paper6_contradicting_evidence
    ):
        """
        Test detection of conflicting claims with opposite predicates.
        
        This test verifies that:
        1. Two claims with same subject/object but opposite predicates are detected as conflicts
        2. The detect_conflicting_claims method returns the conflicting pairs
        
        Scenario:
        - Claim 1: Bacteroides INCREASED in T2D (2 papers)
        - Claim 2: Bacteroides DECREASED in T2D (2 papers)
        - These should be detected as conflicting
        
        Requirements: 20.3, 4.6, 9.1
        """
        # Create first claim (increased)
        claim1 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[paper1_evidence, paper2_evidence],
            claim_type="association",
            p_value=0.001,
            article_type="original_research"
        )
        
        # Create second claim (decreased)
        claim2 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_decreased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[paper5_contradicting_evidence, paper6_contradicting_evidence],
            claim_type="association",
            p_value=0.02,
            article_type="original_research"
        )
        
        # Detect conflicting claims
        conflicts = reifier.detect_conflicting_claims([claim1, claim2])
        
        # Verify conflicts were detected
        assert len(conflicts) == 1, \
            "Should detect 1 conflicting pair"
        
        conflict_pair = conflicts[0]
        assert len(conflict_pair) == 2, \
            "Conflict pair should contain 2 claims"
        
        # Verify the conflicting claims are claim1 and claim2
        assert claim1 in conflict_pair, \
            "Claim 1 should be in the conflict pair"
        assert claim2 in conflict_pair, \
            "Claim 2 should be in the conflict pair"
        
        # Verify both claims have same subject and object
        assert claim1.subject_entity == claim2.subject_entity, \
            "Conflicting claims should have same subject"
        assert claim1.object_entity == claim2.object_entity, \
            "Conflicting claims should have same object"
        
        # Verify predicates are opposite
        assert "increased" in claim1.predicate.lower(), \
            "Claim 1 should have 'increased' in predicate"
        assert "decreased" in claim2.predicate.lower(), \
            "Claim 2 should have 'decreased' in predicate"
    
    def test_no_conflicts_detected_for_same_direction(
        self,
        reifier,
        paper1_evidence,
        paper2_evidence,
        paper3_evidence,
        paper4_evidence
    ):
        """
        Test that no conflicts are detected when all claims have the same direction.
        
        This test verifies that:
        1. Claims with the same predicate are not detected as conflicts
        2. Empty list is returned when no conflicts exist
        
        Requirements: 20.3, 4.6
        """
        # Create two claims with same predicate (both increased)
        claim1 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[paper1_evidence, paper2_evidence],
            claim_type="association",
            p_value=0.001,
            article_type="original_research"
        )
        
        claim2 = reifier.reify_claim(
            subject="Bacteroides fragilis",
            predicate="associated_with_increased_abundance",
            object_entity="Type 2 Diabetes",
            supporting_evidence=[paper3_evidence, paper4_evidence],
            claim_type="association",
            p_value=0.003,
            article_type="meta_analysis"
        )
        
        # Detect conflicting claims
        conflicts = reifier.detect_conflicting_claims([claim1, claim2])
        
        # Verify no conflicts detected
        assert len(conflicts) == 0, \
            "Should not detect conflicts when claims have same direction"


# ========== Run Tests ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
graph/relationship_reifier.py
------------------------------
Relationship reification for scientific claims.

This module converts complex scientific claims into first-class graph entities
to support multi-paper evidence aggregation, consensus tracking, and conflict detection.

Requirements: 4.1, 4.3, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 5.5, 9.1
"""

from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import uuid

from graph.reified_claims import ScientificClaim, EvidenceStrength
from graph.provenance import ProvenanceMetadata


class RelationshipReifier:
    """
    Reifies relationships into first-class claim entities.
    
    Converts relationships extracted from multiple papers into aggregated
    scientific claims with consensus metrics, evidence strength classification,
    and conflict detection.
    
    Requirements: 4.1, 4.3, 4.5, 4.6, 9.1
    """
    
    def _classify_evidence_strength(
        self,
        p_value: Optional[float],
        article_type: Optional[str],
        has_contradicting_evidence: bool = False
    ) -> EvidenceStrength:
        """
        Classify evidence strength based on p-value, study design, and conflicts.
        
        Requirements:
        - 5.1: strong (p<0.01, RCT/meta-analysis)
        - 5.2: moderate (p<0.05)
        - 5.3: weak (p<0.1 or no p-value)
        - 5.4: conflicting (both supporting and contradicting evidence)
        - 5.5: validate p_values in range [0.0, 1.0]
        
        Args:
            p_value: Statistical p-value (None if not available)
            article_type: Type of study (e.g., "original_research", "meta_analysis")
            has_contradicting_evidence: Whether claim has contradicting evidence
        
        Returns:
            EvidenceStrength classification
        
        Raises:
            ValueError: If p_value is not in range [0.0, 1.0]
        """
        # Requirement 5.4: Conflicting evidence takes precedence
        if has_contradicting_evidence:
            return EvidenceStrength.CONFLICTING
        
        # Requirement 5.5: Validate p_value range
        if p_value is not None:
            if not (0.0 <= p_value <= 1.0):
                raise ValueError(
                    f"p_value must be in range [0.0, 1.0], got {p_value}"
                )
        
        # Requirement 5.1: Strong evidence (p<0.01 AND RCT/meta-analysis)
        if p_value is not None and p_value < 0.01:
            if article_type in ["original_research", "meta_analysis"]:
                return EvidenceStrength.STRONG
        
        # Requirement 5.2: Moderate evidence (p<0.05)
        if p_value is not None and p_value < 0.05:
            return EvidenceStrength.MODERATE
        
        # Requirement 5.3: Weak evidence (p<0.1 or no p-value)
        return EvidenceStrength.WEAK
    
    def reify_claim(
        self,
        subject: str,
        predicate: str,
        object_entity: str,
        supporting_evidence: List[ProvenanceMetadata],
        claim_type: str = "association",
        p_value: Optional[float] = None,
        article_type: Optional[str] = None,
        publication_date: Optional[str] = None,  # ISO date string from paper metadata
    ) -> ScientificClaim:
        """
        Create a reified claim from multiple pieces of evidence.
        
        Preconditions:
        - subject, predicate, object_entity are non-empty strings
        - supporting_evidence contains at least one ProvenanceMetadata
        - All evidence items have confidence_score >= 0.5
        - p_value (if provided) is in range [0.0, 1.0]
        
        Postconditions:
        - Returns ScientificClaim with unique claim_id
        - supporting_papers list matches evidence sources
        - consensus_confidence reflects agreement across evidence
        - evidence_strength determined by p_value and study design
        
        Requirements:
        - 4.1: Create reified claim node aggregating supporting evidence
        - 4.3: Calculate consensus_confidence as weighted average
        - 5.1, 5.2, 5.3, 5.4, 5.5: Evidence strength classification
        
        Args:
            subject: Subject entity (e.g., "Bacteroides fragilis")
            predicate: Predicate (e.g., "associated_with_increased_abundance")
            object_entity: Object entity (e.g., "Type 2 Diabetes")
            supporting_evidence: List of provenance metadata from supporting papers
            claim_type: Type of claim (default: "association")
            p_value: Statistical p-value for the claim (optional)
            article_type: Type of study (e.g., "original_research", "meta_analysis")
        
        Returns:
            ScientificClaim with aggregated evidence
        
        Raises:
            ValueError: If preconditions are not met
        """
        # Validate preconditions
        if not subject or not subject.strip():
            raise ValueError("subject must be a non-empty string")
        if not predicate or not predicate.strip():
            raise ValueError("predicate must be a non-empty string")
        if not object_entity or not object_entity.strip():
            raise ValueError("object_entity must be a non-empty string")
        if not supporting_evidence:
            raise ValueError("supporting_evidence must contain at least one ProvenanceMetadata")
        
        # Validate all evidence has confidence >= 0.5
        for evidence in supporting_evidence:
            if evidence.confidence_score < 0.5:
                raise ValueError(
                    f"All evidence items must have confidence_score >= 0.5, "
                    f"found {evidence.confidence_score}"
                )
        
        # Generate unique claim_id using UUID (Requirement 4.1)
        claim_id = str(uuid.uuid4())
        
        # Extract unique paper IDs from supporting evidence
        supporting_papers = list(set(evidence.paper_id for evidence in supporting_evidence))
        
        # consensus_confidence = arithmetic mean of confidence scores across supporting evidence
        # Scientifically: average extraction quality across all supporting papers
        n = len(supporting_evidence)
        if n == 1:
            consensus_confidence = supporting_evidence[0].confidence_score
        else:
            consensus_confidence = sum(e.confidence_score for e in supporting_evidence) / n
        consensus_confidence = max(0.0, min(1.0, consensus_confidence))
        
        # For initial claim creation, all evidence supports the claim
        # so effect_direction_consistency is 1.0
        effect_direction_consistency = 1.0
        
        # Classify evidence strength (Requirements 5.1-5.5)
        evidence_strength = self._classify_evidence_strength(
            p_value=p_value,
            article_type=article_type,
            has_contradicting_evidence=False  # No contradicting evidence at creation
        )
        
        # Use extraction timestamps as proxy for temporal bounds
        # In a full implementation, this would use paper publication dates
        # For now, use current UTC time as the claim creation timestamp
        now_iso = datetime.now(timezone.utc).isoformat()
        # Use publication date if provided, otherwise use current UTC timestamp
        first_reported = publication_date if publication_date else now_iso
        last_updated = now_iso
        
        # Create the reified claim
        claim = ScientificClaim(
            claim_id=claim_id,
            claim_type=claim_type,
            subject_entity=subject,
            predicate=predicate,
            object_entity=object_entity,
            supporting_papers=supporting_papers,
            contradicting_papers=[],
            total_sample_size=0,  # Will be calculated from paper metadata
            evidence_strength=evidence_strength,
            consensus_confidence=consensus_confidence,
            effect_direction_consistency=effect_direction_consistency,
            first_reported=first_reported,
            last_updated=last_updated,
            pooled_effect_size=None,
            effect_size_variance=None,
            meta_analysis_performed=False,
        )
        
        return claim
    
    def update_claim_with_new_evidence(
        self,
        claim: ScientificClaim,
        new_evidence: ProvenanceMetadata,
        supports: bool,
        p_value: Optional[float] = None,
        article_type: Optional[str] = None
    ) -> ScientificClaim:
        """
        Update an existing claim with new supporting or contradicting evidence.
        
        Preconditions:
        - claim is a valid ScientificClaim
        - new_evidence is valid ProvenanceMetadata
        - supports is True if evidence supports claim, False if contradicts
        - p_value (if provided) is in range [0.0, 1.0]
        
        Postconditions:
        - Returns updated ScientificClaim
        - Paper ID added to supporting_papers or contradicting_papers
        - consensus_confidence recalculated
        - last_updated timestamp updated to current time
        - evidence_strength set to "conflicting" if contradicting evidence exists
        
        Requirements:
        - 4.5: Update last_updated timestamp
        - 4.6: Update evidence_strength if contradicting evidence changes classification
        - 5.4: Set evidence_strength to "conflicting" when both supporting and contradicting evidence exists
        
        Args:
            claim: Existing scientific claim
            new_evidence: New provenance metadata
            supports: True if evidence supports claim, False if contradicts
            p_value: Statistical p-value for the new evidence (optional)
            article_type: Type of study (e.g., "original_research", "meta_analysis")
        
        Returns:
            Updated ScientificClaim
        
        Raises:
            ValueError: If preconditions are not met
        """
        # Validate preconditions
        if new_evidence.confidence_score < 0.5:
            raise ValueError(
                f"new_evidence must have confidence_score >= 0.5, "
                f"found {new_evidence.confidence_score}"
            )
        
        # Check if paper is already in the claim
        paper_id = new_evidence.paper_id
        if paper_id in claim.supporting_papers or paper_id in claim.contradicting_papers:
            # Paper already included, return claim unchanged
            return claim
        
        # Create updated lists
        supporting_papers = claim.supporting_papers.copy()
        contradicting_papers = claim.contradicting_papers.copy()
        
        if supports:
            supporting_papers.append(paper_id)
        else:
            contradicting_papers.append(paper_id)
        
        # Recalculate consensus_confidence (Requirement 4.5)
        # For simplicity, we maintain the existing consensus and adjust slightly
        # In a full implementation, we would track all evidence and recalculate
        total_papers = len(supporting_papers) + len(contradicting_papers)
        support_ratio = len(supporting_papers) / total_papers if total_papers > 0 else 0.0
        
        # Adjust consensus confidence based on new evidence
        if supports:
            # New supporting evidence increases confidence slightly
            consensus_confidence = (
                claim.consensus_confidence * 0.9 + new_evidence.confidence_score * 0.1
            )
        else:
            # New contradicting evidence decreases confidence
            consensus_confidence = claim.consensus_confidence * 0.8
        
        # Ensure consensus_confidence stays in valid range
        consensus_confidence = max(0.0, min(1.0, consensus_confidence))
        
        # Recalculate effect_direction_consistency
        effect_direction_consistency = support_ratio
        
        # Update evidence_strength (Requirements 4.6, 5.4)
        # Requirement 5.4: Set to "conflicting" when both supporting and contradicting evidence exists
        has_contradicting_evidence = len(contradicting_papers) > 0
        evidence_strength = self._classify_evidence_strength(
            p_value=p_value,
            article_type=article_type,
            has_contradicting_evidence=has_contradicting_evidence
        )
        
        # Update last_updated timestamp (Requirement 4.5)
        last_updated = datetime.now(timezone.utc).isoformat()
        
        # Create updated claim
        updated_claim = ScientificClaim(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            subject_entity=claim.subject_entity,
            predicate=claim.predicate,
            object_entity=claim.object_entity,
            supporting_papers=supporting_papers,
            contradicting_papers=contradicting_papers,
            total_sample_size=claim.total_sample_size,
            evidence_strength=evidence_strength,
            consensus_confidence=consensus_confidence,
            effect_direction_consistency=effect_direction_consistency,
            first_reported=claim.first_reported,
            last_updated=last_updated,
            pooled_effect_size=claim.pooled_effect_size,
            effect_size_variance=claim.effect_size_variance,
            meta_analysis_performed=claim.meta_analysis_performed,
        )
        
        return updated_claim
    
    def detect_conflicting_claims(
        self,
        claims: List[ScientificClaim]
    ) -> List[Tuple[ScientificClaim, ScientificClaim]]:
        """
        Identify pairs of claims that contradict each other.
        
        Preconditions:
        - claims is a non-empty list of ScientificClaim objects
        
        Postconditions:
        - Returns list of claim pairs with opposite predicates
        - Only returns pairs with same subject and object
        - Claims with "associated" in the predicate are never flagged as conflicting
        - Empty list if no conflicts found
        
        Requirements:
        - 4.6: Detect conflicting claims
        - 9.1: Support conflicting evidence detection
        
        Args:
            claims: List of scientific claims to analyze
        
        Returns:
            List of tuples containing conflicting claim pairs
        """
        if not claims:
            return []
        
        conflicts: List[Tuple[ScientificClaim, ScientificClaim]] = []
        
        # Compare each pair of claims
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                claim1 = claims[i]
                claim2 = claims[j]
                
                # Check if claims have same subject and object
                if (claim1.subject_entity == claim2.subject_entity and
                    claim1.object_entity == claim2.object_entity):
                    
                    # Check if predicates are opposite
                    # Common opposite patterns:
                    # - "increased" vs "decreased"
                    # - "associated_with_increased" vs "associated_with_decreased"
                    # - "positive_effect" vs "negative_effect"
                    
                    pred1 = claim1.predicate.lower()
                    pred2 = claim2.predicate.lower()
                    
                    # "associated" is non-committal — it never conflicts with directional claims
                    if "associated" in pred1 or "associated" in pred2:
                        continue
                    
                    is_opposite = False
                    
                    # Check for increased/decreased opposition
                    if ("increased" in pred1 and "decreased" in pred2) or \
                       ("decreased" in pred1 and "increased" in pred2):
                        is_opposite = True
                    
                    # Check for positive/negative opposition
                    if ("positive" in pred1 and "negative" in pred2) or \
                       ("negative" in pred1 and "positive" in pred2):
                        is_opposite = True
                    
                    # Check for up/down opposition
                    if ("up" in pred1 and "down" in pred2) or \
                       ("down" in pred1 and "up" in pred2):
                        is_opposite = True
                    
                    if is_opposite:
                        conflicts.append((claim1, claim2))
        
        return conflicts

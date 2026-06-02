"""
Reified Claim Data Models

This module defines Pydantic models for representing scientific claims as first-class
graph entities that aggregate evidence from multiple papers.

Requirements Addressed:
- 4.1: Reified claim node aggregating supporting evidence
- 4.2: Separate lists of supporting and contradicting paper IDs
- 4.3: Consensus confidence calculation
- 4.4: Effect direction consistency calculation
- 4.5: Temporal evolution tracking
- 4.6: Conflicting evidence detection
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime
from loguru import logger


class EvidenceStrength(str, Enum):
    """
    Classification of relationship quality based on study design and statistical significance.
    
    Values:
    - STRONG: RCT, meta-analysis, p < 0.01
    - MODERATE: Observational with controls, p < 0.05
    - WEAK: Case reports, p < 0.1
    - CONFLICTING: Multiple papers with opposite findings
    """
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    CONFLICTING = "conflicting"


class ScientificClaim(BaseModel):
    """
    A reified scientific claim that can be supported by multiple papers.
    
    This model represents a scientific claim as a first-class entity that aggregates
    evidence from multiple papers, tracks consensus, and identifies conflicts.
    
    Requirements:
    - 4.1: Aggregates supporting evidence from multiple papers
    - 4.2: Maintains separate supporting and contradicting paper lists
    - 4.3: Calculates consensus confidence
    - 4.4: Calculates effect direction consistency
    - 4.5: Tracks temporal evolution
    - 4.6: Handles conflicting evidence
    """
    
    # Claim identification
    claim_id: str = Field(..., description="UUID for this claim")
    claim_type: str = Field(
        ...,
        description="Type of claim: association | intervention_effect | methodology_comparison"
    )
    
    # Claim content (subject, predicate, object triple)
    subject_entity: str = Field(..., description="Subject entity (e.g., 'Bacteroides fragilis')")
    predicate: str = Field(..., description="Predicate (e.g., 'associated_with_increased_abundance')")
    object_entity: str = Field(..., description="Object entity (e.g., 'Type 2 Diabetes')")
    
    # Evidence aggregation (Requirement 4.2)
    supporting_papers: List[str] = Field(
        default_factory=list,
        description="List of paper IDs supporting this claim"
    )
    contradicting_papers: List[str] = Field(
        default_factory=list,
        description="List of paper IDs contradicting this claim"
    )
    total_sample_size: int = Field(
        default=0,
        ge=0,
        description="Sum of sample sizes across all supporting papers"
    )
    
    # Consensus metrics (Requirements 4.3, 4.4)
    evidence_strength: EvidenceStrength = Field(
        ...,
        description="Evidence quality classification"
    )
    consensus_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted average confidence based on agreement across papers"
    )
    effect_direction_consistency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of papers agreeing on the dominant direction"
    )
    
    # Temporal tracking (Requirement 4.5)
    first_reported: str = Field(
        ...,
        description="ISO date of earliest supporting paper"
    )
    last_updated: str = Field(
        ...,
        description="ISO date of most recent supporting paper"
    )
    
    # Statistical aggregation (optional)
    pooled_effect_size: Optional[float] = Field(
        default=None,
        description="Pooled effect size across studies"
    )
    effect_size_variance: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Variance of effect sizes"
    )
    meta_analysis_performed: bool = Field(
        default=False,
        description="Whether meta-analysis has been performed"
    )
    
    @field_validator('consensus_confidence', 'effect_direction_consistency')
    @classmethod
    def validate_consensus_metrics(cls, v: float) -> float:
        """
        Validate that consensus metrics are in the range [0.0, 1.0].
        
        Requirements: 4.3, 4.4
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Consensus metric must be in range [0.0, 1.0], got {v}")
        return v
    
    @field_validator('supporting_papers', 'contradicting_papers')
    @classmethod
    def validate_paper_lists(cls, v: List[str]) -> List[str]:
        """
        Validate that paper ID lists contain unique values.
        
        Requirement: 4.2
        """
        if len(v) != len(set(v)):
            raise ValueError("Paper ID lists must not contain duplicates")
        return v
    
    @field_validator('claim_type')
    @classmethod
    def validate_claim_type(cls, v: str) -> str:
        """Validate claim_type is one of the allowed values."""
        allowed_types = {
            "association", "intervention_effect", "methodology_comparison",
            "open_world",       # LLM-extracted open-world triples
            "mechanistic",      # Causal/mechanistic relationships
            "biomarker",        # Biomarker associations
            "unknown",          # Fallback for unclassified claims
        }
        if v not in allowed_types:
            # Instead of raising, normalize to "unknown" for forward compatibility
            logger.warning(f"Unknown claim_type '{v}', normalizing to 'unknown'")
            return "unknown"
        return v
    
    def model_post_init(self, __context) -> None:
        """
        Post-initialization validation.
        
        Requirements:
        - 4.2: Ensure no overlap between supporting and contradicting papers
        - 4.5: Validate temporal ordering
        """
        # Requirement 4.2: No overlap between supporting and contradicting papers
        supporting_set = set(self.supporting_papers)
        contradicting_set = set(self.contradicting_papers)
        overlap = supporting_set & contradicting_set
        if overlap:
            raise ValueError(
                f"Paper IDs cannot appear in both supporting and contradicting lists: {overlap}"
            )
        
        # Requirement 4.5: Validate temporal ordering (first_reported <= last_updated)
        try:
            first = datetime.fromisoformat(self.first_reported)
            last = datetime.fromisoformat(self.last_updated)
            if first > last:
                raise ValueError(
                    f"first_reported ({self.first_reported}) must be <= last_updated ({self.last_updated})"
                )
        except ValueError as e:
            if "first_reported" in str(e) or "last_updated" in str(e):
                raise
            # If it's an ISO format error, re-raise with more context
            raise ValueError(
                f"Invalid ISO date format for first_reported or last_updated: {e}"
            )


class ReifiedClaimNode(BaseModel):
    """
    A scientific claim as a first-class graph node.
    
    This model extends ScientificClaim with additional graph-specific metadata
    for Neo4j storage and querying.
    
    Requirements:
    - 4.1: First-class graph node for claims
    - 4.2: Evidence aggregation with separate lists
    - 4.3: Consensus confidence tracking
    - 4.4: Effect direction consistency
    - 4.5: Temporal evolution
    - 4.6: Conflicting evidence handling
    """
    
    # Node identification
    node_id: str = Field(..., description="UUID for this claim node")
    node_type: str = Field(
        default="ScientificClaim",
        description="Node type label for Neo4j"
    )
    
    # Claim structure
    claim_type: str = Field(
        ...,
        description="Type of claim: association | intervention_effect | methodology_comparison"
    )
    subject_entity: str = Field(..., description="Entity ID (e.g., taxon ID)")
    predicate: str = Field(..., description="Normalized predicate (e.g., 'increases_in')")
    object_entity: str = Field(..., description="Entity ID (e.g., disease ID)")
    
    # Evidence aggregation (Requirement 4.2)
    supporting_paper_ids: List[str] = Field(
        default_factory=list,
        description="List of paper IDs supporting this claim"
    )
    contradicting_paper_ids: List[str] = Field(
        default_factory=list,
        description="List of paper IDs contradicting this claim"
    )
    total_sample_size: int = Field(
        default=0,
        ge=0,
        description="Sum of sample sizes across all supporting papers"
    )
    
    # Consensus metrics (Requirements 4.3, 4.4)
    evidence_strength: str = Field(
        ...,
        description="Evidence quality: strong | moderate | weak | conflicting"
    )
    consensus_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted average confidence based on agreement across papers"
    )
    effect_direction_consistency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of papers agreeing on the dominant direction"
    )
    
    # Statistical aggregation (optional)
    pooled_effect_size: Optional[float] = Field(
        default=None,
        description="Pooled effect size across studies"
    )
    effect_size_variance: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Variance of effect sizes"
    )
    pooled_p_value: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pooled p-value from meta-analysis"
    )
    heterogeneity_i2: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="I² statistic for meta-analysis heterogeneity"
    )
    
    # Temporal tracking (Requirement 4.5)
    first_reported: datetime = Field(
        ...,
        description="Timestamp of earliest supporting paper"
    )
    last_updated: datetime = Field(
        ...,
        description="Timestamp of most recent supporting paper"
    )
    
    # Metadata
    meta_analysis_performed: bool = Field(
        default=False,
        description="Whether meta-analysis has been performed"
    )
    meta_analysis_method: Optional[str] = Field(
        default=None,
        description="Meta-analysis method: random_effects | fixed_effects"
    )
    created_by: str = Field(
        default="system",
        description="Creator: system | user_id"
    )
    
    @field_validator('consensus_confidence', 'effect_direction_consistency')
    @classmethod
    def validate_consensus_metrics(cls, v: float) -> float:
        """
        Validate that consensus metrics are in the range [0.0, 1.0].
        
        Requirements: 4.3, 4.4
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Consensus metric must be in range [0.0, 1.0], got {v}")
        return v
    
    @field_validator('supporting_paper_ids', 'contradicting_paper_ids')
    @classmethod
    def validate_paper_lists(cls, v: List[str]) -> List[str]:
        """
        Validate that paper ID lists contain unique values.
        
        Requirement: 4.2
        """
        if len(v) != len(set(v)):
            raise ValueError("Paper ID lists must not contain duplicates")
        return v
    
    @field_validator('evidence_strength')
    @classmethod
    def validate_evidence_strength(cls, v: str) -> str:
        """Validate evidence_strength is one of the allowed values."""
        allowed_strengths = {"strong", "moderate", "weak", "conflicting"}
        if v not in allowed_strengths:
            raise ValueError(
                f"evidence_strength must be one of {allowed_strengths}, got '{v}'"
            )
        return v
    
    @field_validator('claim_type')
    @classmethod
    def validate_claim_type(cls, v: str) -> str:
        """Validate claim_type is one of the allowed values."""
        allowed_types = {
            "association", "intervention_effect", "methodology_comparison",
            "open_world",       # LLM-extracted open-world triples
            "mechanistic",      # Causal/mechanistic relationships
            "biomarker",        # Biomarker associations
            "unknown",          # Fallback for unclassified claims
        }
        if v not in allowed_types:
            # Instead of raising, normalize to "unknown" for forward compatibility
            logger.warning(f"Unknown claim_type '{v}', normalizing to 'unknown'")
            return "unknown"
        return v
    
    @field_validator('meta_analysis_method')
    @classmethod
    def validate_meta_analysis_method(cls, v: Optional[str]) -> Optional[str]:
        """Validate meta_analysis_method if provided."""
        if v is not None:
            allowed_methods = {"random_effects", "fixed_effects"}
            if v not in allowed_methods:
                raise ValueError(
                    f"meta_analysis_method must be one of {allowed_methods}, got '{v}'"
                )
        return v
    
    def model_post_init(self, __context) -> None:
        """
        Post-initialization validation.
        
        Requirements:
        - 4.2: Ensure no overlap between supporting and contradicting papers
        - 4.5: Validate temporal ordering
        """
        # Requirement 4.2: No overlap between supporting and contradicting papers
        supporting_set = set(self.supporting_paper_ids)
        contradicting_set = set(self.contradicting_paper_ids)
        overlap = supporting_set & contradicting_set
        if overlap:
            raise ValueError(
                f"Paper IDs cannot appear in both supporting and contradicting lists: {overlap}"
            )
        
        # Requirement 4.5: Validate temporal ordering (first_reported <= last_updated)
        if self.first_reported > self.last_updated:
            raise ValueError(
                f"first_reported ({self.first_reported}) must be <= last_updated ({self.last_updated})"
            )

"""
graph/triple_promotion_models.py
---------------------------------
Data models for the open-world triple promotion pipeline.

Defines Pydantic models for promoted triples, open-world claims,
evidence items, and paper metadata used throughout the promotion
and claim aggregation workflow.

Requirements: 1.1, 2.3, 2.4, 3.1, 3.3, 6.4
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from graph.provenance import ProvenanceMetadata


class EvidenceItem(BaseModel):
    """
    A single piece of evidence supporting an Open_World_Claim.

    Each item records the provenance and strength of one triple
    extracted from a specific paper and section.

    Requirements: 3.1, 6.4
    """

    paper_id: str = Field(..., min_length=1, description="Identifier of the source paper")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score")
    evidence_strength: str = Field(..., description="'strong' | 'moderate' | 'weak'")
    section_type: str = Field(..., description="Section from which triple was extracted")
    source_sentence: str = Field(..., min_length=1, description="Evidence sentence supporting the claim")
    extraction_timestamp: str = Field(..., description="ISO 8601 timestamp of extraction")

    @field_validator("evidence_strength")
    @classmethod
    def validate_evidence_strength(cls, v: str) -> str:
        allowed = {"strong", "moderate", "weak"}
        if v not in allowed:
            raise ValueError(f"evidence_strength must be one of {allowed}, got '{v}'")
        return v


class PaperMetadata(BaseModel):
    """
    Minimal paper-level metadata needed for promotion decisions.

    Used by the TriplePromoter to determine evidence strength and
    track provenance at the paper level.

    Requirements: 6.1, 6.2, 6.3
    """

    paper_id: str = Field(..., min_length=1, description="Identifier of the paper (DOI, PMID, or title)")
    article_type: str = Field(..., description="'original_research', 'meta_analysis', 'review', etc.")
    publication_year: Optional[int] = Field(None, description="Year of publication")
    sections_available: List[str] = Field(
        default_factory=list,
        description="Section types present in the full text",
    )


class PromotedTriple(BaseModel):
    """
    A fully enriched LLM triple ready for Neo4j storage.

    Contains normalized entities, canonical predicate, full provenance,
    evidence strength classification, and source metadata.

    Requirements: 1.1, 2.3, 2.4, 6.4
    """

    # Normalized subject entity
    subject_id: str = Field(..., description="Canonical ontology ID or 'ungrounded:{text}'")
    subject_name: str = Field(..., description="Canonical name of the subject")
    subject_type: str = Field(..., description="Entity type (taxon, disease, gene, etc.)")
    subject_grounded: bool = Field(..., description="Whether entity was grounded to an ontology")
    subject_ontology: Optional[str] = Field(None, description="Ontology name if grounded")

    # Normalized object entity
    object_id: str = Field(..., description="Canonical ontology ID or 'ungrounded:{text}'")
    object_name: str = Field(..., description="Canonical name of the object")
    object_type: str = Field(..., description="Entity type (taxon, disease, gene, etc.)")
    object_grounded: bool = Field(..., description="Whether entity was grounded to an ontology")
    object_ontology: Optional[str] = Field(None, description="Ontology name if grounded")

    # Predicate information
    raw_predicate: str = Field(..., description="Original LLM-extracted predicate")
    canonical_predicate: str = Field(..., description="Normalized form (e.g., 'PRODUCES')")
    predicate_category: str = Field(..., description="Category (e.g., 'biosynthetic')")
    is_novel_predicate: bool = Field(..., description="True if not in PREDICATE_NORMALIZATION")
    relationship_type: str = Field(
        ..., description="Neo4j relationship type (canonical form or RELATES_TO)"
    )

    # Provenance
    provenance: ProvenanceMetadata = Field(..., description="Full provenance record")

    # Evidence
    evidence_strength: str = Field(..., description="'strong' | 'moderate' | 'weak'")
    confidence: float = Field(..., ge=0.5, le=1.0, description="Extraction confidence [0.5, 1.0]")

    # Source
    paper_id: str = Field(..., description="Identifier of the source paper")
    section_type: str = Field(..., description="Section from which triple was extracted")
    extracted_at: str = Field(..., description="ISO 8601 timestamp of extraction")

    @field_validator("evidence_strength")
    @classmethod
    def validate_evidence_strength(cls, v: str) -> str:
        allowed = {"strong", "moderate", "weak"}
        if v not in allowed:
            raise ValueError(f"evidence_strength must be one of {allowed}, got '{v}'")
        return v


class OpenWorldClaim(BaseModel):
    """
    An aggregated ScientificClaim of claim_type 'open_world'.

    Groups multiple LLM-extracted triples reporting the same
    (normalized_subject, canonical_predicate, normalized_object) across
    papers into a single claim with consensus metrics.

    Requirements: 3.1, 3.3, 6.4
    """

    claim_id: str = Field(..., description="UUID identifier for the claim")
    claim_type: str = Field(default="open_world", description="Claim type discriminator")

    # Normalized triple
    subject_id: str = Field(..., description="Canonical ontology ID of subject")
    subject_name: str = Field(..., description="Canonical name of the subject")
    canonical_predicate: str = Field(..., description="Normalized predicate form")
    object_id: str = Field(..., description="Canonical ontology ID of object")
    object_name: str = Field(..., description="Canonical name of the object")

    # Evidence aggregation
    supporting_papers: List[str] = Field(
        default_factory=list, description="Unique paper_ids supporting this claim"
    )
    paper_count: int = Field(default=0, description="Number of distinct supporting papers")
    consensus_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Arithmetic mean of confidence scores"
    )
    evidence_strength: str = Field(..., description="'strong' | 'moderate' | 'weak'")

    # Temporal
    first_reported: str = Field(..., description="Earliest extraction timestamp (ISO 8601)")
    last_updated: str = Field(..., description="Most recent extraction timestamp (ISO 8601)")

    # Individual evidence items
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list, description="Per-paper provenance and strength"
    )

    @field_validator("evidence_strength")
    @classmethod
    def validate_evidence_strength(cls, v: str) -> str:
        allowed = {"strong", "moderate", "weak"}
        if v not in allowed:
            raise ValueError(f"evidence_strength must be one of {allowed}, got '{v}'")
        return v

    @field_validator("paper_count")
    @classmethod
    def validate_paper_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"paper_count must be non-negative, got {v}")
        return v

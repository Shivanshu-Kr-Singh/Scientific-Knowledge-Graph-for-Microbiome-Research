"""
graph/semantic_relationships.py
--------------------------------
Semantic relationship data models for the knowledge graph.

This module defines rich scientific relationships with semantic properties
instead of flat adjacency. Each relationship type carries domain-specific
properties (statistical measures, intervention details, methodology info).

Requirements: 2.1, 2.2, 2.3, 2.4
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from graph.provenance import ProvenanceMetadata


class RelationType(str, Enum):
    """
    Enumeration of relationship types in the knowledge graph.
    
    Each type corresponds to a specific scientific claim pattern
    with its own property schema.
    
    Requirements: 2.1, 2.2, 2.3
    """
    REPORTS_ASSOCIATION = "REPORTS_ASSOCIATION"
    REPORTS_INTERVENTION_EFFECT = "REPORTS_INTERVENTION_EFFECT"
    USES_METHODOLOGY = "USES_METHODOLOGY"


class SemanticRelationship(BaseModel):
    """
    A relationship with rich scientific semantics.
    
    Unlike flat adjacency edges, semantic relationships carry domain-specific
    properties that capture the scientific meaning of the claim:
    - Associations include direction, statistical measures, effect sizes
    - Interventions include type, duration, dosage, effect direction
    - Methodology includes sequencing platform, sample size, data availability
    
    Requirements: 2.1, 2.2, 2.3, 2.4
    """
    
    # Core relationship (Requirement 2.1, 2.2, 2.3)
    source_entity: str = Field(..., description="Source entity identifier (e.g., paper ID, taxon ID)")
    target_entity: str = Field(..., description="Target entity identifier (e.g., taxon ID, disease ID)")
    relation_type: RelationType = Field(..., description="Type of relationship")
    
    # Scientific semantics (relation-type specific)
    # Requirements: 2.1 (associations), 2.2 (interventions), 2.3 (methodology)
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relation-type specific properties"
    )
    # For REPORTS_ASSOCIATION (Requirement 2.1):
    #   - direction: "increased" | "decreased" | "no_change"
    #   - comparison: "disease vs healthy" | "pre vs post"
    #   - statistical_measure: "LDA score" | "fold change" | "relative abundance"
    #   - effect_size: float
    #   - p_value: float
    #   - adjusted_p_value: Optional[float]
    #
    # For REPORTS_INTERVENTION_EFFECT (Requirement 2.2):
    #   - intervention_type: "probiotic" | "FMT" | "diet" | "antibiotic"
    #   - effect_direction: "increased" | "decreased"
    #   - duration: "4 weeks" | "6 months"
    #   - dosage: Optional[str]
    #   - sample_size: Optional[int]
    #
    # For USES_METHODOLOGY (Requirement 2.3):
    #   - method_name: "16S rRNA" | "shotgun metagenomics"
    #   - sequencing_platform: "Illumina" | "PacBio"
    #   - sample_size: int
    #   - data_availability: Optional[str]
    
    # Provenance (Requirement 3.1, 3.2)
    provenance: ProvenanceMetadata = Field(..., description="Complete provenance tracking")
    
    # Quality indicators (Requirement 2.4, 5.1, 5.2, 5.3)
    evidence_strength: str = Field(..., description="strong | moderate | weak")
    extraction_confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence 0.0-1.0")
    
    @field_validator('evidence_strength')
    @classmethod
    def validate_evidence_strength(cls, v: str) -> str:
        """
        Validate evidence_strength is one of the allowed values.
        
        Requirement 5.1, 5.2, 5.3: Evidence strength classification
        """
        allowed = {"strong", "moderate", "weak", "conflicting"}
        if v not in allowed:
            raise ValueError(f"evidence_strength must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('extraction_confidence')
    @classmethod
    def validate_extraction_confidence(cls, v: float) -> float:
        """
        Validate extraction_confidence meets minimum threshold.
        
        Requirement 2.4: System SHALL only create relationships with
        extraction confidence >= 0.5
        """
        if v < 0.5:
            raise ValueError(
                f"extraction_confidence must be >= 0.5 (Requirement 2.4), got {v}"
            )
        return v
    
    @field_validator('properties')
    @classmethod
    def validate_properties_structure(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """
        Validate that properties dict contains required fields for the relation type.
        
        Requirements: 2.1, 2.2, 2.3
        """
        # Note: relation_type is not yet available in field_validator context
        # This validation will be performed in model_validator
        return v
    
    def validate_association_properties(self) -> None:
        """
        Validate properties for REPORTS_ASSOCIATION relationships.
        
        Requirement 2.1: Associations must capture direction, comparison context,
        statistical measure type, effect size, and p-value.
        """
        required_fields = {"direction", "comparison", "statistical_measure"}
        missing = required_fields - set(self.properties.keys())
        if missing:
            raise ValueError(
                f"REPORTS_ASSOCIATION requires properties: {required_fields}, "
                f"missing: {missing}"
            )
        
        # Validate direction values
        direction = self.properties.get("direction")
        if direction not in {"increased", "decreased", "no_change"}:
            raise ValueError(
                f"direction must be 'increased', 'decreased', or 'no_change', "
                f"got '{direction}'"
            )
        
        # Validate p_value if present
        if "p_value" in self.properties:
            p_value = self.properties["p_value"]
            if not isinstance(p_value, (int, float)) or not (0.0 <= p_value <= 1.0):
                raise ValueError(f"p_value must be in range [0.0, 1.0], got {p_value}")
        
        # Validate adjusted_p_value if present
        if "adjusted_p_value" in self.properties:
            adj_p = self.properties["adjusted_p_value"]
            if not isinstance(adj_p, (int, float)) or not (0.0 <= adj_p <= 1.0):
                raise ValueError(
                    f"adjusted_p_value must be in range [0.0, 1.0], got {adj_p}"
                )
    
    def validate_intervention_properties(self) -> None:
        """
        Validate properties for REPORTS_INTERVENTION_EFFECT relationships.
        
        Requirement 2.2: Interventions must capture intervention type,
        effect direction, duration, dosage, and sample size.
        """
        required_fields = {"intervention_type", "effect_direction"}
        missing = required_fields - set(self.properties.keys())
        if missing:
            raise ValueError(
                f"REPORTS_INTERVENTION_EFFECT requires properties: {required_fields}, "
                f"missing: {missing}"
            )
        
        # Validate intervention_type values
        intervention_type = self.properties.get("intervention_type")
        allowed_types = {"probiotic", "FMT", "diet", "antibiotic", "prebiotic", "synbiotic", "other"}
        if intervention_type not in allowed_types:
            raise ValueError(
                f"intervention_type must be one of {allowed_types}, "
                f"got '{intervention_type}'"
            )
        
        # Validate effect_direction values
        effect_direction = self.properties.get("effect_direction")
        if effect_direction not in {"increased", "decreased", "no_change"}:
            raise ValueError(
                f"effect_direction must be 'increased', 'decreased', or 'no_change', "
                f"got '{effect_direction}'"
            )
        
        # Validate sample_size if present
        if "sample_size" in self.properties:
            sample_size = self.properties["sample_size"]
            if not isinstance(sample_size, int) or sample_size <= 0:
                raise ValueError(f"sample_size must be a positive integer, got {sample_size}")
    
    def validate_methodology_properties(self) -> None:
        """
        Validate properties for USES_METHODOLOGY relationships.
        
        Requirement 2.3: Methodology must capture method name, sequencing platform,
        sample size, and data availability status.
        """
        required_fields = {"method_name"}
        missing = required_fields - set(self.properties.keys())
        if missing:
            raise ValueError(
                f"USES_METHODOLOGY requires properties: {required_fields}, "
                f"missing: {missing}"
            )
        
        # Validate sample_size if present
        if "sample_size" in self.properties:
            sample_size = self.properties["sample_size"]
            if not isinstance(sample_size, int) or sample_size <= 0:
                raise ValueError(f"sample_size must be a positive integer, got {sample_size}")
    
    def model_post_init(self, __context) -> None:
        """
        Perform relation-type specific validation after model initialization.
        
        Requirements: 2.1, 2.2, 2.3
        """
        if self.relation_type == RelationType.REPORTS_ASSOCIATION:
            self.validate_association_properties()
        elif self.relation_type == RelationType.REPORTS_INTERVENTION_EFFECT:
            self.validate_intervention_properties()
        elif self.relation_type == RelationType.USES_METHODOLOGY:
            self.validate_methodology_properties()


# Factory functions for creating semantic relationships

def create_association_relationship(
    source_entity: str,
    target_entity: str,
    direction: str,
    comparison: str,
    statistical_measure: str,
    provenance: ProvenanceMetadata,
    evidence_strength: str,
    extraction_confidence: float,
    effect_size: Optional[float] = None,
    p_value: Optional[float] = None,
    adjusted_p_value: Optional[float] = None,
) -> SemanticRelationship:
    """
    Factory function to create a REPORTS_ASSOCIATION relationship.
    
    Requirement 2.1: Extract taxon-disease associations with statistical properties.
    
    Args:
        source_entity: Source entity identifier (e.g., paper ID)
        target_entity: Target entity identifier (e.g., taxon ID)
        direction: "increased" | "decreased" | "no_change"
        comparison: Comparison context (e.g., "T2D vs healthy")
        statistical_measure: Type of measure (e.g., "LDA score", "fold change")
        provenance: Complete provenance metadata
        evidence_strength: "strong" | "moderate" | "weak"
        extraction_confidence: Confidence score [0.5, 1.0]
        effect_size: Optional effect size value
        p_value: Optional p-value [0.0, 1.0]
        adjusted_p_value: Optional adjusted p-value [0.0, 1.0]
    
    Returns:
        SemanticRelationship with REPORTS_ASSOCIATION type
    """
    properties = {
        "direction": direction,
        "comparison": comparison,
        "statistical_measure": statistical_measure,
    }
    
    if effect_size is not None:
        properties["effect_size"] = effect_size
    if p_value is not None:
        properties["p_value"] = p_value
    if adjusted_p_value is not None:
        properties["adjusted_p_value"] = adjusted_p_value
    
    return SemanticRelationship(
        source_entity=source_entity,
        target_entity=target_entity,
        relation_type=RelationType.REPORTS_ASSOCIATION,
        properties=properties,
        provenance=provenance,
        evidence_strength=evidence_strength,
        extraction_confidence=extraction_confidence,
    )


def create_intervention_relationship(
    source_entity: str,
    target_entity: str,
    intervention_type: str,
    effect_direction: str,
    provenance: ProvenanceMetadata,
    evidence_strength: str,
    extraction_confidence: float,
    duration: Optional[str] = None,
    dosage: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> SemanticRelationship:
    """
    Factory function to create a REPORTS_INTERVENTION_EFFECT relationship.
    
    Requirement 2.2: Extract intervention-taxon effects from RCT or intervention studies.
    
    Args:
        source_entity: Source entity identifier (e.g., paper ID)
        target_entity: Target entity identifier (e.g., taxon ID)
        intervention_type: "probiotic" | "FMT" | "diet" | "antibiotic" | etc.
        effect_direction: "increased" | "decreased" | "no_change"
        provenance: Complete provenance metadata
        evidence_strength: "strong" | "moderate" | "weak"
        extraction_confidence: Confidence score [0.5, 1.0]
        duration: Optional intervention duration (e.g., "4 weeks")
        dosage: Optional dosage information
        sample_size: Optional sample size
    
    Returns:
        SemanticRelationship with REPORTS_INTERVENTION_EFFECT type
    """
    properties = {
        "intervention_type": intervention_type,
        "effect_direction": effect_direction,
    }
    
    if duration is not None:
        properties["duration"] = duration
    if dosage is not None:
        properties["dosage"] = dosage
    if sample_size is not None:
        properties["sample_size"] = sample_size
    
    return SemanticRelationship(
        source_entity=source_entity,
        target_entity=target_entity,
        relation_type=RelationType.REPORTS_INTERVENTION_EFFECT,
        properties=properties,
        provenance=provenance,
        evidence_strength=evidence_strength,
        extraction_confidence=extraction_confidence,
    )


def create_methodology_relationship(
    source_entity: str,
    target_entity: str,
    method_name: str,
    provenance: ProvenanceMetadata,
    evidence_strength: str,
    extraction_confidence: float,
    sequencing_platform: Optional[str] = None,
    sample_size: Optional[int] = None,
    data_availability: Optional[str] = None,
) -> SemanticRelationship:
    """
    Factory function to create a USES_METHODOLOGY relationship.
    
    Requirement 2.3: Extract methodology information (sequencing type, sample size,
    data availability).
    
    Args:
        source_entity: Source entity identifier (e.g., paper ID)
        target_entity: Target entity identifier (e.g., method ID)
        method_name: Method name (e.g., "16S rRNA", "shotgun metagenomics")
        provenance: Complete provenance metadata
        evidence_strength: "strong" | "moderate" | "weak"
        extraction_confidence: Confidence score [0.5, 1.0]
        sequencing_platform: Optional platform (e.g., "Illumina", "PacBio")
        sample_size: Optional sample size
        data_availability: Optional data availability status
    
    Returns:
        SemanticRelationship with USES_METHODOLOGY type
    """
    properties = {
        "method_name": method_name,
    }
    
    if sequencing_platform is not None:
        properties["sequencing_platform"] = sequencing_platform
    if sample_size is not None:
        properties["sample_size"] = sample_size
    if data_availability is not None:
        properties["data_availability"] = data_availability
    
    return SemanticRelationship(
        source_entity=source_entity,
        target_entity=target_entity,
        relation_type=RelationType.USES_METHODOLOGY,
        properties=properties,
        provenance=provenance,
        evidence_strength=evidence_strength,
        extraction_confidence=extraction_confidence,
    )

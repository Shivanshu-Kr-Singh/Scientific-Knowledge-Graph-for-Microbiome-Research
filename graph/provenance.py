"""
graph/provenance.py
-------------------
Provenance tracking for knowledge graph relationships.

This module implements complete lineage tracking for every graph edge,
recording the source text, extraction method, timestamp, and confidence
for reproducibility and validation.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 10.2
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from nlp.enriched_record import EnrichedPaperRecord, ParsedSection
from graph.extractor_registry import get_registered_method_ids


# Get registered extraction methods from the registry
# Requirements: 10.1, 10.2
def _get_registered_methods():
    """Get the current set of registered extraction methods."""
    return get_registered_method_ids()


REGISTERED_EXTRACTION_METHODS = _get_registered_methods()


class ProvenanceMetadata(BaseModel):
    """
    Complete provenance information for a graph relationship.
    
    Tracks the complete lineage of every graph edge from source text
    to final relationship, enabling reproducibility and validation.
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    
    # Source identification (Requirement 3.1)
    paper_id: str = Field(..., description="DOI, PMID, or title")
    section_type: str = Field(..., description="abstract | methods | results | discussion")
    source_sentence: str = Field(..., min_length=1, description="Exact sentence that supports this relationship")
    sentence_offset: Optional[int] = Field(None, description="Character offset in section")
    
    # Extraction metadata (Requirement 3.2)
    extraction_method: str = Field(..., description="Registered extractor identifier")
    extraction_timestamp: datetime = Field(..., description="When extraction occurred")
    extractor_version: str = Field(..., description="Version of the extraction model/method")
    llm_prompt_hash: Optional[str] = Field(None, description="Hash of LLM prompt if applicable")
    
    # Confidence and validation (Requirement 3.5)
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence 0.0-1.0")
    validation_status: str = Field(
        default="unvalidated",
        description="unvalidated | human_verified | cross_validated"
    )
    validator_id: Optional[str] = Field(None, description="User ID if human-validated")
    
    # Context (Requirement 3.4)
    surrounding_context: Optional[str] = Field(None, description="±2 sentences for context")
    figure_table_ref: Optional[str] = Field(None, description="If claim references a figure/table")
    
    @field_validator('section_type')
    @classmethod
    def validate_section_type(cls, v: str) -> str:
        """Validate section_type is one of the allowed values."""
        allowed = {"abstract", "methods", "results", "discussion", "introduction", "other"}
        if v not in allowed:
            raise ValueError(f"section_type must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('validation_status')
    @classmethod
    def validate_validation_status(cls, v: str) -> str:
        """Validate validation_status is one of the allowed values."""
        allowed = {"unvalidated", "human_verified", "cross_validated"}
        if v not in allowed:
            raise ValueError(f"validation_status must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator('extraction_method')
    @classmethod
    def validate_extraction_method(cls, v: str) -> str:
        """
        Validate extraction_method is registered.
        
        Requirement 10.2: System SHALL validate that extraction_method exists
        in the registered extractors list before allowing relationship creation.
        """
        if v not in REGISTERED_EXTRACTION_METHODS:
            raise ValueError(
                f"extraction_method '{v}' is not registered. "
                f"Valid methods: {sorted(REGISTERED_EXTRACTION_METHODS)}"
            )
        return v


class ProvenanceEncoder:
    """
    Encodes provenance metadata for graph relationships.
    
    Creates complete provenance records for every extracted relationship,
    tracking extraction method, version, timestamp, and linking back to
    source sentences and sections.
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 10.2
    """
    
    def encode(
        self,
        paper: EnrichedPaperRecord,
        section: ParsedSection,
        sentence: str,
        extraction_method: str,
        confidence: float,
        extractor_version: str = "1.0",
        llm_prompt_hash: Optional[str] = None,
        surrounding_context: Optional[str] = None,
        figure_table_ref: Optional[str] = None,
        sentence_offset: Optional[int] = None,
    ) -> ProvenanceMetadata:
        """
        Create provenance metadata for a relationship extracted from text.
        
        Preconditions:
        - paper is a valid EnrichedPaperRecord with non-null identifier
        - section is a ParsedSection from paper.sections
        - sentence is a non-empty string from section.content
        - extraction_method is a registered extractor identifier
        - confidence is in range [0.0, 1.0]
        
        Postconditions:
        - Returns ProvenanceMetadata with all required fields populated
        - extraction_timestamp is set to current UTC time
        - source_sentence exactly matches input sentence
        
        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 10.2
        
        Args:
            paper: The enriched paper record containing the relationship
            section: The parsed section containing the source sentence
            sentence: The exact sentence supporting this relationship
            extraction_method: Registered extractor identifier (e.g., "regex_ner")
            confidence: Extraction confidence score [0.0, 1.0]
            extractor_version: Version of the extraction model/method
            llm_prompt_hash: Hash of LLM prompt if applicable (Requirement 3.3)
            surrounding_context: ±2 sentences for context (Requirement 3.4)
            figure_table_ref: Figure/table reference if applicable (Requirement 3.4)
            sentence_offset: Character offset in section
        
        Returns:
            ProvenanceMetadata with all required fields populated
        
        Raises:
            ValueError: If validation fails (invalid confidence, unregistered method, etc.)
        """
        # Validate preconditions
        if not sentence or not sentence.strip():
            raise ValueError("sentence must be a non-empty string")
        
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in range [0.0, 1.0], got {confidence}")
        
        if extraction_method not in REGISTERED_EXTRACTION_METHODS:
            raise ValueError(
                f"extraction_method '{extraction_method}' is not registered. "
                f"Valid methods: {sorted(REGISTERED_EXTRACTION_METHODS)}"
            )
        
        # Get paper identifier (prefer DOI, fallback to PMID, then title)
        paper_id = paper.doi or paper.pmid or paper.title
        if not paper_id:
            raise ValueError("paper must have at least one of: doi, pmid, or title")
        
        # Create provenance metadata
        # Requirement 3.2: Record extraction method, extractor version, timestamp, confidence
        provenance = ProvenanceMetadata(
            # Source identification (Requirement 3.1)
            paper_id=paper_id,
            section_type=section.section_type,
            source_sentence=sentence.strip(),
            sentence_offset=sentence_offset,
            
            # Extraction metadata (Requirement 3.2)
            extraction_method=extraction_method,
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version=extractor_version,
            llm_prompt_hash=llm_prompt_hash,
            
            # Confidence and validation (Requirement 3.5)
            confidence_score=confidence,
            validation_status="unvalidated",
            validator_id=None,
            
            # Context (Requirement 3.4)
            surrounding_context=surrounding_context,
            figure_table_ref=figure_table_ref,
        )
        
        return provenance
    
    def validate_provenance(self, provenance: ProvenanceMetadata) -> bool:
        """
        Verify that provenance metadata is complete and valid.
        
        Preconditions:
        - provenance is a ProvenanceMetadata instance
        
        Postconditions:
        - Returns True if all required fields are present and valid
        - Returns False if any field is missing or invalid
        
        Requirements: 3.5
        
        Args:
            provenance: The provenance metadata to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields are present and non-empty
            # Requirement 3.5: System SHALL reject any relationship that lacks
            # required provenance fields
            required_fields = [
                'paper_id',
                'section_type',
                'source_sentence',
                'extraction_method',
                'extraction_timestamp',
                'extractor_version',
                'confidence_score',
            ]
            
            for field in required_fields:
                value = getattr(provenance, field, None)
                if value is None or (isinstance(value, str) and not value.strip()):
                    return False
            
            # Validate confidence score is in range [0.0, 1.0]
            # Requirement 3.5: SHALL validate that provenance values are reasonable
            if not (0.0 <= provenance.confidence_score <= 1.0):
                return False
            
            # Validate extraction_method is registered
            # Requirement 10.2
            if provenance.extraction_method not in REGISTERED_EXTRACTION_METHODS:
                return False
            
            # Validate timestamp is positive (not in the future by more than 1 minute)
            # Requirement 3.5: positive timestamps
            now = datetime.now(timezone.utc)
            if provenance.extraction_timestamp > now:
                # Allow small clock skew (1 minute)
                if (provenance.extraction_timestamp - now).total_seconds() > 60:
                    return False
            
            # All validations passed
            return True
            
        except Exception:
            # Any exception during validation means invalid
            return False

"""
graph/data_validator.py
------------------------
Data validation module for knowledge graph relationships.

This module validates relationship data before loading into Neo4j,
ensuring data quality and storing invalid relationships for manual review.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, UTC
import json
import logging
from pathlib import Path

from graph.semantic_relationships import SemanticRelationship


logger = logging.getLogger(__name__)


class ValidationError(BaseModel):
    """
    Represents a validation error for a relationship.
    
    Requirement 14.5: Store invalid relationships for manual review
    """
    relationship_id: str = Field(..., description="Unique identifier for the relationship")
    error_type: str = Field(..., description="Type of validation error")
    error_message: str = Field(..., description="Detailed error message")
    field_name: Optional[str] = Field(None, description="Field that failed validation")
    invalid_value: Optional[Any] = Field(None, description="The invalid value")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When validation failed")


class ValidationResult(BaseModel):
    """
    Result of validating a batch of relationships.
    
    Requirement 14.5: Track valid and invalid relationships
    """
    valid_relationships: List[SemanticRelationship] = Field(default_factory=list)
    invalid_relationships: List[Tuple[SemanticRelationship, List[ValidationError]]] = Field(default_factory=list)
    total_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0


class DataValidator:
    """
    Validates relationship data before loading into Neo4j.
    
    This validator ensures:
    - Confidence scores are in range [0.0, 1.0]
    - P-values are in range [0.0, 1.0]
    - Direction values are valid
    - Evidence strength values are valid
    
    Invalid relationships are stored in a validation queue for manual review.
    
    Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
    """
    
    def __init__(self, validation_queue_path: Optional[str] = None):
        """
        Initialize the data validator.
        
        Args:
            validation_queue_path: Path to store invalid relationships for manual review.
                                   Defaults to 'data/validation_queue.json'
        
        Requirement 14.5: Store invalid relationships in validation queue
        """
        if validation_queue_path is None:
            validation_queue_path = "data/validation_queue.json"
        
        self.validation_queue_path = Path(validation_queue_path)
        self.validation_queue_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataValidator with queue at {self.validation_queue_path}")
    
    def validate_confidence_score(
        self,
        relationship: SemanticRelationship
    ) -> List[ValidationError]:
        """
        Validate that confidence score is in range [0.0, 1.0].
        
        Requirement 14.1: System SHALL validate that all confidence scores
        are in the range [0.0, 1.0]
        
        Args:
            relationship: SemanticRelationship to validate
        
        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []
        
        confidence = relationship.extraction_confidence
        if not isinstance(confidence, (int, float)):
            errors.append(ValidationError(
                relationship_id=self._get_relationship_id(relationship),
                error_type="invalid_type",
                error_message=f"Confidence score must be numeric, got {type(confidence).__name__}",
                field_name="extraction_confidence",
                invalid_value=confidence
            ))
        elif not (0.0 <= confidence <= 1.0):
            errors.append(ValidationError(
                relationship_id=self._get_relationship_id(relationship),
                error_type="out_of_range",
                error_message=f"Confidence score must be in range [0.0, 1.0], got {confidence}",
                field_name="extraction_confidence",
                invalid_value=confidence
            ))
        
        return errors
    
    def validate_p_value(
        self,
        relationship: SemanticRelationship
    ) -> List[ValidationError]:
        """
        Validate that p_value (when present) is in range [0.0, 1.0].
        
        Requirement 14.2: System SHALL validate that all p_values (when present)
        are in the range [0.0, 1.0]
        
        Args:
            relationship: SemanticRelationship to validate
        
        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []
        
        # Check if p_value exists in properties
        if "p_value" not in relationship.properties:
            return errors  # p_value is optional
        
        p_value = relationship.properties["p_value"]
        
        if p_value is None:
            return errors  # None is acceptable for optional field
        
        if not isinstance(p_value, (int, float)):
            errors.append(ValidationError(
                relationship_id=self._get_relationship_id(relationship),
                error_type="invalid_type",
                error_message=f"p_value must be numeric, got {type(p_value).__name__}",
                field_name="p_value",
                invalid_value=p_value
            ))
        elif not (0.0 <= p_value <= 1.0):
            errors.append(ValidationError(
                relationship_id=self._get_relationship_id(relationship),
                error_type="out_of_range",
                error_message=f"p_value must be in range [0.0, 1.0], got {p_value}",
                field_name="p_value",
                invalid_value=p_value
            ))
        
        # Also check adjusted_p_value if present
        if "adjusted_p_value" in relationship.properties:
            adj_p_value = relationship.properties["adjusted_p_value"]
            
            if adj_p_value is not None:
                if not isinstance(adj_p_value, (int, float)):
                    errors.append(ValidationError(
                        relationship_id=self._get_relationship_id(relationship),
                        error_type="invalid_type",
                        error_message=f"adjusted_p_value must be numeric, got {type(adj_p_value).__name__}",
                        field_name="adjusted_p_value",
                        invalid_value=adj_p_value
                    ))
                elif not (0.0 <= adj_p_value <= 1.0):
                    errors.append(ValidationError(
                        relationship_id=self._get_relationship_id(relationship),
                        error_type="out_of_range",
                        error_message=f"adjusted_p_value must be in range [0.0, 1.0], got {adj_p_value}",
                        field_name="adjusted_p_value",
                        invalid_value=adj_p_value
                    ))
        
        return errors
    
    def validate_direction(
        self,
        relationship: SemanticRelationship
    ) -> List[ValidationError]:
        """
        Validate that direction values are in the allowed set.
        
        Requirement 14.3: System SHALL validate that direction values are in
        the set {"increased", "decreased", "no_change"}
        
        Args:
            relationship: SemanticRelationship to validate
        
        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []
        
        allowed_directions = {"increased", "decreased", "no_change"}
        
        # Check direction in properties (for associations and interventions)
        if "direction" in relationship.properties:
            direction = relationship.properties["direction"]
            
            if direction is not None and direction not in allowed_directions:
                errors.append(ValidationError(
                    relationship_id=self._get_relationship_id(relationship),
                    error_type="invalid_value",
                    error_message=f"direction must be one of {allowed_directions}, got '{direction}'",
                    field_name="direction",
                    invalid_value=direction
                ))
        
        # Check effect_direction for intervention relationships
        if "effect_direction" in relationship.properties:
            effect_direction = relationship.properties["effect_direction"]
            
            if effect_direction is not None and effect_direction not in allowed_directions:
                errors.append(ValidationError(
                    relationship_id=self._get_relationship_id(relationship),
                    error_type="invalid_value",
                    error_message=f"effect_direction must be one of {allowed_directions}, got '{effect_direction}'",
                    field_name="effect_direction",
                    invalid_value=effect_direction
                ))
        
        return errors
    
    def validate_evidence_strength(
        self,
        relationship: SemanticRelationship
    ) -> List[ValidationError]:
        """
        Validate that evidence_strength is in the allowed set.
        
        Requirement 14.4: System SHALL validate that evidence_strength values
        are in the set {"strong", "moderate", "weak", "conflicting"}
        
        Args:
            relationship: SemanticRelationship to validate
        
        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []
        
        allowed_strengths = {"strong", "moderate", "weak", "conflicting"}
        evidence_strength = relationship.evidence_strength
        
        if evidence_strength not in allowed_strengths:
            errors.append(ValidationError(
                relationship_id=self._get_relationship_id(relationship),
                error_type="invalid_value",
                error_message=f"evidence_strength must be one of {allowed_strengths}, got '{evidence_strength}'",
                field_name="evidence_strength",
                invalid_value=evidence_strength
            ))
        
        return errors
    
    def validate_relationship(
        self,
        relationship: SemanticRelationship
    ) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a single relationship against all validation rules.
        
        Requirements: 14.1, 14.2, 14.3, 14.4
        
        Args:
            relationship: SemanticRelationship to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors = []
        
        # Requirement 14.1: Validate confidence scores
        all_errors.extend(self.validate_confidence_score(relationship))
        
        # Requirement 14.2: Validate p_values
        all_errors.extend(self.validate_p_value(relationship))
        
        # Requirement 14.3: Validate direction values
        all_errors.extend(self.validate_direction(relationship))
        
        # Requirement 14.4: Validate evidence_strength values
        all_errors.extend(self.validate_evidence_strength(relationship))
        
        is_valid = len(all_errors) == 0
        
        return is_valid, all_errors
    
    def validate_batch(
        self,
        relationships: List[SemanticRelationship]
    ) -> ValidationResult:
        """
        Validate a batch of relationships.
        
        Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
        
        Args:
            relationships: List of SemanticRelationship objects to validate
        
        Returns:
            ValidationResult with valid and invalid relationships separated
        """
        result = ValidationResult(total_count=len(relationships))
        
        for relationship in relationships:
            is_valid, errors = self.validate_relationship(relationship)
            
            if is_valid:
                result.valid_relationships.append(relationship)
                result.valid_count += 1
            else:
                result.invalid_relationships.append((relationship, errors))
                result.invalid_count += 1
                
                # Log validation failure
                logger.warning(
                    f"Relationship validation failed: {self._get_relationship_id(relationship)} "
                    f"with {len(errors)} error(s)"
                )
        
        logger.info(
            f"Validation complete: {result.valid_count} valid, "
            f"{result.invalid_count} invalid out of {result.total_count} total"
        )
        
        return result
    
    def store_invalid_relationships(
        self,
        invalid_relationships: List[Tuple[SemanticRelationship, List[ValidationError]]]
    ) -> None:
        """
        Store invalid relationships in validation queue for manual review.
        
        Requirement 14.5: System SHALL store relationships that fail validation
        in a separate validation queue for manual review rather than discarding
        them completely
        
        Args:
            invalid_relationships: List of (relationship, errors) tuples
        """
        if not invalid_relationships:
            logger.info("No invalid relationships to store")
            return
        
        # Load existing queue if it exists
        existing_queue = []
        if self.validation_queue_path.exists():
            try:
                with open(self.validation_queue_path, 'r', encoding='utf-8') as f:
                    existing_queue = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load existing validation queue: {e}")
                existing_queue = []
        
        # Prepare new entries
        new_entries = []
        for relationship, errors in invalid_relationships:
            entry = {
                "relationship_id": self._get_relationship_id(relationship),
                "source_entity": relationship.source_entity,
                "target_entity": relationship.target_entity,
                "relation_type": relationship.relation_type.value,
                "properties": relationship.properties,
                "provenance": {
                    "paper_id": relationship.provenance.paper_id,
                    "section_type": relationship.provenance.section_type,
                    "source_sentence": relationship.provenance.source_sentence,
                    "extraction_method": relationship.provenance.extraction_method,
                    "extraction_timestamp": relationship.provenance.extraction_timestamp.isoformat(),
                },
                "extraction_confidence": relationship.extraction_confidence,
                "evidence_strength": relationship.evidence_strength,
                "validation_errors": [
                    {
                        "error_type": error.error_type,
                        "error_message": error.error_message,
                        "field_name": error.field_name,
                        "invalid_value": error.invalid_value,
                        "timestamp": error.timestamp.isoformat(),
                    }
                    for error in errors
                ],
                "queued_at": datetime.now(UTC).isoformat(),
            }
            new_entries.append(entry)
        
        # Append to existing queue
        existing_queue.extend(new_entries)
        
        # Write back to file
        try:
            with open(self.validation_queue_path, 'w', encoding='utf-8') as f:
                json.dump(existing_queue, f, indent=2)
            
            logger.info(
                f"Stored {len(new_entries)} invalid relationships in validation queue "
                f"at {self.validation_queue_path}"
            )
        except Exception as e:
            logger.error(f"Failed to write validation queue: {e}")
            raise
    
    def _get_relationship_id(self, relationship: SemanticRelationship) -> str:
        """
        Generate a unique identifier for a relationship.
        
        Args:
            relationship: SemanticRelationship
        
        Returns:
            Unique identifier string
        """
        return (
            f"{relationship.source_entity}--"
            f"{relationship.relation_type.value}--"
            f"{relationship.target_entity}--"
            f"{relationship.provenance.paper_id}"
        )
    
    def get_validation_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the validation queue.
        
        Returns:
            Dictionary with queue statistics
        """
        if not self.validation_queue_path.exists():
            return {
                "queue_size": 0,
                "queue_path": str(self.validation_queue_path),
                "exists": False,
            }
        
        try:
            with open(self.validation_queue_path, 'r', encoding='utf-8') as f:
                queue = json.load(f)
            
            # Count error types
            error_type_counts = {}
            for entry in queue:
                for error in entry.get("validation_errors", []):
                    error_type = error.get("error_type", "unknown")
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            return {
                "queue_size": len(queue),
                "queue_path": str(self.validation_queue_path),
                "exists": True,
                "error_type_counts": error_type_counts,
            }
        except Exception as e:
            logger.error(f"Failed to read validation queue stats: {e}")
            return {
                "queue_size": 0,
                "queue_path": str(self.validation_queue_path),
                "exists": True,
                "error": str(e),
            }

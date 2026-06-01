"""
api/input_validator.py
----------------------
Input validation and sanitization for the Query API.

This module provides comprehensive input validation including:
- Entity validation against the Neo4j knowledge graph
- Numeric threshold validation
- Input sanitization to prevent Cypher injection attacks
- Detailed error response generation

**Validates: Requirement 18.2**
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from neo4j import Driver
from loguru import logger
import re


class ValidationError(BaseModel):
    """Detailed validation error information."""
    field: str
    value: Any
    error_type: str
    message: str


class ValidationResult(BaseModel):
    """Result of input validation."""
    is_valid: bool
    errors: List[ValidationError] = []
    sanitized_values: Dict[str, Any] = {}


class InputValidator:
    """
    Validates and sanitizes user inputs for the Query API.
    
    **Validates: Requirement 18.2**
    """
    
    # Cypher injection patterns to detect and block
    CYPHER_INJECTION_PATTERNS = [
        r'(?i)\bMATCH\b',
        r'(?i)\bWHERE\b',
        r'(?i)\bRETURN\b',
        r'(?i)\bCREATE\b',
        r'(?i)\bDELETE\b',
        r'(?i)\bDETACH\b',
        r'(?i)\bMERGE\b',
        r'(?i)\bSET\b',
        r'(?i)\bREMOVE\b',
        r'(?i)\bDROP\b',
        r'(?i)\bCALL\b',
        r'(?i)\bUNION\b',
        r'(?i)\bUNWIND\b',
        r'(?i)\bWITH\b',
        r'[;\{\}]',  # Semicolons and braces
        r'--',  # SQL-style comments
        r'/\*',  # Multi-line comments
        r'\*/',
    ]
    
    # Allowed characters for entity names (alphanumeric, spaces, hyphens, apostrophes, parentheses, underscores)
    ENTITY_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-\'().,_]+$')
    
    # Maximum lengths for string inputs
    MAX_ENTITY_NAME_LENGTH = 200
    MAX_LIST_SIZE = 50
    
    # Numeric validation ranges
    MIN_YEAR = 1900
    MAX_YEAR = 2100
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    MIN_PAPERS = 1
    MAX_PAPERS = 10000
    MIN_SAMPLE_SIZE = 1
    MAX_SAMPLE_SIZE = 1000000
    MIN_TOP_N = 1
    MAX_TOP_N = 100
    
    def __init__(self, neo4j_driver: Optional[Driver] = None):
        """
        Initialize the input validator.
        
        Args:
            neo4j_driver: Optional Neo4j driver for entity validation.
                         If None, entity validation against database is skipped.
        """
        self.driver = neo4j_driver
        self._entity_cache: Dict[str, set] = {
            'diseases': set(),
            'taxa': set(),
            'methods': set()
        }
        self._cache_loaded = False
    
    def _load_entity_cache(self) -> None:
        """
        Load entity names from the database into cache.
        
        This is called lazily on first validation request.
        """
        if self._cache_loaded or not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                # Load disease names
                result = session.run("MATCH (d:Disease) RETURN d.name AS name LIMIT 10000")
                self._entity_cache['diseases'] = {record['name'] for record in result if record['name']}
                
                # Load taxon names
                result = session.run("MATCH (t:Taxon) RETURN t.name AS name LIMIT 10000")
                self._entity_cache['taxa'] = {record['name'] for record in result if record['name']}
                
                # Load method names
                result = session.run("MATCH (m:Method) RETURN m.name AS name LIMIT 10000")
                self._entity_cache['methods'] = {record['name'] for record in result if record['name']}
                
                self._cache_loaded = True
                logger.info(
                    f"Loaded entity cache: {len(self._entity_cache['diseases'])} diseases, "
                    f"{len(self._entity_cache['taxa'])} taxa, {len(self._entity_cache['methods'])} methods"
                )
        except Exception as e:
            logger.error(f"Failed to load entity cache: {e}")
            # Continue without cache - validation will be less strict
    
    def sanitize_string(self, value: str, field_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Sanitize a string input to prevent injection attacks.
        
        Args:
            value: The string value to sanitize
            field_name: Name of the field being validated (for error messages)
        
        Returns:
            Tuple of (is_valid, sanitized_value, error_message)
        """
        if not isinstance(value, str):
            return False, None, f"{field_name} must be a string"
        
        # Check for empty string
        if not value.strip():
            return False, None, f"{field_name} cannot be empty"
        
        # Check length
        if len(value) > self.MAX_ENTITY_NAME_LENGTH:
            return False, None, f"{field_name} exceeds maximum length of {self.MAX_ENTITY_NAME_LENGTH}"
        
        # Check for Cypher injection patterns
        for pattern in self.CYPHER_INJECTION_PATTERNS:
            if re.search(pattern, value):
                logger.warning(f"Potential Cypher injection detected in {field_name}: {value}")
                return False, None, f"{field_name} contains invalid characters or patterns"
        
        # Check against allowed character pattern
        if not self.ENTITY_NAME_PATTERN.match(value):
            return False, None, f"{field_name} contains invalid characters"
        
        # Sanitize by trimming whitespace
        sanitized = value.strip()
        
        return True, sanitized, None
    
    def validate_entity_name(
        self,
        entity_name: str,
        entity_type: str,
        field_name: str,
        check_existence: bool = True
    ) -> ValidationError | None:
        """
        Validate an entity name (disease, taxon, method).
        
        Args:
            entity_name: The entity name to validate
            entity_type: Type of entity ('disease', 'taxon', 'method')
            field_name: Name of the field being validated
            check_existence: Whether to check if entity exists in database
        
        Returns:
            ValidationError if validation fails, None if valid
        """
        # Sanitize the string
        is_valid, sanitized, error_msg = self.sanitize_string(entity_name, field_name)
        if not is_valid:
            return ValidationError(
                field=field_name,
                value=entity_name,
                error_type="sanitization_failed",
                message=error_msg
            )
        
        # Check existence in database if requested and driver is available
        if check_existence and self.driver:
            # Load cache if not already loaded
            if not self._cache_loaded:
                self._load_entity_cache()
            
            # Map entity type to cache key
            cache_key = f"{entity_type}s" if entity_type in ['disease', 'taxon', 'method'] else entity_type
            
            if cache_key in self._entity_cache and self._entity_cache[cache_key]:
                if sanitized not in self._entity_cache[cache_key]:
                    # Try case-insensitive match
                    lower_cache = {name.lower() for name in self._entity_cache[cache_key]}
                    if sanitized.lower() not in lower_cache:
                        return ValidationError(
                            field=field_name,
                            value=entity_name,
                            error_type="entity_not_found",
                            message=f"{entity_type.capitalize()} '{sanitized}' not found in knowledge graph"
                        )
        
        return None
    
    def validate_numeric_threshold(
        self,
        value: Any,
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_none: bool = False
    ) -> ValidationError | None:
        """
        Validate a numeric threshold value.
        
        Args:
            value: The numeric value to validate
            field_name: Name of the field being validated
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            allow_none: Whether None is an acceptable value
        
        Returns:
            ValidationError if validation fails, None if valid
        """
        if value is None:
            if allow_none:
                return None
            return ValidationError(
                field=field_name,
                value=value,
                error_type="required_field",
                message=f"{field_name} is required"
            )
        
        # Check type
        if not isinstance(value, (int, float)):
            return ValidationError(
                field=field_name,
                value=value,
                error_type="invalid_type",
                message=f"{field_name} must be a number"
            )
        
        # Check for NaN or infinity
        if isinstance(value, float):
            if value != value:  # NaN check
                return ValidationError(
                    field=field_name,
                    value=value,
                    error_type="invalid_value",
                    message=f"{field_name} cannot be NaN"
                )
            if value == float('inf') or value == float('-inf'):
                return ValidationError(
                    field=field_name,
                    value=value,
                    error_type="invalid_value",
                    message=f"{field_name} cannot be infinity"
                )
        
        # Check range
        if min_value is not None and value < min_value:
            return ValidationError(
                field=field_name,
                value=value,
                error_type="out_of_range",
                message=f"{field_name} must be >= {min_value}"
            )
        
        if max_value is not None and value > max_value:
            return ValidationError(
                field=field_name,
                value=value,
                error_type="out_of_range",
                message=f"{field_name} must be <= {max_value}"
            )
        
        return None
    
    def validate_string_list(
        self,
        values: List[str],
        field_name: str,
        entity_type: Optional[str] = None,
        check_existence: bool = False
    ) -> List[ValidationError]:
        """
        Validate a list of string values.
        
        Args:
            values: List of strings to validate
            field_name: Name of the field being validated
            entity_type: If provided, validate as entity names of this type
            check_existence: Whether to check entity existence in database
        
        Returns:
            List of ValidationErrors (empty if all valid)
        """
        errors = []
        
        # Check type
        if not isinstance(values, list):
            errors.append(ValidationError(
                field=field_name,
                value=values,
                error_type="invalid_type",
                message=f"{field_name} must be a list"
            ))
            return errors
        
        # Check list size
        if len(values) == 0:
            errors.append(ValidationError(
                field=field_name,
                value=values,
                error_type="empty_list",
                message=f"{field_name} cannot be empty"
            ))
            return errors
        
        if len(values) > self.MAX_LIST_SIZE:
            errors.append(ValidationError(
                field=field_name,
                value=values,
                error_type="list_too_large",
                message=f"{field_name} cannot contain more than {self.MAX_LIST_SIZE} items"
            ))
            return errors
        
        # Validate each item
        for i, value in enumerate(values):
            if entity_type:
                error = self.validate_entity_name(
                    value,
                    entity_type,
                    f"{field_name}[{i}]",
                    check_existence
                )
                if error:
                    errors.append(error)
            else:
                is_valid, _, error_msg = self.sanitize_string(value, f"{field_name}[{i}]")
                if not is_valid:
                    errors.append(ValidationError(
                        field=f"{field_name}[{i}]",
                        value=value,
                        error_type="sanitization_failed",
                        message=error_msg
                    ))
        
        return errors
    
    def validate_year_range(
        self,
        year_start: int,
        year_end: int
    ) -> List[ValidationError]:
        """
        Validate a year range.
        
        Args:
            year_start: Start year
            year_end: End year
        
        Returns:
            List of ValidationErrors (empty if valid)
        """
        errors = []
        
        # Validate individual years
        error = self.validate_numeric_threshold(
            year_start,
            "year_start",
            self.MIN_YEAR,
            self.MAX_YEAR
        )
        if error:
            errors.append(error)
        
        error = self.validate_numeric_threshold(
            year_end,
            "year_end",
            self.MIN_YEAR,
            self.MAX_YEAR
        )
        if error:
            errors.append(error)
        
        # Check that start <= end
        if not errors and year_start > year_end:
            errors.append(ValidationError(
                field="year_range",
                value={"year_start": year_start, "year_end": year_end},
                error_type="invalid_range",
                message=f"year_start ({year_start}) must be <= year_end ({year_end})"
            ))
        
        return errors
    
    def invalidate_cache(self) -> None:
        """
        Invalidate the entity cache.
        
        Call this when entities are added/removed from the database.
        """
        self._entity_cache = {
            'diseases': set(),
            'taxa': set(),
            'methods': set()
        }
        self._cache_loaded = False
        logger.info("Entity validation cache invalidated")


def create_error_response(errors: List[ValidationError]) -> Dict[str, Any]:
    """
    Create a detailed error response from validation errors.
    
    Args:
        errors: List of validation errors
    
    Returns:
        Dictionary with error details suitable for HTTP 400 response
    """
    return {
        "error": "Validation failed",
        "details": [
            {
                "field": error.field,
                "value": str(error.value) if error.value is not None else None,
                "error_type": error.error_type,
                "message": error.message
            }
            for error in errors
        ]
    }

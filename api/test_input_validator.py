"""
api/test_input_validator.py
----------------------------
Unit tests for input validation and sanitization.

This module tests the InputValidator class to ensure it correctly:
- Validates entity names against the database
- Validates numeric thresholds
- Sanitizes inputs to prevent injection attacks
- Returns detailed error messages

**Validates: Requirement 18.2**
"""

import pytest
from unittest.mock import Mock, MagicMock
from api.input_validator import (
    InputValidator,
    ValidationError,
    ValidationResult,
    create_error_response
)


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = Mock()
    session = Mock()
    
    # Mock disease query results
    disease_result = Mock()
    disease_result.__iter__ = Mock(return_value=iter([
        {'name': 'Type 2 Diabetes'},
        {'name': 'IBD'},
        {'name': "Crohn's Disease"}
    ]))
    
    # Mock taxon query results
    taxon_result = Mock()
    taxon_result.__iter__ = Mock(return_value=iter([
        {'name': 'Bacteroides fragilis'},
        {'name': 'Escherichia coli'}
    ]))
    
    # Mock method query results
    method_result = Mock()
    method_result.__iter__ = Mock(return_value=iter([
        {'name': '16S rRNA sequencing'},
        {'name': 'shotgun metagenomics'}
    ]))
    
    # Configure session.run to return appropriate results
    def run_side_effect(query):
        if 'Disease' in query:
            return disease_result
        elif 'Taxon' in query:
            return taxon_result
        elif 'Method' in query:
            return method_result
        return Mock(__iter__=Mock(return_value=iter([])))
    
    session.run = Mock(side_effect=run_side_effect)
    driver.session = Mock(return_value=session)
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    
    return driver


class TestInputValidatorInitialization:
    """Test InputValidator initialization."""
    
    def test_init_without_driver(self):
        """Test initialization without Neo4j driver."""
        validator = InputValidator()
        assert validator.driver is None
        assert not validator._cache_loaded
    
    def test_init_with_driver(self, mock_driver):
        """Test initialization with Neo4j driver."""
        validator = InputValidator(neo4j_driver=mock_driver)
        assert validator.driver is not None
        assert not validator._cache_loaded


class TestStringSanitization:
    """Test string sanitization for injection prevention."""
    
    def test_valid_string(self):
        """Test sanitization of valid string."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("Type 2 Diabetes", "disease")
        
        assert is_valid
        assert sanitized == "Type 2 Diabetes"
        assert error is None
    
    def test_string_with_whitespace(self):
        """Test sanitization trims whitespace."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("  Type 2 Diabetes  ", "disease")
        
        assert is_valid
        assert sanitized == "Type 2 Diabetes"
        assert error is None
    
    def test_empty_string(self):
        """Test empty string is rejected."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("", "disease")
        
        assert not is_valid
        assert sanitized is None
        assert "cannot be empty" in error
    
    def test_cypher_injection_match(self):
        """Test Cypher MATCH keyword is blocked."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("MATCH (n) RETURN n", "disease")
        
        assert not is_valid
        assert "invalid characters or patterns" in error
    
    def test_cypher_injection_where(self):
        """Test Cypher WHERE keyword is blocked."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("disease WHERE 1=1", "disease")
        
        assert not is_valid
        assert "invalid characters or patterns" in error
    
    def test_cypher_injection_delete(self):
        """Test Cypher DELETE keyword is blocked."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("DELETE all nodes", "disease")
        
        assert not is_valid
        assert "invalid characters or patterns" in error
    
    def test_cypher_injection_semicolon(self):
        """Test semicolon is blocked."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("disease; DROP DATABASE", "disease")
        
        assert not is_valid
        assert "invalid characters or patterns" in error
    
    def test_cypher_injection_braces(self):
        """Test braces are blocked."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("disease {prop: 'value'}", "disease")
        
        assert not is_valid
        assert "invalid characters or patterns" in error
    
    def test_string_too_long(self):
        """Test string exceeding max length is rejected."""
        validator = InputValidator()
        long_string = "a" * (validator.MAX_ENTITY_NAME_LENGTH + 1)
        is_valid, sanitized, error = validator.sanitize_string(long_string, "disease")
        
        assert not is_valid
        assert "exceeds maximum length" in error
    
    def test_invalid_characters(self):
        """Test string with invalid characters is rejected."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("disease@#$%", "disease")
        
        assert not is_valid
        assert "invalid characters" in error
    
    def test_valid_characters_with_apostrophe(self):
        """Test string with apostrophe is accepted."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("Crohn's Disease", "disease")
        
        assert is_valid
        assert sanitized == "Crohn's Disease"
    
    def test_valid_characters_with_parentheses(self):
        """Test string with parentheses is accepted."""
        validator = InputValidator()
        is_valid, sanitized, error = validator.sanitize_string("Disease (Type A)", "disease")
        
        assert is_valid
        assert sanitized == "Disease (Type A)"


class TestEntityValidation:
    """Test entity name validation."""
    
    def test_valid_disease_name(self, mock_driver):
        """Test validation of valid disease name."""
        validator = InputValidator(neo4j_driver=mock_driver)
        error = validator.validate_entity_name(
            "Type 2 Diabetes",
            "disease",
            "disease",
            check_existence=True
        )
        
        assert error is None
    
    def test_invalid_disease_name(self, mock_driver):
        """Test validation of non-existent disease name."""
        validator = InputValidator(neo4j_driver=mock_driver)
        error = validator.validate_entity_name(
            "Nonexistent Disease",
            "disease",
            "disease",
            check_existence=True
        )
        
        assert error is not None
        assert error.error_type == "entity_not_found"
        assert "not found in knowledge graph" in error.message
    
    def test_case_insensitive_match(self, mock_driver):
        """Test case-insensitive entity matching."""
        validator = InputValidator(neo4j_driver=mock_driver)
        error = validator.validate_entity_name(
            "type 2 diabetes",  # lowercase
            "disease",
            "disease",
            check_existence=True
        )
        
        # Should match "Type 2 Diabetes" case-insensitively
        assert error is None
    
    def test_validation_without_existence_check(self):
        """Test validation without checking existence."""
        validator = InputValidator()
        error = validator.validate_entity_name(
            "Any Disease",
            "disease",
            "disease",
            check_existence=False
        )
        
        # Should pass sanitization without checking database
        assert error is None
    
    def test_entity_with_injection_attempt(self, mock_driver):
        """Test entity name with injection attempt is rejected."""
        validator = InputValidator(neo4j_driver=mock_driver)
        error = validator.validate_entity_name(
            "Disease MATCH (n)",
            "disease",
            "disease",
            check_existence=True
        )
        
        assert error is not None
        assert error.error_type == "sanitization_failed"


class TestNumericValidation:
    """Test numeric threshold validation."""
    
    def test_valid_integer(self):
        """Test validation of valid integer."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            5,
            "min_papers",
            min_value=1,
            max_value=100
        )
        
        assert error is None
    
    def test_valid_float(self):
        """Test validation of valid float."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            0.75,
            "confidence",
            min_value=0.0,
            max_value=1.0
        )
        
        assert error is None
    
    def test_value_below_minimum(self):
        """Test value below minimum is rejected."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            0,
            "min_papers",
            min_value=1,
            max_value=100
        )
        
        assert error is not None
        assert error.error_type == "out_of_range"
        assert ">= 1" in error.message
    
    def test_value_above_maximum(self):
        """Test value above maximum is rejected."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            150,
            "min_papers",
            min_value=1,
            max_value=100
        )
        
        assert error is not None
        assert error.error_type == "out_of_range"
        assert "<= 100" in error.message
    
    def test_nan_value(self):
        """Test NaN value is rejected."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            float('nan'),
            "confidence",
            min_value=0.0,
            max_value=1.0
        )
        
        assert error is not None
        assert error.error_type == "invalid_value"
        assert "NaN" in error.message
    
    def test_infinity_value(self):
        """Test infinity value is rejected."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            float('inf'),
            "confidence",
            min_value=0.0,
            max_value=1.0
        )
        
        assert error is not None
        assert error.error_type == "invalid_value"
        assert "infinity" in error.message
    
    def test_none_value_not_allowed(self):
        """Test None value is rejected when not allowed."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            None,
            "min_papers",
            min_value=1,
            max_value=100,
            allow_none=False
        )
        
        assert error is not None
        assert error.error_type == "required_field"
    
    def test_none_value_allowed(self):
        """Test None value is accepted when allowed."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            None,
            "optional_field",
            min_value=1,
            max_value=100,
            allow_none=True
        )
        
        assert error is None
    
    def test_invalid_type(self):
        """Test non-numeric value is rejected."""
        validator = InputValidator()
        error = validator.validate_numeric_threshold(
            "not a number",
            "min_papers",
            min_value=1,
            max_value=100
        )
        
        assert error is not None
        assert error.error_type == "invalid_type"


class TestListValidation:
    """Test list validation."""
    
    def test_valid_string_list(self):
        """Test validation of valid string list."""
        validator = InputValidator()
        errors = validator.validate_string_list(
            ["probiotic", "FMT", "diet"],
            "intervention_types"
        )
        
        assert len(errors) == 0
    
    def test_empty_list(self):
        """Test empty list is rejected."""
        validator = InputValidator()
        errors = validator.validate_string_list(
            [],
            "intervention_types"
        )
        
        assert len(errors) == 1
        assert errors[0].error_type == "empty_list"
    
    def test_list_too_large(self):
        """Test list exceeding max size is rejected."""
        validator = InputValidator()
        large_list = ["item"] * (validator.MAX_LIST_SIZE + 1)
        errors = validator.validate_string_list(
            large_list,
            "intervention_types"
        )
        
        assert len(errors) == 1
        assert errors[0].error_type == "list_too_large"
    
    def test_list_with_invalid_item(self):
        """Test list with invalid item is rejected."""
        validator = InputValidator()
        errors = validator.validate_string_list(
            ["valid", "MATCH (n)", "also valid"],
            "intervention_types"
        )
        
        assert len(errors) == 1
        assert "intervention_types[1]" in errors[0].field
    
    def test_list_with_entity_validation(self, mock_driver):
        """Test list validation with entity existence check."""
        validator = InputValidator(neo4j_driver=mock_driver)
        errors = validator.validate_string_list(
            ["16S rRNA sequencing", "shotgun metagenomics"],
            "sequencing_methods",
            entity_type="method",
            check_existence=True
        )
        
        assert len(errors) == 0
    
    def test_list_with_nonexistent_entity(self, mock_driver):
        """Test list with non-existent entity is rejected."""
        validator = InputValidator(neo4j_driver=mock_driver)
        errors = validator.validate_string_list(
            ["16S rRNA sequencing", "nonexistent method"],
            "sequencing_methods",
            entity_type="method",
            check_existence=True
        )
        
        assert len(errors) == 1
        assert "not found in knowledge graph" in errors[0].message


class TestYearRangeValidation:
    """Test year range validation."""
    
    def test_valid_year_range(self):
        """Test validation of valid year range."""
        validator = InputValidator()
        errors = validator.validate_year_range(2020, 2024)
        
        assert len(errors) == 0
    
    def test_invalid_year_range(self):
        """Test year_start > year_end is rejected."""
        validator = InputValidator()
        errors = validator.validate_year_range(2024, 2020)
        
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_range"
        assert "must be <=" in errors[0].message
    
    def test_year_below_minimum(self):
        """Test year below minimum is rejected."""
        validator = InputValidator()
        errors = validator.validate_year_range(1800, 2024)
        
        assert len(errors) >= 1
        assert any(e.error_type == "out_of_range" for e in errors)
    
    def test_year_above_maximum(self):
        """Test year above maximum is rejected."""
        validator = InputValidator()
        errors = validator.validate_year_range(2020, 2200)
        
        assert len(errors) >= 1
        assert any(e.error_type == "out_of_range" for e in errors)
    
    def test_equal_years(self):
        """Test equal start and end years is valid."""
        validator = InputValidator()
        errors = validator.validate_year_range(2024, 2024)
        
        assert len(errors) == 0


class TestCacheManagement:
    """Test entity cache management."""
    
    def test_cache_invalidation(self, mock_driver):
        """Test cache invalidation."""
        validator = InputValidator(neo4j_driver=mock_driver)
        
        # Load cache
        validator._load_entity_cache()
        assert validator._cache_loaded
        assert len(validator._entity_cache['diseases']) > 0
        
        # Invalidate cache
        validator.invalidate_cache()
        assert not validator._cache_loaded
        assert len(validator._entity_cache['diseases']) == 0


class TestErrorResponseCreation:
    """Test error response creation."""
    
    def test_create_error_response_single_error(self):
        """Test creating error response with single error."""
        errors = [
            ValidationError(
                field="disease",
                value="Invalid Disease",
                error_type="entity_not_found",
                message="Disease 'Invalid Disease' not found"
            )
        ]
        
        response = create_error_response(errors)
        
        assert "error" in response
        assert "details" in response
        assert len(response["details"]) == 1
        assert response["details"][0]["field"] == "disease"
        assert response["details"][0]["error_type"] == "entity_not_found"
    
    def test_create_error_response_multiple_errors(self):
        """Test creating error response with multiple errors."""
        errors = [
            ValidationError(
                field="disease",
                value="Invalid Disease",
                error_type="entity_not_found",
                message="Disease not found"
            ),
            ValidationError(
                field="min_papers",
                value=-1,
                error_type="out_of_range",
                message="min_papers must be >= 1"
            )
        ]
        
        response = create_error_response(errors)
        
        assert len(response["details"]) == 2
        assert response["details"][0]["field"] == "disease"
        assert response["details"][1]["field"] == "min_papers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

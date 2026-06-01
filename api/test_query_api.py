"""
api/test_query_api.py
---------------------
Unit tests for the query API endpoints.

This module tests the FastAPI endpoints to ensure they correctly wrap
the ResearchQueryEngine methods and return proper JSON responses.

**Validates: Requirements 1.1, 1.2, 1.3, 13.4a, 18.1, 18.2, 18.3, 18.4, 20.3**
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from api.query_api import app, QueryResult


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_query_engine():
    """Create a mock query engine."""
    mock_engine = Mock()
    
    # Mock successful query result
    mock_result = QueryResult(
        query_description="Test query",
        results=[
            {
                "taxon_name": "Bacteroides fragilis",
                "paper_count": 5,
                "consensus_confidence": 0.85
            }
        ],
        result_count=1,
        execution_time_ms=100.0,
        aggregation_method="test",
        confidence_threshold=0.7,
        timeout=False,
        error=None
    )
    
    # Configure all query methods to return the mock result
    mock_engine.query_cross_study_associations.return_value = mock_result
    mock_engine.query_intervention_evidence.return_value = mock_result
    mock_engine.query_methodology_landscape.return_value = mock_result
    mock_engine.query_top_associations_by_evidence.return_value = mock_result
    mock_engine.query_conflicting_evidence.return_value = mock_result
    mock_engine.get_cache_stats.return_value = {"hits": 10, "misses": 5}
    mock_engine.invalidate_cache.return_value = 15
    
    return mock_engine


class TestRootEndpoints:
    """Test root and health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "documentation" in data
    
    def test_health_check_without_engine(self, client):
        """Test health check fails when engine is not initialized."""
        # The engine won't be initialized in test mode
        response = client.get("/health")
        assert response.status_code == 503


class TestCrossStudyAssociationsEndpoint:
    """Test the cross-study associations endpoint."""
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_valid_request(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test a valid cross-study associations request."""
        mock_engine_global.return_value = mock_query_engine
        
        # Configure mock validator to pass all validations
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        
        # Make the global query_engine and input_validator point to our mocks
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 0.7,
            "require_open_data": True
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query_result"] is not None
        assert data["error"] is None
        
        # Verify the query engine was called with correct parameters
        mock_query_engine.query_cross_study_associations.assert_called_once_with(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7,
            require_open_data=True
        )
    
    def test_invalid_study_type(self, client):
        """Test request with invalid study_type."""
        request_data = {
            "disease": "Type 2 Diabetes",
            "study_type": "invalid_type",
            "min_papers": 3,
            "confidence_threshold": 0.7,
            "require_open_data": True
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_confidence_threshold(self, client):
        """Test request with invalid confidence_threshold."""
        request_data = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 1.5,  # Invalid: > 1.0
            "require_open_data": True
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        assert response.status_code == 422  # Validation error


class TestInterventionEvidenceEndpoint:
    """Test the intervention evidence endpoint."""
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_valid_request(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test a valid intervention evidence request."""
        # Configure mock validator to pass all validations
        mock_validator.validate_string_list.return_value = []
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "intervention_types": ["probiotic", "FMT"],
            "min_sample_size": 50,
            "evidence_strength": "strong"
        }
        
        response = client.post("/query/intervention-evidence", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query_result"] is not None
        
        # Verify the query engine was called
        mock_query_engine.query_intervention_evidence.assert_called_once()
    
    def test_empty_intervention_types(self, client):
        """Test request with empty intervention_types list."""
        request_data = {
            "intervention_types": [],
            "min_sample_size": 50,
            "evidence_strength": "strong"
        }
        
        response = client.post("/query/intervention-evidence", json=request_data)
        assert response.status_code == 422  # Validation error


class TestMethodologyLandscapeEndpoint:
    """Test the methodology landscape endpoint."""
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_valid_request(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test a valid methodology landscape request."""
        # Configure mock validator to pass all validations
        mock_validator.validate_year_range.return_value = []
        mock_validator.validate_string_list.return_value = []
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "year_start": 2020,
            "year_end": 2024,
            "sequencing_methods": ["16S rRNA sequencing", "shotgun metagenomics"],
            "require_deposited_data": True
        }
        
        response = client.post("/query/methodology-landscape", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query_result"] is not None
        
        # Verify the query engine was called
        mock_query_engine.query_methodology_landscape.assert_called_once()
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_invalid_year_range(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test request with invalid year range (start > end)."""
        from api.input_validator import ValidationError
        
        # Configure mock validator to reject invalid year range
        mock_validator.validate_year_range.return_value = [
            ValidationError(
                field="year_range",
                value={"year_start": 2024, "year_end": 2020},
                error_type="invalid_range",
                message="year_start (2024) must be <= year_end (2020)"
            )
        ]
        mock_validator.validate_string_list.return_value = []
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "year_start": 2024,
            "year_end": 2020,  # Invalid: end < start
            "sequencing_methods": ["16S rRNA sequencing"],
            "require_deposited_data": True
        }
        
        response = client.post("/query/methodology-landscape", json=request_data)
        assert response.status_code == 400  # Bad request


class TestTopAssociationsEndpoint:
    """Test the top associations endpoint."""
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_valid_request(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test a valid top associations request."""
        # Configure mock validator to pass all validations
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "disease": "IBD",
            "top_n": 10,
            "min_confidence": 0.7
        }
        
        response = client.post("/query/top-associations", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query_result"] is not None
        
        # Verify the query engine was called
        mock_query_engine.query_top_associations_by_evidence.assert_called_once()


class TestConflictingEvidenceEndpoint:
    """Test the conflicting evidence endpoint."""
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_valid_request(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test a valid conflicting evidence request."""
        # Configure mock validator to pass all validations
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "disease": "Crohn's Disease",
            "min_papers_per_direction": 2
        }
        
        response = client.post("/query/conflicting-evidence", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query_result"] is not None
        
        # Verify the query engine was called
        mock_query_engine.query_conflicting_evidence.assert_called_once()


class TestCacheEndpoints:
    """Test cache management endpoints."""
    
    @patch("api.query_api.query_engine")
    def test_invalidate_cache(self, mock_engine_global, client, mock_query_engine):
        """Test cache invalidation endpoint."""
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        
        response = client.post("/cache/invalidate")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "invalidated_count" in data
        
        # Verify the query engine method was called
        mock_query_engine.invalidate_cache.assert_called_once()
    
    @patch("api.query_api.query_engine")
    def test_get_cache_stats(self, mock_engine_global, client, mock_query_engine):
        """Test cache stats endpoint."""
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        
        response = client.get("/cache/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["cache_enabled"] is True
        assert "stats" in data
        
        # Verify the query engine method was called
        mock_query_engine.get_cache_stats.assert_called_once()


class TestRequestValidation:
    """Test request validation for all endpoints."""
    
    def test_missing_required_field(self, client):
        """Test request with missing required field."""
        from api.rate_limiter import rate_limiter
        rate_limiter.reset()
        
        request_data = {
            # Missing "disease" field
            "study_type": "RCT",
            "min_papers": 3
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_field_type(self, client):
        """Test request with invalid field type."""
        from api.rate_limiter import rate_limiter
        rate_limiter.reset()
        
        request_data = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": "three",  # Should be int
            "confidence_threshold": 0.7
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        assert response.status_code == 422  # Validation error


class TestInputValidationIntegration:
    """Test input validation integration with API endpoints."""
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_cypher_injection_prevention(self, mock_validator, mock_engine, client):
        """Test that Cypher injection attempts are blocked."""
        from api.input_validator import ValidationError
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mock validator to reject injection attempt
        mock_validator.validate_entity_name.return_value = ValidationError(
            field="disease",
            value="MATCH (n) RETURN n",
            error_type="sanitization_failed",
            message="disease contains invalid characters or patterns"
        )
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.input_validator = mock_validator
        api.query_api.query_engine = mock_engine
        
        request_data = {
            "disease": "MATCH (n) RETURN n",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 0.7,
            "require_open_data": True
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        
        # Should return 400 Bad Request with validation error
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "details" in data
        assert len(data["details"]) > 0
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_invalid_numeric_threshold(self, mock_validator, mock_engine, client):
        """Test that invalid numeric thresholds are rejected."""
        from api.input_validator import ValidationError
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mock validator to reject out-of-range value
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.side_effect = [
            None,  # min_papers is valid
            ValidationError(
                field="confidence_threshold",
                value=1.5,
                error_type="out_of_range",
                message="confidence_threshold must be <= 1.0"
            )
        ]
        
        import api.query_api
        api.query_api.input_validator = mock_validator
        api.query_api.query_engine = mock_engine
        
        request_data = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 1.5,  # Invalid: > 1.0
            "require_open_data": True
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        
        # Pydantic catches this first and returns 422
        assert response.status_code == 422
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_nonexistent_entity(self, mock_validator, mock_engine, client):
        """Test that non-existent entities are rejected."""
        from api.input_validator import ValidationError
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mock validator to reject non-existent entity
        mock_validator.validate_entity_name.return_value = ValidationError(
            field="disease",
            value="Nonexistent Disease",
            error_type="entity_not_found",
            message="Disease 'Nonexistent Disease' not found in knowledge graph"
        )
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.input_validator = mock_validator
        api.query_api.query_engine = mock_engine
        
        request_data = {
            "disease": "Nonexistent Disease",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 0.7,
            "require_open_data": True
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "entity_not_found" in str(data["details"])
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_invalid_year_range(self, mock_validator, mock_engine, client):
        """Test that invalid year ranges are rejected."""
        from api.input_validator import ValidationError
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mock validator to reject invalid year range
        mock_validator.validate_year_range.return_value = [
            ValidationError(
                field="year_range",
                value={"year_start": 2024, "year_end": 2020},
                error_type="invalid_range",
                message="year_start (2024) must be <= year_end (2020)"
            )
        ]
        mock_validator.validate_string_list.return_value = []
        
        import api.query_api
        api.query_api.input_validator = mock_validator
        api.query_api.query_engine = mock_engine
        
        request_data = {
            "year_start": 2024,
            "year_end": 2020,
            "sequencing_methods": ["16S rRNA sequencing"],
            "require_deposited_data": True
        }
        
        response = client.post("/query/methodology-landscape", json=request_data)
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "invalid_range" in str(data["details"])
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_empty_list_validation(self, mock_validator, mock_engine, client):
        """Test that empty lists are rejected."""
        from api.input_validator import ValidationError
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mock validator to reject empty list
        mock_validator.validate_string_list.return_value = [
            ValidationError(
                field="intervention_types",
                value=[],
                error_type="empty_list",
                message="intervention_types cannot be empty"
            )
        ]
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.input_validator = mock_validator
        api.query_api.query_engine = mock_engine
        
        request_data = {
            "intervention_types": [],
            "min_sample_size": 50,
            "evidence_strength": "strong"
        }
        
        response = client.post("/query/intervention-evidence", json=request_data)
        
        # Pydantic catches this first and returns 422
        assert response.status_code == 422


class TestQueryTimeout:
    """Test query timeout handling in API endpoints.
    
    **Validates: Requirement 13.4a**
    """
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_query_timeout_returns_partial_results(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test that when a query times out, API returns partial results with timeout=True."""
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mock validator to pass all validations
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        
        # Create a timeout result with partial results
        timeout_result = QueryResult(
            query_description="Cross-study associations query (timed out)",
            results=[
                {
                    "taxon_name": "Bacteroides fragilis",
                    "paper_count": 3,
                    "consensus_confidence": 0.75
                },
                {
                    "taxon_name": "Faecalibacterium prausnitzii",
                    "paper_count": 2,
                    "consensus_confidence": 0.68
                }
            ],
            result_count=2,
            execution_time_ms=30500.0,  # Exceeded 30 second timeout
            aggregation_method="weighted_average",
            confidence_threshold=0.7,
            timeout=True,  # Timeout flag set
            error=None
        )
        
        # Configure mock engine to return timeout result
        mock_query_engine.query_cross_study_associations.return_value = timeout_result
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 0.7,
            "require_open_data": True
        }
        
        response = client.post("/query/cross-study-associations", json=request_data)
        
        # Should return 200 OK (not an error status)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["query_result"] is not None
        
        # Verify timeout flag is set
        query_result = data["query_result"]
        assert query_result["timeout"] is True
        
        # Verify partial results are returned
        assert query_result["result_count"] == 2
        assert len(query_result["results"]) == 2
        
        # Verify execution time shows timeout occurred
        assert query_result["execution_time_ms"] > 30000
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_timeout_flag_properly_set_in_response(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test that timeout flag is properly included in QueryResult."""
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mock validator
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        
        # Create timeout result
        timeout_result = QueryResult(
            query_description="Top associations query (timed out)",
            results=[{"taxon_name": "Bacteroides", "score": 0.8}],
            result_count=1,
            execution_time_ms=31000.0,
            timeout=True,
            error=None
        )
        
        mock_query_engine.query_top_associations_by_evidence.return_value = timeout_result
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "disease": "IBD",
            "top_n": 10,
            "min_confidence": 0.7
        }
        
        response = client.post("/query/top-associations", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        query_result = data["query_result"]
        
        # Verify timeout field exists and is True
        assert "timeout" in query_result
        assert query_result["timeout"] is True
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_timeout_returns_200_not_error(self, mock_validator, mock_engine_global, client, mock_query_engine):
        """Test that timeout with partial results returns 200 OK, not an error status."""
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mock validator
        mock_validator.validate_string_list.return_value = []
        mock_validator.validate_numeric_threshold.return_value = None
        
        # Create timeout result with partial results
        timeout_result = QueryResult(
            query_description="Intervention evidence query (timed out)",
            results=[
                {"intervention": "probiotic", "evidence_strength": "moderate"}
            ],
            result_count=1,
            execution_time_ms=32000.0,
            timeout=True,
            error=None
        )
        
        mock_query_engine.query_intervention_evidence.return_value = timeout_result
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "intervention_types": ["probiotic", "FMT"],
            "min_sample_size": 50,
            "evidence_strength": "strong"
        }
        
        response = client.post("/query/intervention-evidence", json=request_data)
        
        # Should return 200 OK, not 500 or other error
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["error"] is None
        
        # Timeout is indicated in the query_result, not as an API error
        assert data["query_result"]["timeout"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestRateLimitingIntegration:
    """Test rate limiting integration with API endpoints."""
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_rate_limit_allows_requests_within_limit(self, mock_validator, mock_engine, client, mock_query_engine):
        """Test that requests within rate limit are allowed."""
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mocks
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 0.7,
            "require_open_data": True
        }
        
        # Make 5 requests (well within the 10 req/min limit)
        for i in range(5):
            response = client.post("/query/cross-study-associations", json=request_data)
            assert response.status_code == 200, f"Request {i+1} failed"
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_rate_limit_blocks_excessive_requests(self, mock_validator, mock_engine, client, mock_query_engine):
        """Test that requests exceeding rate limit are blocked with 429."""
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mocks
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 0.7,
            "require_open_data": True
        }
        
        # Make 10 requests (the limit)
        for i in range(10):
            response = client.post("/query/cross-study-associations", json=request_data)
            assert response.status_code == 200, f"Request {i+1} should succeed"
        
        # 11th request should be blocked
        response = client.post("/query/cross-study-associations", json=request_data)
        assert response.status_code == 429
        
        # Check response contains rate limit information
        data = response.json()
        assert "error" in str(data).lower() or "rate limit" in str(data).lower()
        
        # Check Retry-After header is present
        assert "retry-after" in response.headers
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_rate_limit_applies_to_all_endpoints(self, mock_validator, mock_engine, client, mock_query_engine):
        """Test that rate limit applies across all query endpoints."""
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mocks
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        mock_validator.validate_string_list.return_value = []
        mock_validator.validate_year_range.return_value = []
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        # Make requests to different endpoints
        endpoints_and_data = [
            ("/query/cross-study-associations", {
                "disease": "Type 2 Diabetes",
                "study_type": "RCT",
                "min_papers": 3,
                "confidence_threshold": 0.7,
                "require_open_data": True
            }),
            ("/query/top-associations", {
                "disease": "IBD",
                "top_n": 10,
                "min_confidence": 0.7
            }),
            ("/query/conflicting-evidence", {
                "disease": "Crohn's Disease",
                "min_papers_per_direction": 2
            }),
        ]
        
        # Make 10 requests across different endpoints
        for i in range(10):
            endpoint, data = endpoints_and_data[i % len(endpoints_and_data)]
            response = client.post(endpoint, json=data)
            assert response.status_code == 200, f"Request {i+1} to {endpoint} should succeed"
        
        # 11th request should be blocked regardless of endpoint
        endpoint, data = endpoints_and_data[0]
        response = client.post(endpoint, json=data)
        assert response.status_code == 429


class TestQueryComplexityLimitIntegration:
    """Test query complexity limit integration with API endpoints."""
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_top_n_capped_at_1000(self, mock_validator, mock_engine, client, mock_query_engine):
        """Test that top_n parameter is capped at 1000."""
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        # Configure mocks
        mock_validator.validate_entity_name.return_value = None
        mock_validator.validate_numeric_threshold.return_value = None
        
        import api.query_api
        api.query_api.query_engine = mock_query_engine
        api.query_api.input_validator = mock_validator
        
        request_data = {
            "disease": "IBD",
            "top_n": 1000,  # At the limit
            "min_confidence": 0.7
        }
        
        response = client.post("/query/top-associations", json=request_data)
        assert response.status_code == 200
        
        # Verify query engine was called with capped value
        mock_query_engine.query_top_associations_by_evidence.assert_called_once()
        call_args = mock_query_engine.query_top_associations_by_evidence.call_args
        assert call_args.kwargs["top_n"] == 1000
    
    @patch("api.query_api.query_engine")
    @patch("api.query_api.input_validator")
    def test_top_n_exceeding_1000_rejected_by_pydantic(self, mock_validator, mock_engine, client, mock_query_engine):
        """Test that top_n > 1000 is rejected by Pydantic validation."""
        from api.rate_limiter import rate_limiter
        
        # Reset rate limiter
        rate_limiter.reset()
        
        request_data = {
            "disease": "IBD",
            "top_n": 5000,  # Exceeds limit
            "min_confidence": 0.7
        }
        
        response = client.post("/query/top-associations", json=request_data)
        
        # Should be rejected by Pydantic before reaching the endpoint
        assert response.status_code == 422


class TestLimitsEndpoint:
    """Test the /limits endpoint."""
    
    def test_get_limits(self, client):
        """Test that /limits endpoint returns correct information."""
        response = client.get("/limits")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "rate_limiting" in data
        assert "query_complexity" in data
        
        # Check rate limiting info
        rate_limit = data["rate_limiting"]
        assert rate_limit["max_requests_per_window"] == 10
        assert rate_limit["window_seconds"] == 60
        
        # Check query complexity info
        complexity = data["query_complexity"]
        assert complexity["max_result_count"] == 1000
        assert complexity["max_query_depth"] == 5

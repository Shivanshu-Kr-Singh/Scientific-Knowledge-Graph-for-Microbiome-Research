"""
api/test_query_complexity_limiter.py
-------------------------------------
Unit tests for the query complexity limiter.

This module tests the query complexity limiting functionality to ensure it
correctly enforces max result count and query depth limits.

**Validates: Requirement 18.3**
"""

import pytest
from api.query_complexity_limiter import QueryComplexityLimiter


@pytest.fixture
def limiter():
    """Create a query complexity limiter with default settings."""
    return QueryComplexityLimiter()


class TestQueryComplexityLimiter:
    """Test query complexity limiter functionality."""
    
    def test_initialization(self, limiter):
        """Test limiter initializes with correct parameters."""
        assert limiter.max_result_count == 1000
        assert limiter.max_query_depth == 5
    
    def test_validate_result_count_within_limit(self, limiter):
        """Test that result counts within limit are not modified."""
        result = limiter.validate_result_count_limit(100)
        assert result == 100
    
    def test_validate_result_count_at_limit(self, limiter):
        """Test that result count at limit is allowed."""
        result = limiter.validate_result_count_limit(1000)
        assert result == 1000
    
    def test_validate_result_count_exceeds_limit(self, limiter):
        """Test that result counts exceeding limit are capped."""
        result = limiter.validate_result_count_limit(5000)
        assert result == 1000  # Capped to max
    
    def test_validate_result_count_none(self, limiter):
        """Test that None returns the maximum limit."""
        result = limiter.validate_result_count_limit(None)
        assert result == 1000
    
    def test_apply_result_limit_to_cypher_without_limit(self, limiter):
        """Test adding LIMIT clause to query without one."""
        query = "MATCH (n:Paper) RETURN n"
        limited_query = limiter.apply_result_limit_to_cypher(query)
        
        assert "LIMIT 1000" in limited_query
        assert limited_query == "MATCH (n:Paper) RETURN n LIMIT 1000"
    
    def test_apply_result_limit_to_cypher_with_existing_limit(self, limiter):
        """Test that queries with LIMIT are not modified."""
        query = "MATCH (n:Paper) RETURN n LIMIT 50"
        limited_query = limiter.apply_result_limit_to_cypher(query)
        
        # Should not add another LIMIT
        assert limited_query == query
    
    def test_apply_result_limit_case_insensitive(self, limiter):
        """Test that LIMIT detection is case-insensitive."""
        query = "MATCH (n:Paper) RETURN n limit 50"
        limited_query = limiter.apply_result_limit_to_cypher(query)
        
        # Should not add another LIMIT
        assert limited_query == query
    
    def test_validate_query_depth_no_depth_param(self, limiter):
        """Test that queries without depth parameters pass validation."""
        query_params = {"disease": "Type 2 Diabetes", "min_papers": 3}
        
        # Should not raise exception
        limiter.validate_query_depth(query_params)
    
    def test_validate_query_depth_within_limit(self, limiter):
        """Test that query depth within limit passes validation."""
        query_params = {"max_hops": 3}
        
        # Should not raise exception
        limiter.validate_query_depth(query_params)
    
    def test_validate_query_depth_at_limit(self, limiter):
        """Test that query depth at limit passes validation."""
        query_params = {"max_hops": 5}
        
        # Should not raise exception
        limiter.validate_query_depth(query_params)
    
    def test_validate_query_depth_exceeds_limit(self, limiter):
        """Test that query depth exceeding limit raises error."""
        query_params = {"max_hops": 10}
        
        with pytest.raises(ValueError) as exc_info:
            limiter.validate_query_depth(query_params)
        
        assert "exceeds maximum allowed depth" in str(exc_info.value)
    
    def test_limit_result_set_within_limit(self, limiter):
        """Test that result sets within limit are not modified."""
        results = [{"id": i} for i in range(100)]
        limited_results = limiter.limit_result_set(results, "test_query")
        
        assert len(limited_results) == 100
        assert limited_results == results
    
    def test_limit_result_set_at_limit(self, limiter):
        """Test that result sets at limit are not modified."""
        results = [{"id": i} for i in range(1000)]
        limited_results = limiter.limit_result_set(results, "test_query")
        
        assert len(limited_results) == 1000
        assert limited_results == results
    
    def test_limit_result_set_exceeds_limit(self, limiter):
        """Test that result sets exceeding limit are truncated."""
        results = [{"id": i} for i in range(2000)]
        limited_results = limiter.limit_result_set(results, "test_query")
        
        assert len(limited_results) == 1000
        assert limited_results == results[:1000]
    
    def test_get_limits(self, limiter):
        """Test that get_limits returns correct values."""
        limits = limiter.get_limits()
        
        assert limits["max_result_count"] == 1000
        assert limits["max_query_depth"] == 5
    
    def test_custom_limits(self):
        """Test creating limiter with custom limits."""
        custom_limiter = QueryComplexityLimiter(
            max_result_count=500,
            max_query_depth=3
        )
        
        assert custom_limiter.max_result_count == 500
        assert custom_limiter.max_query_depth == 3
        
        # Test that custom limits are enforced
        result = custom_limiter.validate_result_count_limit(1000)
        assert result == 500  # Capped to custom max


class TestQueryComplexityLimiterEdgeCases:
    """Test edge cases for query complexity limiter."""
    
    def test_zero_result_count(self, limiter):
        """Test handling of zero result count."""
        result = limiter.validate_result_count_limit(0)
        assert result == 0  # Should allow 0 (empty result set)
    
    def test_negative_result_count(self, limiter):
        """Test handling of negative result count."""
        # Negative values should be allowed through (will be caught by Pydantic)
        result = limiter.validate_result_count_limit(-1)
        assert result == -1
    
    def test_very_large_result_count(self, limiter):
        """Test handling of very large result count."""
        result = limiter.validate_result_count_limit(1_000_000)
        assert result == 1000  # Capped to max
    
    def test_empty_cypher_query(self, limiter):
        """Test handling of empty Cypher query."""
        query = ""
        limited_query = limiter.apply_result_limit_to_cypher(query)
        
        assert "LIMIT 1000" in limited_query
    
    def test_cypher_query_with_trailing_whitespace(self, limiter):
        """Test handling of query with trailing whitespace."""
        query = "MATCH (n:Paper) RETURN n   \n  "
        limited_query = limiter.apply_result_limit_to_cypher(query)
        
        assert "LIMIT 1000" in limited_query
        # Should strip trailing whitespace before adding LIMIT
        assert limited_query.endswith("LIMIT 1000")
    
    def test_empty_result_set(self, limiter):
        """Test handling of empty result set."""
        results = []
        limited_results = limiter.limit_result_set(results, "test_query")
        
        assert len(limited_results) == 0
        assert limited_results == []
    
    def test_single_result(self, limiter):
        """Test handling of single result."""
        results = [{"id": 1}]
        limited_results = limiter.limit_result_set(results, "test_query")
        
        assert len(limited_results) == 1
        assert limited_results == results


class TestQueryComplexityLimiterIntegration:
    """Integration tests for query complexity limiter."""
    
    def test_typical_query_workflow(self, limiter):
        """Test typical workflow of validating and limiting a query."""
        # 1. Validate result count parameter
        requested_top_n = 50
        limited_top_n = limiter.validate_result_count_limit(requested_top_n)
        assert limited_top_n == 50
        
        # 2. Validate query depth
        query_params = {"disease": "Type 2 Diabetes", "top_n": limited_top_n}
        limiter.validate_query_depth(query_params)
        
        # 3. Apply limit to Cypher query
        cypher_query = "MATCH (n:Taxon)-[r:ASSOCIATED_WITH]->(d:Disease) RETURN n"
        limited_query = limiter.apply_result_limit_to_cypher(cypher_query)
        assert "LIMIT 1000" in limited_query
        
        # 4. Limit result set (safety net)
        results = [{"taxon": f"Taxon_{i}"} for i in range(50)]
        limited_results = limiter.limit_result_set(results, "top_associations")
        assert len(limited_results) == 50
    
    def test_excessive_request_workflow(self, limiter):
        """Test workflow with excessive request parameters."""
        # 1. Request excessive result count
        requested_top_n = 5000
        limited_top_n = limiter.validate_result_count_limit(requested_top_n)
        assert limited_top_n == 1000  # Capped
        
        # 2. Simulate query returning too many results
        results = [{"taxon": f"Taxon_{i}"} for i in range(2000)]
        limited_results = limiter.limit_result_set(results, "top_associations")
        assert len(limited_results) == 1000  # Truncated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

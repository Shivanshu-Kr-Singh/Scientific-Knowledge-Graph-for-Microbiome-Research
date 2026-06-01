"""
api/test_rate_limiter.py
-------------------------
Unit tests for the rate limiter.

This module tests the rate limiting functionality to ensure it correctly
enforces the 10 queries per minute per user limit.

**Validates: Requirement 18.4**
"""

import pytest
import time
from unittest.mock import Mock
from fastapi import HTTPException, Request
from api.rate_limiter import RateLimiter


@pytest.fixture
def rate_limiter():
    """Create a rate limiter with short window for testing."""
    # Use 5 requests per 2 seconds for faster testing
    return RateLimiter(max_requests=5, window_seconds=2)


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = Mock(spec=Request)
    request.client = Mock()
    request.client.host = "127.0.0.1"
    request.headers = {}
    return request


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    def test_initialization(self, rate_limiter):
        """Test rate limiter initializes with correct parameters."""
        assert rate_limiter.max_requests == 5
        assert rate_limiter.window_seconds == 2
    
    def test_first_request_allowed(self, rate_limiter, mock_request):
        """Test that the first request is always allowed."""
        is_allowed, remaining = rate_limiter.check_rate_limit(mock_request)
        
        assert is_allowed is True
        assert remaining == 4  # 5 max - 1 used = 4 remaining
    
    def test_requests_within_limit(self, rate_limiter, mock_request):
        """Test that requests within the limit are allowed."""
        # Make 5 requests (the limit)
        for i in range(5):
            is_allowed, remaining = rate_limiter.check_rate_limit(mock_request)
            assert is_allowed is True
            assert remaining == 4 - i
    
    def test_request_exceeds_limit(self, rate_limiter, mock_request):
        """Test that requests exceeding the limit are blocked."""
        # Make 5 requests (the limit)
        for _ in range(5):
            is_allowed, _ = rate_limiter.check_rate_limit(mock_request)
            assert is_allowed is True
        
        # 6th request should be blocked
        is_allowed, remaining = rate_limiter.check_rate_limit(mock_request)
        assert is_allowed is False
        assert remaining == 0
    
    def test_rate_limit_resets_after_window(self, rate_limiter, mock_request):
        """Test that rate limit resets after the time window expires."""
        # Make 5 requests (the limit)
        for _ in range(5):
            is_allowed, _ = rate_limiter.check_rate_limit(mock_request)
            assert is_allowed is True
        
        # 6th request should be blocked
        is_allowed, _ = rate_limiter.check_rate_limit(mock_request)
        assert is_allowed is False
        
        # Wait for window to expire (2 seconds + small buffer)
        time.sleep(2.1)
        
        # Request should now be allowed
        is_allowed, remaining = rate_limiter.check_rate_limit(mock_request)
        assert is_allowed is True
        assert remaining == 4
    
    def test_different_clients_tracked_separately(self, rate_limiter):
        """Test that different clients have separate rate limits."""
        # Create two different clients
        request1 = Mock(spec=Request)
        request1.client = Mock()
        request1.client.host = "127.0.0.1"
        request1.headers = {}
        
        request2 = Mock(spec=Request)
        request2.client = Mock()
        request2.client.host = "192.168.1.1"
        request2.headers = {}
        
        # Make 5 requests from client 1 (the limit)
        for _ in range(5):
            is_allowed, _ = rate_limiter.check_rate_limit(request1)
            assert is_allowed is True
        
        # Client 1 should be blocked
        is_allowed, _ = rate_limiter.check_rate_limit(request1)
        assert is_allowed is False
        
        # Client 2 should still be allowed
        is_allowed, remaining = rate_limiter.check_rate_limit(request2)
        assert is_allowed is True
        assert remaining == 4
    
    def test_x_forwarded_for_header(self, rate_limiter):
        """Test that X-Forwarded-For header is used when present."""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}
        
        # Should use the first IP from X-Forwarded-For
        client_id = rate_limiter._get_client_id(request)
        assert client_id == "203.0.113.1"
    
    def test_middleware_callable_allows_request(self, rate_limiter, mock_request):
        """Test that middleware allows requests within limit."""
        import asyncio
        
        # Should not raise exception
        asyncio.run(rate_limiter(mock_request))
    
    def test_middleware_callable_blocks_request(self, rate_limiter, mock_request):
        """Test that middleware blocks requests exceeding limit."""
        import asyncio
        
        # Make 5 requests (the limit)
        for _ in range(5):
            asyncio.run(rate_limiter(mock_request))
        
        # 6th request should raise HTTPException
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(rate_limiter(mock_request))
        
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)
        assert "Retry-After" in exc_info.value.headers
    
    def test_get_stats(self, rate_limiter, mock_request):
        """Test that get_stats returns correct statistics."""
        # Make some requests
        for _ in range(3):
            rate_limiter.check_rate_limit(mock_request)
        
        stats = rate_limiter.get_stats()
        
        assert stats["total_clients"] == 1
        assert stats["total_requests_in_window"] == 3
        assert stats["max_requests_per_window"] == 5
        assert stats["window_seconds"] == 2
    
    def test_reset(self, rate_limiter, mock_request):
        """Test that reset clears all counters."""
        # Make some requests
        for _ in range(3):
            rate_limiter.check_rate_limit(mock_request)
        
        # Reset
        rate_limiter.reset()
        
        # Stats should show no requests
        stats = rate_limiter.get_stats()
        assert stats["total_clients"] == 0
        assert stats["total_requests_in_window"] == 0
    
    def test_sliding_window_behavior(self, rate_limiter, mock_request):
        """Test that sliding window correctly expires old requests."""
        # Make 3 requests
        for _ in range(3):
            rate_limiter.check_rate_limit(mock_request)
        
        # Wait 1 second (half the window)
        time.sleep(1.0)
        
        # Make 2 more requests (total 5, at the limit)
        for _ in range(2):
            is_allowed, _ = rate_limiter.check_rate_limit(mock_request)
            assert is_allowed is True
        
        # Next request should be blocked
        is_allowed, _ = rate_limiter.check_rate_limit(mock_request)
        assert is_allowed is False
        
        # Wait another 1.1 seconds (first 3 requests should expire)
        time.sleep(1.1)
        
        # Should be able to make 3 more requests
        for i in range(3):
            is_allowed, _ = rate_limiter.check_rate_limit(mock_request)
            assert is_allowed is True


class TestRateLimiterProduction:
    """Test rate limiter with production settings."""
    
    def test_production_limits(self):
        """Test rate limiter with production settings (10 req/min)."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {}
        
        # Make 10 requests (the limit)
        for i in range(10):
            is_allowed, remaining = limiter.check_rate_limit(request)
            assert is_allowed is True
            assert remaining == 9 - i
        
        # 11th request should be blocked
        is_allowed, remaining = limiter.check_rate_limit(request)
        assert is_allowed is False
        assert remaining == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

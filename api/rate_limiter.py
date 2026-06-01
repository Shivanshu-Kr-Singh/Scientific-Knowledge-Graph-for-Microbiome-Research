"""
api/rate_limiter.py
-------------------
Rate limiting middleware for the query API.

This module implements rate limiting to prevent abuse and ensure fair resource
allocation across users. It tracks requests per user/IP and enforces limits.

**Validates: Requirement 18.4**

Features:
- 10 queries per minute per user/IP
- Returns 429 Too Many Requests when limit exceeded
- Sliding window rate limiting
- Thread-safe implementation
"""

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, Deque, Tuple
from fastapi import Request, HTTPException, status
from loguru import logger


class RateLimiter:
    """
    Rate limiter using sliding window algorithm.
    
    Tracks request timestamps per client and enforces rate limits.
    
    **Validates: Requirement 18.4**
    """
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Store request timestamps per client
        # Key: client_id (IP or user ID), Value: deque of timestamps
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        
        # Lock for thread-safe access
        self._lock = Lock()
        
        logger.info(
            f"Rate limiter initialized: {max_requests} requests per "
            f"{window_seconds} seconds"
        )
    
    def _get_client_id(self, request: Request) -> str:
        """
        Extract client identifier from request.
        
        Uses X-Forwarded-For header if available (for proxied requests),
        otherwise falls back to client IP.
        
        Args:
            request: FastAPI request object
        
        Returns:
            Client identifier string
        """
        # Check for X-Forwarded-For header (common in proxied environments)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, use the first one
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Fall back to direct client IP
            client_ip = request.client.host if request.client else "unknown"
        
        return client_ip
    
    def _clean_old_requests(self, timestamps: Deque[float], current_time: float):
        """
        Remove timestamps outside the current window.
        
        Args:
            timestamps: Deque of request timestamps
            current_time: Current timestamp
        """
        cutoff_time = current_time - self.window_seconds
        
        # Remove old timestamps from the left (oldest)
        while timestamps and timestamps[0] < cutoff_time:
            timestamps.popleft()
    
    def check_rate_limit(self, request: Request) -> Tuple[bool, int]:
        """
        Check if request is within rate limit.
        
        Args:
            request: FastAPI request object
        
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        client_id = self._get_client_id(request)
        current_time = time.time()
        
        with self._lock:
            # Get request history for this client
            timestamps = self._requests[client_id]
            
            # Clean up old requests outside the window
            self._clean_old_requests(timestamps, current_time)
            
            # Check if limit is exceeded
            request_count = len(timestamps)
            
            if request_count >= self.max_requests:
                # Rate limit exceeded
                logger.warning(
                    f"Rate limit exceeded for client {client_id}: "
                    f"{request_count} requests in last {self.window_seconds}s"
                )
                return False, 0
            
            # Add current request timestamp
            timestamps.append(current_time)
            
            remaining = self.max_requests - (request_count + 1)
            
            logger.debug(
                f"Rate limit check passed for {client_id}: "
                f"{request_count + 1}/{self.max_requests} requests, "
                f"{remaining} remaining"
            )
            
            return True, remaining
    
    async def __call__(self, request: Request):
        """
        Middleware callable for FastAPI.
        
        Args:
            request: FastAPI request object
        
        Raises:
            HTTPException: 429 Too Many Requests if rate limit exceeded
        """
        is_allowed, remaining = self.check_rate_limit(request)
        
        if not is_allowed:
            # Calculate retry-after time
            client_id = self._get_client_id(request)
            timestamps = self._requests[client_id]
            
            if timestamps:
                # Time until oldest request expires
                oldest_timestamp = timestamps[0]
                retry_after = int(self.window_seconds - (time.time() - oldest_timestamp)) + 1
            else:
                retry_after = self.window_seconds
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.max_requests} requests per {self.window_seconds} seconds",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with stats (total_clients, total_requests)
        """
        with self._lock:
            total_clients = len(self._requests)
            total_requests = sum(len(timestamps) for timestamps in self._requests.values())
            
            return {
                "total_clients": total_clients,
                "total_requests_in_window": total_requests,
                "max_requests_per_window": self.max_requests,
                "window_seconds": self.window_seconds
            }
    
    def reset(self):
        """
        Reset all rate limit counters.
        
        Useful for testing or administrative purposes.
        """
        with self._lock:
            self._requests.clear()
            logger.info("Rate limiter reset: all counters cleared")


# Global rate limiter instance
# 10 queries per minute per user (Requirement 18.4)
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

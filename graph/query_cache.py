"""
graph/query_cache.py
--------------------
Query result caching layer for ResearchQueryEngine

This module provides a caching layer for research queries with:
- 24-hour TTL for cached results
- Query parameters as cache keys
- Cache invalidation when new data is loaded
- Thread-safe operations

Requirements: 13.5
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import threading


class QueryCache:
    """
    Thread-safe cache for query results with TTL support.
    
    This cache stores query results keyed by query parameters to avoid
    redundant database queries for common research questions.
    
    **Validates: Requirement 13.5**
    
    Features:
    - 24-hour TTL for cached entries
    - Automatic expiration of stale entries
    - Thread-safe operations using locks
    - Cache invalidation support
    - Query parameter-based cache keys
    
    Usage:
        cache = QueryCache(ttl_hours=24)
        
        # Try to get cached result
        result = cache.get("query_cross_study_associations", {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3
        })
        
        if result is None:
            # Cache miss - execute query
            result = execute_query(...)
            cache.set("query_cross_study_associations", params, result)
        
        # Invalidate cache when new data is loaded
        cache.invalidate_all()
    """
    
    def __init__(self, ttl_hours: int = 24):
        """
        Initialize the query cache.
        
        Args:
            ttl_hours: Time-to-live for cached entries in hours (default: 24)
        
        Preconditions:
        - ttl_hours > 0
        
        Postconditions:
        - Cache is initialized with empty storage
        - TTL is set to ttl_hours
        """
        if ttl_hours <= 0:
            raise ValueError("ttl_hours must be positive")
        
        self.ttl_hours = ttl_hours
        self.ttl_seconds = ttl_hours * 3600
        
        # Cache storage: {cache_key: (result, timestamp)}
        self._cache: Dict[str, Tuple[Any, float]] = {}
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
    
    def _generate_cache_key(self, query_name: str, parameters: Dict[str, Any]) -> str:
        """
        Generate a cache key from query name and parameters.
        
        The cache key is a hash of the query name and sorted parameters
        to ensure consistent keys for identical queries.
        
        Args:
            query_name: Name of the query method (e.g., "query_cross_study_associations")
            parameters: Dictionary of query parameters
        
        Returns:
            SHA256 hash string as cache key
        
        Preconditions:
        - query_name is non-empty string
        - parameters is a dictionary
        
        Postconditions:
        - Returns consistent hash for same query_name and parameters
        - Different parameters produce different hashes
        
        Example:
            key1 = cache._generate_cache_key("query_cross_study_associations", {
                "disease": "Type 2 Diabetes",
                "study_type": "RCT"
            })
            key2 = cache._generate_cache_key("query_cross_study_associations", {
                "study_type": "RCT",
                "disease": "Type 2 Diabetes"
            })
            # key1 == key2 (order doesn't matter)
        """
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(parameters, sort_keys=True)
        
        # Create cache key from query name and parameters
        cache_input = f"{query_name}:{sorted_params}"
        
        # Generate SHA256 hash
        cache_key = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
        
        return cache_key
    
    def _is_expired(self, timestamp: float) -> bool:
        """
        Check if a cache entry has expired based on TTL.
        
        Args:
            timestamp: Unix timestamp when entry was cached
        
        Returns:
            True if entry has expired, False otherwise
        
        Preconditions:
        - timestamp is a valid Unix timestamp
        
        Postconditions:
        - Returns True if current_time - timestamp > ttl_seconds
        - Returns False otherwise
        """
        current_time = time.time()
        age_seconds = current_time - timestamp
        return age_seconds > self.ttl_seconds
    
    def get(self, query_name: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """
        Retrieve a cached query result if available and not expired.
        
        **Validates: Requirement 13.5 (caching with 24-hour TTL)**
        
        Args:
            query_name: Name of the query method
            parameters: Dictionary of query parameters
        
        Returns:
            Cached result if available and not expired, None otherwise
        
        Preconditions:
        - query_name is non-empty string
        - parameters is a dictionary
        
        Postconditions:
        - Returns cached result if found and not expired
        - Returns None if cache miss or expired entry
        - Expired entries are automatically removed
        - Increments hit or miss counter
        
        Example:
            result = cache.get("query_cross_study_associations", {
                "disease": "Type 2 Diabetes",
                "study_type": "RCT",
                "min_papers": 3,
                "confidence_threshold": 0.7
            })
            
            if result is not None:
                # Cache hit - use cached result
                return result
            else:
                # Cache miss - execute query
                result = execute_query(...)
                cache.set(query_name, parameters, result)
        """
        cache_key = self._generate_cache_key(query_name, parameters)
        
        with self._lock:
            if cache_key not in self._cache:
                # Cache miss
                self._misses += 1
                return None
            
            result, timestamp = self._cache[cache_key]
            
            # Check if expired
            if self._is_expired(timestamp):
                # Remove expired entry
                del self._cache[cache_key]
                self._misses += 1
                return None
            
            # Cache hit
            self._hits += 1
            return result
    
    def set(self, query_name: str, parameters: Dict[str, Any], result: Any) -> None:
        """
        Store a query result in the cache.
        
        **Validates: Requirement 13.5 (caching with query parameters as key)**
        
        Args:
            query_name: Name of the query method
            parameters: Dictionary of query parameters
            result: Query result to cache
        
        Preconditions:
        - query_name is non-empty string
        - parameters is a dictionary
        - result is not None
        
        Postconditions:
        - Result is stored in cache with current timestamp
        - Cache key is generated from query_name and parameters
        - Entry will expire after ttl_hours
        
        Example:
            result = execute_query(...)
            cache.set("query_cross_study_associations", {
                "disease": "Type 2 Diabetes",
                "study_type": "RCT",
                "min_papers": 3
            }, result)
        """
        if result is None:
            # Don't cache None results
            return
        
        cache_key = self._generate_cache_key(query_name, parameters)
        current_time = time.time()
        
        with self._lock:
            self._cache[cache_key] = (result, current_time)
    
    def invalidate_all(self) -> int:
        """
        Invalidate all cached entries.
        
        This should be called when new data is loaded into the knowledge graph
        to ensure queries return fresh results.
        
        **Validates: Requirement 13.5 (cache invalidation when new data is loaded)**
        
        Returns:
            Number of entries that were invalidated
        
        Postconditions:
        - All cache entries are removed
        - Returns count of removed entries
        - Increments invalidation counter
        
        Example:
            # After loading new papers into the graph
            load_papers_to_graph(new_papers)
            
            # Invalidate cache to ensure fresh results
            count = cache.invalidate_all()
            print(f"Invalidated {count} cached queries")
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._invalidations += 1
            return count
    
    def invalidate_query(self, query_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Invalidate a specific cached query.
        
        Args:
            query_name: Name of the query method
            parameters: Dictionary of query parameters
        
        Returns:
            True if entry was found and removed, False otherwise
        
        Postconditions:
        - Specified cache entry is removed if it exists
        - Returns True if entry was found, False otherwise
        
        Example:
            # Invalidate specific query
            cache.invalidate_query("query_cross_study_associations", {
                "disease": "Type 2 Diabetes",
                "study_type": "RCT"
            })
        """
        cache_key = self._generate_cache_key(query_name, parameters)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        This method can be called periodically to free memory from expired entries.
        
        Returns:
            Number of expired entries that were removed
        
        Postconditions:
        - All expired entries are removed
        - Returns count of removed entries
        
        Example:
            # Periodic cleanup
            removed = cache.cleanup_expired()
            print(f"Removed {removed} expired entries")
        """
        current_time = time.time()
        
        with self._lock:
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics:
            - size: Number of entries in cache
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate (0.0-1.0)
            - invalidations: Number of times cache was invalidated
            - ttl_hours: TTL in hours
        
        Example:
            stats = cache.get_stats()
            print(f"Cache hit rate: {stats['hit_rate']:.2%}")
            print(f"Cache size: {stats['size']} entries")
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "invalidations": self._invalidations,
                "ttl_hours": self.ttl_hours
            }
    
    def clear_stats(self) -> None:
        """
        Reset cache statistics counters.
        
        Postconditions:
        - Hit, miss, and invalidation counters are reset to 0
        - Cache contents are not affected
        """
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._invalidations = 0

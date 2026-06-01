"""
graph/test_query_cache.py
--------------------------
Unit tests for QueryCache class

Tests cover:
- Cache initialization
- Cache key generation
- Cache get/set operations
- TTL expiration
- Cache invalidation
- Thread safety
- Cache statistics

Requirements: 13.5
"""

import pytest
import time
import threading
from graph.query_cache import QueryCache


class TestQueryCacheInitialization:
    """Test suite for QueryCache initialization."""
    
    def test_default_initialization(self):
        """Test cache initializes with default 24-hour TTL."""
        cache = QueryCache()
        
        assert cache.ttl_hours == 24
        assert cache.ttl_seconds == 24 * 3600
        assert cache._cache == {}
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._invalidations == 0
    
    def test_custom_ttl_initialization(self):
        """Test cache initializes with custom TTL."""
        cache = QueryCache(ttl_hours=12)
        
        assert cache.ttl_hours == 12
        assert cache.ttl_seconds == 12 * 3600
    
    def test_invalid_ttl_raises_error(self):
        """Test that invalid TTL raises ValueError."""
        with pytest.raises(ValueError, match="ttl_hours must be positive"):
            QueryCache(ttl_hours=0)
        
        with pytest.raises(ValueError, match="ttl_hours must be positive"):
            QueryCache(ttl_hours=-1)


class TestCacheKeyGeneration:
    """Test suite for cache key generation."""
    
    def test_cache_key_consistency(self):
        """Test that same parameters produce same cache key."""
        cache = QueryCache()
        
        params = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3
        }
        
        key1 = cache._generate_cache_key("query_cross_study_associations", params)
        key2 = cache._generate_cache_key("query_cross_study_associations", params)
        
        assert key1 == key2
    
    def test_cache_key_parameter_order_independence(self):
        """Test that parameter order doesn't affect cache key."""
        cache = QueryCache()
        
        params1 = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3
        }
        
        params2 = {
            "min_papers": 3,
            "disease": "Type 2 Diabetes",
            "study_type": "RCT"
        }
        
        key1 = cache._generate_cache_key("query_cross_study_associations", params1)
        key2 = cache._generate_cache_key("query_cross_study_associations", params2)
        
        assert key1 == key2
    
    def test_cache_key_different_parameters(self):
        """Test that different parameters produce different cache keys."""
        cache = QueryCache()
        
        params1 = {"disease": "Type 2 Diabetes", "study_type": "RCT"}
        params2 = {"disease": "Type 2 Diabetes", "study_type": "observational"}
        
        key1 = cache._generate_cache_key("query_cross_study_associations", params1)
        key2 = cache._generate_cache_key("query_cross_study_associations", params2)
        
        assert key1 != key2
    
    def test_cache_key_different_query_names(self):
        """Test that different query names produce different cache keys."""
        cache = QueryCache()
        
        params = {"disease": "Type 2 Diabetes"}
        
        key1 = cache._generate_cache_key("query_cross_study_associations", params)
        key2 = cache._generate_cache_key("query_intervention_evidence", params)
        
        assert key1 != key2


class TestCacheGetSet:
    """Test suite for cache get/set operations."""
    
    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        cache = QueryCache()
        
        result = cache.get("query_cross_study_associations", {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT"
        })
        
        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0
    
    def test_cache_hit_returns_result(self):
        """Test that cache hit returns stored result."""
        cache = QueryCache()
        
        params = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT"
        }
        
        # Store result
        expected_result = {"taxon": "Bacteroides", "paper_count": 5}
        cache.set("query_cross_study_associations", params, expected_result)
        
        # Retrieve result
        result = cache.get("query_cross_study_associations", params)
        
        assert result == expected_result
        assert cache._hits == 1
        assert cache._misses == 0
    
    def test_cache_set_none_does_not_cache(self):
        """Test that setting None does not cache the result."""
        cache = QueryCache()
        
        params = {"disease": "Type 2 Diabetes"}
        
        cache.set("query_cross_study_associations", params, None)
        
        result = cache.get("query_cross_study_associations", params)
        
        assert result is None
        assert cache._misses == 1
    
    def test_cache_multiple_queries(self):
        """Test caching multiple different queries."""
        cache = QueryCache()
        
        # Store multiple results
        cache.set("query_cross_study_associations", 
                 {"disease": "Type 2 Diabetes"}, 
                 {"result": "diabetes"})
        
        cache.set("query_intervention_evidence", 
                 {"intervention": "probiotic"}, 
                 {"result": "probiotic"})
        
        # Retrieve results
        result1 = cache.get("query_cross_study_associations", 
                           {"disease": "Type 2 Diabetes"})
        result2 = cache.get("query_intervention_evidence", 
                           {"intervention": "probiotic"})
        
        assert result1 == {"result": "diabetes"}
        assert result2 == {"result": "probiotic"}
        assert cache._hits == 2


class TestCacheTTLExpiration:
    """Test suite for cache TTL and expiration."""
    
    def test_expired_entry_returns_none(self):
        """Test that expired entries return None."""
        # Create cache with 1-second TTL
        cache = QueryCache(ttl_hours=1/3600)  # 1 second
        
        params = {"disease": "Type 2 Diabetes"}
        
        # Store result
        cache.set("query_cross_study_associations", params, {"result": "test"})
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Try to retrieve - should be expired
        result = cache.get("query_cross_study_associations", params)
        
        assert result is None
        assert cache._misses == 1
    
    def test_expired_entry_removed_from_cache(self):
        """Test that expired entries are removed from cache."""
        # Create cache with 1-second TTL
        cache = QueryCache(ttl_hours=1/3600)  # 1 second
        
        params = {"disease": "Type 2 Diabetes"}
        
        # Store result
        cache.set("query_cross_study_associations", params, {"result": "test"})
        
        assert len(cache._cache) == 1
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Try to retrieve - should remove expired entry
        cache.get("query_cross_study_associations", params)
        
        assert len(cache._cache) == 0
    
    def test_non_expired_entry_returns_result(self):
        """Test that non-expired entries return results."""
        # Create cache with 10-second TTL
        cache = QueryCache(ttl_hours=10/3600)  # 10 seconds
        
        params = {"disease": "Type 2 Diabetes"}
        
        # Store result
        cache.set("query_cross_study_associations", params, {"result": "test"})
        
        # Wait a bit but not enough to expire
        time.sleep(0.5)
        
        # Should still be cached
        result = cache.get("query_cross_study_associations", params)
        
        assert result == {"result": "test"}
        assert cache._hits == 1
    
    def test_cleanup_expired_removes_expired_entries(self):
        """Test that cleanup_expired removes expired entries."""
        # Create cache with 1-second TTL
        cache = QueryCache(ttl_hours=1/3600)  # 1 second
        
        # Store multiple results
        cache.set("query1", {"param": "1"}, {"result": "1"})
        cache.set("query2", {"param": "2"}, {"result": "2"})
        cache.set("query3", {"param": "3"}, {"result": "3"})
        
        assert len(cache._cache) == 3
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup expired entries
        removed = cache.cleanup_expired()
        
        assert removed == 3
        assert len(cache._cache) == 0


class TestCacheInvalidation:
    """Test suite for cache invalidation."""
    
    def test_invalidate_all_clears_cache(self):
        """Test that invalidate_all clears all entries."""
        cache = QueryCache()
        
        # Store multiple results
        cache.set("query1", {"param": "1"}, {"result": "1"})
        cache.set("query2", {"param": "2"}, {"result": "2"})
        cache.set("query3", {"param": "3"}, {"result": "3"})
        
        assert len(cache._cache) == 3
        
        # Invalidate all
        count = cache.invalidate_all()
        
        assert count == 3
        assert len(cache._cache) == 0
        assert cache._invalidations == 1
    
    def test_invalidate_all_empty_cache(self):
        """Test that invalidate_all on empty cache returns 0."""
        cache = QueryCache()
        
        count = cache.invalidate_all()
        
        assert count == 0
        assert cache._invalidations == 1
    
    def test_invalidate_query_removes_specific_entry(self):
        """Test that invalidate_query removes specific entry."""
        cache = QueryCache()
        
        # Store multiple results
        cache.set("query1", {"param": "1"}, {"result": "1"})
        cache.set("query2", {"param": "2"}, {"result": "2"})
        
        # Invalidate specific query
        removed = cache.invalidate_query("query1", {"param": "1"})
        
        assert removed is True
        assert len(cache._cache) == 1
        
        # Verify query1 is gone
        result1 = cache.get("query1", {"param": "1"})
        assert result1 is None
        
        # Verify query2 is still there
        result2 = cache.get("query2", {"param": "2"})
        assert result2 == {"result": "2"}
    
    def test_invalidate_query_nonexistent_entry(self):
        """Test that invalidate_query on nonexistent entry returns False."""
        cache = QueryCache()
        
        removed = cache.invalidate_query("query1", {"param": "1"})
        
        assert removed is False


class TestCacheStatistics:
    """Test suite for cache statistics."""
    
    def test_initial_stats(self):
        """Test initial cache statistics."""
        cache = QueryCache(ttl_hours=12)
        
        stats = cache.get_stats()
        
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["invalidations"] == 0
        assert stats["ttl_hours"] == 12
    
    def test_stats_after_operations(self):
        """Test cache statistics after various operations."""
        cache = QueryCache()
        
        # Cache miss
        cache.get("query1", {"param": "1"})
        
        # Cache set and hit
        cache.set("query1", {"param": "1"}, {"result": "1"})
        cache.get("query1", {"param": "1"})
        
        # Another miss
        cache.get("query2", {"param": "2"})
        
        # Invalidate
        cache.invalidate_all()
        
        stats = cache.get_stats()
        
        assert stats["size"] == 0  # Invalidated
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 1/3
        assert stats["invalidations"] == 1
    
    def test_clear_stats_resets_counters(self):
        """Test that clear_stats resets counters."""
        cache = QueryCache()
        
        # Perform operations
        cache.get("query1", {"param": "1"})
        cache.set("query1", {"param": "1"}, {"result": "1"})
        cache.get("query1", {"param": "1"})
        cache.invalidate_all()
        
        # Clear stats
        cache.clear_stats()
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["invalidations"] == 0
        assert stats["size"] == 0  # Cache was invalidated before clear_stats


class TestCacheThreadSafety:
    """Test suite for cache thread safety."""
    
    def test_concurrent_get_set(self):
        """Test concurrent get/set operations are thread-safe."""
        cache = QueryCache()
        results = []
        
        def worker(thread_id):
            # Each thread sets and gets its own result
            params = {"thread_id": thread_id}
            cache.set("query", params, {"result": thread_id})
            result = cache.get("query", params)
            results.append(result)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all results are correct
        assert len(results) == 10
        for i, result in enumerate(results):
            # Each thread should get its own result
            assert result is not None
    
    def test_concurrent_invalidation(self):
        """Test concurrent invalidation is thread-safe."""
        cache = QueryCache()
        
        # Pre-populate cache
        for i in range(100):
            cache.set("query", {"id": i}, {"result": i})
        
        def invalidate_worker():
            cache.invalidate_all()
        
        # Create multiple threads that invalidate
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=invalidate_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Cache should be empty
        assert len(cache._cache) == 0
        assert cache._invalidations == 5


class TestCacheIntegration:
    """Integration tests for cache with realistic scenarios."""
    
    def test_cache_workflow(self):
        """Test complete cache workflow: miss, set, hit, invalidate."""
        cache = QueryCache()
        
        params = {
            "disease": "Type 2 Diabetes",
            "study_type": "RCT",
            "min_papers": 3,
            "confidence_threshold": 0.7
        }
        
        # Initial cache miss
        result = cache.get("query_cross_study_associations", params)
        assert result is None
        
        # Set result
        query_result = {
            "taxon_name": "Bacteroides fragilis",
            "paper_count": 5,
            "consensus_confidence": 0.85
        }
        cache.set("query_cross_study_associations", params, query_result)
        
        # Cache hit
        result = cache.get("query_cross_study_associations", params)
        assert result == query_result
        
        # Invalidate cache (simulating new data load)
        count = cache.invalidate_all()
        assert count == 1
        
        # Cache miss after invalidation
        result = cache.get("query_cross_study_associations", params)
        assert result is None
        
        # Verify stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["invalidations"] == 1
    
    def test_multiple_query_types_cached(self):
        """Test caching multiple different query types."""
        cache = QueryCache()
        
        # Cache different query types
        cache.set("query_cross_study_associations", 
                 {"disease": "Type 2 Diabetes"}, 
                 {"result": "associations"})
        
        cache.set("query_intervention_evidence", 
                 {"intervention_types": ["probiotic"]}, 
                 {"result": "interventions"})
        
        cache.set("query_methodology_landscape", 
                 {"year_start": 2020, "year_end": 2024}, 
                 {"result": "methodology"})
        
        # Retrieve all
        result1 = cache.get("query_cross_study_associations", 
                           {"disease": "Type 2 Diabetes"})
        result2 = cache.get("query_intervention_evidence", 
                           {"intervention_types": ["probiotic"]})
        result3 = cache.get("query_methodology_landscape", 
                           {"year_start": 2020, "year_end": 2024})
        
        assert result1 == {"result": "associations"}
        assert result2 == {"result": "interventions"}
        assert result3 == {"result": "methodology"}
        
        stats = cache.get_stats()
        assert stats["size"] == 3
        assert stats["hits"] == 3

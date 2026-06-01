"""
graph/test_research_query_engine_caching.py
--------------------------------------------
Integration tests for ResearchQueryEngine caching functionality

Tests cover:
- Cache integration with query methods
- Cache hit/miss behavior
- Cache invalidation
- Cache statistics
- Caching disabled mode

Requirements: 13.5
"""

import pytest
from unittest.mock import Mock, MagicMock
from graph.research_query_engine import ResearchQueryEngine, QueryResult


class TestResearchQueryEngineCaching:
    """Test suite for ResearchQueryEngine caching integration."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    @pytest.fixture
    def engine_with_cache(self, mock_driver):
        """Create engine with caching enabled."""
        return ResearchQueryEngine(mock_driver, enable_cache=True, cache_ttl_hours=24)
    
    @pytest.fixture
    def engine_without_cache(self, mock_driver):
        """Create engine with caching disabled."""
        return ResearchQueryEngine(mock_driver, enable_cache=False)
    
    def test_engine_initializes_with_cache(self, mock_driver):
        """Test that engine initializes with cache enabled by default."""
        engine = ResearchQueryEngine(mock_driver)
        
        assert engine.enable_cache is True
        assert engine.cache is not None
        assert engine.cache.ttl_hours == 24
    
    def test_engine_initializes_without_cache(self, mock_driver):
        """Test that engine can be initialized without cache."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=False)
        
        assert engine.enable_cache is False
        assert engine.cache is None
    
    def test_engine_custom_cache_ttl(self, mock_driver):
        """Test that engine can be initialized with custom cache TTL."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True, cache_ttl_hours=12)
        
        assert engine.cache.ttl_hours == 12


class TestQueryCachingBehavior:
    """Test suite for query caching behavior."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver with query results."""
        driver = Mock()
        session = Mock()
        
        # Mock query results
        mock_record = {
            "taxon_name": "Bacteroides fragilis",
            "paper_count": 5,
            "consensus_confidence": 0.85,
            "consensus_direction": "increased",
            "direction_consistency": 0.80,
            "increased_count": 4,
            "decreased_count": 1,
            "no_change_count": 0,
            "paper_ids": ["PMID:12345", "PMID:67890"]
        }
        
        session.run.return_value = [mock_record]
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        
        return driver
    
    def test_first_query_is_cache_miss(self, mock_driver):
        """Test that first query execution is a cache miss."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Execute query
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7
        )
        
        # Verify query executed
        assert result.result_count == 1
        
        # Check cache stats
        stats = engine.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
    
    def test_second_identical_query_is_cache_hit(self, mock_driver):
        """Test that second identical query is a cache hit."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Execute query twice with same parameters
        result1 = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7
        )
        
        result2 = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7
        )
        
        # Both should return same results
        assert result1.result_count == result2.result_count
        assert result1.results == result2.results
        
        # Check cache stats
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        
        # Verify driver was only called once (second query used cache)
        assert mock_driver.session.return_value.__enter__.return_value.run.call_count == 1
    
    def test_different_parameters_cause_cache_miss(self, mock_driver):
        """Test that different parameters cause cache miss."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Execute query with different parameters
        result1 = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7
        )
        
        result2 = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="observational",  # Different parameter
            min_papers=3,
            confidence_threshold=0.7
        )
        
        # Check cache stats - both should be misses
        stats = engine.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2
        
        # Verify driver was called twice
        assert mock_driver.session.return_value.__enter__.return_value.run.call_count == 2
    
    def test_cache_works_for_all_query_methods(self, mock_driver):
        """Test that caching works for all query methods."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Mock different results for different queries
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.return_value = [{"result": "test"}]
        
        # Execute different query types twice each
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        
        engine.query_intervention_evidence(intervention_types=["probiotic"])
        engine.query_intervention_evidence(intervention_types=["probiotic"])
        
        engine.query_methodology_landscape(
            year_start=2020, 
            year_end=2024, 
            sequencing_methods=["16S rRNA sequencing"]
        )
        engine.query_methodology_landscape(
            year_start=2020, 
            year_end=2024, 
            sequencing_methods=["16S rRNA sequencing"]
        )
        
        engine.query_top_associations_by_evidence(disease="IBD")
        engine.query_top_associations_by_evidence(disease="IBD")
        
        engine.query_conflicting_evidence(disease="Crohn's Disease")
        engine.query_conflicting_evidence(disease="Crohn's Disease")
        
        # Check cache stats - should have 5 hits and 5 misses
        stats = engine.get_cache_stats()
        assert stats["hits"] == 5
        assert stats["misses"] == 5
        
        # Verify driver was only called 5 times (once per unique query)
        assert session.run.call_count == 5


class TestCacheInvalidation:
    """Test suite for cache invalidation."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        session.run.return_value = [{"result": "test"}]
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    def test_invalidate_cache_clears_all_entries(self, mock_driver):
        """Test that invalidate_cache clears all cached entries."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Execute and cache multiple queries
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        engine.query_intervention_evidence(intervention_types=["probiotic"])
        
        # Verify cache has entries
        stats = engine.get_cache_stats()
        assert stats["size"] == 2
        
        # Invalidate cache
        count = engine.invalidate_cache()
        
        assert count == 2
        
        # Verify cache is empty
        stats = engine.get_cache_stats()
        assert stats["size"] == 0
        assert stats["invalidations"] == 1
    
    def test_queries_after_invalidation_are_cache_misses(self, mock_driver):
        """Test that queries after invalidation are cache misses."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Execute query twice (second should be cache hit)
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        
        # Invalidate cache
        engine.invalidate_cache()
        
        # Execute same query again (should be cache miss)
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
    
    def test_invalidate_cache_without_cache_returns_zero(self, mock_driver):
        """Test that invalidate_cache returns 0 when caching is disabled."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=False)
        
        count = engine.invalidate_cache()
        
        assert count == 0


class TestCacheStatistics:
    """Test suite for cache statistics."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        session.run.return_value = [{"result": "test"}]
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    def test_get_cache_stats_with_cache_enabled(self, mock_driver):
        """Test get_cache_stats returns statistics when cache is enabled."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        stats = engine.get_cache_stats()
        
        assert stats is not None
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "invalidations" in stats
        assert "ttl_hours" in stats
    
    def test_get_cache_stats_without_cache_returns_none(self, mock_driver):
        """Test get_cache_stats returns None when cache is disabled."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=False)
        
        stats = engine.get_cache_stats()
        
        assert stats is None
    
    def test_cache_stats_track_operations(self, mock_driver):
        """Test that cache stats correctly track operations."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Execute queries
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        engine.query_cross_study_associations(disease="Type 2 Diabetes")  # Cache hit
        engine.query_cross_study_associations(disease="IBD")  # Cache miss
        
        stats = engine.get_cache_stats()
        
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 1/3


class TestCachingDisabled:
    """Test suite for behavior when caching is disabled."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        session.run.return_value = [{"result": "test"}]
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    def test_queries_execute_without_cache(self, mock_driver):
        """Test that queries execute normally when caching is disabled."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=False)
        
        # Execute query twice
        result1 = engine.query_cross_study_associations(disease="Type 2 Diabetes")
        result2 = engine.query_cross_study_associations(disease="Type 2 Diabetes")
        
        # Both should execute successfully
        assert result1.error is None
        assert result2.error is None
        
        # Verify driver was called twice (no caching)
        session = mock_driver.session.return_value.__enter__.return_value
        assert session.run.call_count == 2
    
    def test_cache_methods_work_without_cache(self, mock_driver):
        """Test that cache methods work gracefully when caching is disabled."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=False)
        
        # These should not raise errors
        count = engine.invalidate_cache()
        assert count == 0
        
        stats = engine.get_cache_stats()
        assert stats is None


class TestCacheErrorHandling:
    """Test suite for cache error handling."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver that raises errors."""
        driver = Mock()
        session = Mock()
        session.run.side_effect = Exception("Database error")
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    def test_failed_queries_not_cached(self, mock_driver):
        """Test that failed queries are not cached."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Execute query that will fail
        result = engine.query_cross_study_associations(disease="Type 2 Diabetes")
        
        # Verify query failed
        assert result.error is not None
        
        # Verify result was not cached
        stats = engine.get_cache_stats()
        assert stats["size"] == 0
        assert stats["misses"] == 1


class TestCacheIntegrationScenarios:
    """Integration tests for realistic caching scenarios."""
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        session.run.return_value = [{"result": "test"}]
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    def test_researcher_workflow_with_caching(self, mock_driver):
        """Test typical researcher workflow with caching."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Researcher explores Type 2 Diabetes associations
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        engine.query_cross_study_associations(disease="Type 2 Diabetes")  # Cache hit
        
        # Researcher checks interventions
        engine.query_intervention_evidence(intervention_types=["probiotic"])
        engine.query_intervention_evidence(intervention_types=["probiotic"])  # Cache hit
        
        # Researcher goes back to associations
        engine.query_cross_study_associations(disease="Type 2 Diabetes")  # Cache hit
        
        # Check cache performance
        stats = engine.get_cache_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 3/5
        
        # Verify only 2 actual database queries
        session = mock_driver.session.return_value.__enter__.return_value
        assert session.run.call_count == 2
    
    def test_data_load_invalidation_workflow(self, mock_driver):
        """Test workflow where new data is loaded and cache is invalidated."""
        engine = ResearchQueryEngine(mock_driver, enable_cache=True)
        
        # Initial queries
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        engine.query_cross_study_associations(disease="Type 2 Diabetes")  # Cache hit
        
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1
        
        # Simulate new data load
        # In real scenario: load_new_papers_to_graph()
        engine.invalidate_cache()
        
        # Query again - should be cache miss
        engine.query_cross_study_associations(disease="Type 2 Diabetes")
        
        stats = engine.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["invalidations"] == 1

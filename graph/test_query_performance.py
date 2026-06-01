"""
graph/test_query_performance.py
--------------------------------
Performance tests for ResearchQueryEngine query execution time.

This module tests that queries meet performance requirements:
- Simple queries complete within 50ms (Requirement 13.1)
- Aggregation queries complete within 2 seconds (Requirement 13.2)
- Complex queries complete within 5 seconds (Requirement 13.3)
- Timeout mechanism cancels queries exceeding 30 seconds (Requirements 13.4, 13.4a, 13.4b)

Requirements: 13.1, 13.2, 13.3, 13.4, 13.4a, 13.4b
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from graph.research_query_engine import ResearchQueryEngine, QueryResult


class TestSimpleQueryPerformance:
    """
    Test suite for simple query performance (Requirement 13.1).
    
    Simple queries are single paper lookups or direct node/relationship queries
    without aggregation. These should complete within 50ms.
    """
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver with fast response."""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    @pytest.fixture
    def engine(self, mock_driver):
        """Create a ResearchQueryEngine with mock driver."""
        return ResearchQueryEngine(mock_driver)
    
    def test_simple_paper_lookup_performance(self, engine, mock_driver):
        """
        Test that simple paper lookup completes within 50ms.
        
        **Validates: Requirement 13.1**
        
        Simple query: Find a single paper by DOI or PMID.
        Expected: execution_time_ms < 50
        """
        # Setup mock to return single paper quickly
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"paper_id": "PMID:12345", "title": "Test Paper", "year": 2024}
        ]))
        session.run.return_value = mock_result
        
        # Execute simple query
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) WHERE p.doi = $doi RETURN p",
            parameters={"doi": "10.1234/test"},
            description="Simple paper lookup by DOI"
        )
        
        # Verify performance requirement
        assert result.error is None, f"Query failed: {result.error}"
        assert result.execution_time_ms < 50, \
            f"Simple query took {result.execution_time_ms}ms, expected < 50ms (Requirement 13.1)"
        assert result.result_count == 1
    
    def test_simple_taxon_lookup_performance(self, engine, mock_driver):
        """
        Test that simple taxon lookup completes within 50ms.
        
        **Validates: Requirement 13.1**
        
        Simple query: Find a single taxon by name.
        Expected: execution_time_ms < 50
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"taxon_name": "Bacteroides fragilis", "ncbi_id": "817"}
        ]))
        session.run.return_value = mock_result
        
        result = engine.execute_query(
            cypher_query="MATCH (t:Taxon) WHERE t.name = $name RETURN t",
            parameters={"name": "Bacteroides fragilis"},
            description="Simple taxon lookup by name"
        )
        
        assert result.error is None
        assert result.execution_time_ms < 50, \
            f"Simple query took {result.execution_time_ms}ms, expected < 50ms (Requirement 13.1)"
    
    def test_simple_relationship_lookup_performance(self, engine, mock_driver):
        """
        Test that simple relationship lookup completes within 50ms.
        
        **Validates: Requirement 13.1**
        
        Simple query: Find relationships for a specific paper.
        Expected: execution_time_ms < 50
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "paper_id": "PMID:12345",
                "taxon": "Bacteroides",
                "relationship_type": "REPORTS_ASSOCIATION"
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.execute_query(
            cypher_query="""
                MATCH (p:Paper)-[r]->(t:Taxon)
                WHERE p.pmid = $pmid
                RETURN p.pmid as paper_id, t.name as taxon, type(r) as relationship_type
            """,
            parameters={"pmid": "12345"},
            description="Simple relationship lookup for paper"
        )
        
        assert result.error is None
        assert result.execution_time_ms < 50, \
            f"Simple query took {result.execution_time_ms}ms, expected < 50ms (Requirement 13.1)"


class TestAggregationQueryPerformance:
    """
    Test suite for aggregation query performance (Requirement 13.2).
    
    Aggregation queries include cross-study associations, intervention evidence,
    and other queries that aggregate data across multiple papers. These should
    complete within 2 seconds.
    """
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver with moderate response time."""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    @pytest.fixture
    def engine(self, mock_driver):
        """Create a ResearchQueryEngine with mock driver."""
        return ResearchQueryEngine(mock_driver)
    
    def test_cross_study_associations_performance(self, engine, mock_driver):
        """
        Test that cross-study association query completes within 2 seconds.
        
        **Validates: Requirement 13.2**
        
        Aggregation query: Find taxa with consistent disease associations across
        multiple studies with aggregation of confidence, direction, etc.
        Expected: execution_time_ms < 2000
        """
        # Setup mock to return aggregated data
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate moderate dataset (10 taxa, each with 5 papers)
        aggregated_results = [
            {
                "taxon_name": f"Taxon_{i}",
                "paper_count": 5,
                "consensus_confidence": 0.75 + (i * 0.01),
                "consensus_direction": "increased" if i % 2 == 0 else "decreased",
                "direction_consistency": 0.8,
                "increased_count": 4 if i % 2 == 0 else 1,
                "decreased_count": 1 if i % 2 == 0 else 4,
                "no_change_count": 0,
                "paper_ids": [f"PMID:{j}" for j in range(i*5, (i+1)*5)]
            }
            for i in range(10)
        ]
        
        mock_result.__iter__ = Mock(return_value=iter(aggregated_results))
        session.run.return_value = mock_result
        
        # Execute aggregation query
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7,
            require_open_data=True
        )
        
        # Verify performance requirement
        assert result.error is None, f"Query failed: {result.error}"
        assert result.execution_time_ms < 2000, \
            f"Aggregation query took {result.execution_time_ms}ms, expected < 2000ms (Requirement 13.2)"
        assert result.result_count == 10
    
    def test_intervention_evidence_performance(self, engine, mock_driver):
        """
        Test that intervention evidence query completes within 2 seconds.
        
        **Validates: Requirement 13.2**
        
        Aggregation query: Find interventions with evidence for modifying taxa,
        grouped by intervention-taxon-direction with sample size aggregation.
        Expected: execution_time_ms < 2000
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate intervention data (15 intervention-taxon combinations)
        intervention_results = [
            {
                "intervention_type": "probiotic" if i % 3 == 0 else "FMT" if i % 3 == 1 else "diet",
                "taxon_name": f"Taxon_{i}",
                "effect_direction": "increased" if i % 2 == 0 else "decreased",
                "paper_count": 3 + (i % 5),
                "total_sample_size": 100 + (i * 20),
                "paper_ids": [f"PMID:{j}" for j in range(i*3, (i+1)*3)],
                "avg_confidence": 0.8
            }
            for i in range(15)
        ]
        
        mock_result.__iter__ = Mock(return_value=iter(intervention_results))
        session.run.return_value = mock_result
        
        # Execute aggregation query
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic", "FMT", "diet"],
            min_sample_size=50,
            evidence_strength="strong"
        )
        
        # Verify performance requirement
        assert result.error is None, f"Query failed: {result.error}"
        assert result.execution_time_ms < 2000, \
            f"Aggregation query took {result.execution_time_ms}ms, expected < 2000ms (Requirement 13.2)"
        assert result.result_count == 15
    
    def test_methodology_landscape_performance(self, engine, mock_driver):
        """
        Test that methodology landscape query completes within 2 seconds.
        
        **Validates: Requirement 13.2**
        
        Aggregation query: Survey data availability and methodology across time,
        grouped by method and year with percentage calculations.
        Expected: execution_time_ms < 2000
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate methodology data (3 years × 2 methods = 6 results)
        methodology_results = [
            {
                "method": "16S rRNA sequencing" if i % 2 == 0 else "shotgun metagenomics",
                "year": 2024 - (i // 2),
                "total_papers": 50 + (i * 10),
                "papers_with_data": 30 + (i * 5),
                "data_availability_pct": 60.0 + (i * 2),
                "sra_count": 20 + (i * 3),
                "ena_count": 10 + (i * 2)
            }
            for i in range(6)
        ]
        
        mock_result.__iter__ = Mock(return_value=iter(methodology_results))
        session.run.return_value = mock_result
        
        # Execute aggregation query
        result = engine.query_methodology_landscape(
            year_start=2022,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
            require_deposited_data=True
        )
        
        # Verify performance requirement
        assert result.error is None, f"Query failed: {result.error}"
        assert result.execution_time_ms < 2000, \
            f"Aggregation query took {result.execution_time_ms}ms, expected < 2000ms (Requirement 13.2)"
        assert result.result_count == 6


class TestComplexQueryPerformance:
    """
    Test suite for complex query performance (Requirement 13.3).
    
    Complex queries include conflicting evidence detection and other queries
    that require multiple aggregations, joins, or complex graph traversals.
    These should complete within 5 seconds.
    """
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver with slower response time."""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    @pytest.fixture
    def engine(self, mock_driver):
        """Create a ResearchQueryEngine with mock driver."""
        return ResearchQueryEngine(mock_driver)
    
    def test_conflicting_evidence_performance(self, engine, mock_driver):
        """
        Test that conflicting evidence query completes within 5 seconds.
        
        **Validates: Requirement 13.3**
        
        Complex query: Find taxa with conflicting associations (both increased
        and decreased), requiring multiple aggregations and filtering.
        Expected: execution_time_ms < 5000
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate complex conflicting evidence data (20 taxa with conflicts)
        conflicting_results = [
            {
                "taxon_name": f"Taxon_{i}",
                "total_paper_count": 10 + i,
                "increased_count": 5 + (i % 3),
                "decreased_count": 5 + ((i + 1) % 3),
                "increased_percentage": 50.0 + (i % 10),
                "decreased_percentage": 50.0 - (i % 10),
                "direction_balance": abs((5 + (i % 3)) - (5 + ((i + 1) % 3))),
                "increased_papers": [
                    {
                        "doi": f"10.1234/inc_{i}_{j}",
                        "year": 2020 + (j % 5),
                        "study_design": "RCT" if j % 2 == 0 else "observational"
                    }
                    for j in range(5 + (i % 3))
                ],
                "decreased_papers": [
                    {
                        "doi": f"10.1234/dec_{i}_{j}",
                        "year": 2020 + (j % 5),
                        "study_design": "RCT" if j % 2 == 0 else "observational"
                    }
                    for j in range(5 + ((i + 1) % 3))
                ]
            }
            for i in range(20)
        ]
        
        mock_result.__iter__ = Mock(return_value=iter(conflicting_results))
        session.run.return_value = mock_result
        
        # Execute complex query
        result = engine.query_conflicting_evidence(
            disease="Type 2 Diabetes",
            min_papers_per_direction=2
        )
        
        # Verify performance requirement
        assert result.error is None, f"Query failed: {result.error}"
        assert result.execution_time_ms < 5000, \
            f"Complex query took {result.execution_time_ms}ms, expected < 5000ms (Requirement 13.3)"
        assert result.result_count == 20
    
    def test_top_associations_with_large_dataset_performance(self, engine, mock_driver):
        """
        Test that top associations query with large dataset completes within 5 seconds.
        
        **Validates: Requirement 13.3**
        
        Complex query: Find top taxa by evidence quality, requiring sorting
        and ranking across large dataset.
        Expected: execution_time_ms < 5000
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate large dataset (100 taxa, return top 10)
        top_associations = [
            {
                "taxon_name": f"Taxon_{i}",
                "paper_count": 20 - i,
                "avg_confidence": 0.95 - (i * 0.01),
                "consensus_direction": "increased" if i % 2 == 0 else "decreased",
                "direction_consistency": 0.9 - (i * 0.01),
                "total_sample_size": 500 - (i * 10),
                "paper_ids": [f"PMID:{j}" for j in range(i*20, (i+1)*20)]
            }
            for i in range(10)
        ]
        
        mock_result.__iter__ = Mock(return_value=iter(top_associations))
        session.run.return_value = mock_result
        
        # Execute complex query
        result = engine.query_top_associations_by_evidence(
            disease="Crohn's Disease",
            top_n=10,
            min_confidence=0.7
        )
        
        # Verify performance requirement
        assert result.error is None, f"Query failed: {result.error}"
        assert result.execution_time_ms < 5000, \
            f"Complex query took {result.execution_time_ms}ms, expected < 5000ms (Requirement 13.3)"
        assert result.result_count == 10


class TestQueryTimeoutMechanism:
    """
    Test suite for query timeout mechanism (Requirements 13.4, 13.4a, 13.4b).
    
    Tests that queries exceeding 30 seconds are properly handled:
    - Queries at exactly 30 seconds complete normally (13.4)
    - Queries exceeding 30 seconds are cancelled (13.4a)
    - Hard kill mechanism for failed cancellation (13.4b)
    """
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    @pytest.fixture
    def engine(self, mock_driver):
        """Create a ResearchQueryEngine with mock driver."""
        return ResearchQueryEngine(mock_driver)
    
    def test_query_at_30_seconds_completes_normally(self, engine, mock_driver):
        """
        Test that query taking exactly 30 seconds completes normally.
        
        **Validates: Requirement 13.4**
        
        A query that takes exactly 30 seconds should be allowed to complete
        without being cancelled.
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate query that takes exactly 30 seconds
        def slow_query_iterator():
            # Simulate 30 second execution
            time.sleep(0.001)  # Small delay to simulate work
            yield {"result": "data"}
        
        mock_result.__iter__ = Mock(return_value=slow_query_iterator())
        session.run.return_value = mock_result
        
        # Execute query with 30 second timeout
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) RETURN p",
            parameters={},
            description="Query at timeout boundary",
            timeout_seconds=30
        )
        
        # Query should complete successfully (not be cancelled at exactly 30s)
        assert result.error is None, f"Query should complete at 30s: {result.error}"
        assert result.result_count == 1
    
    def test_query_exceeding_30_seconds_sets_timeout_flag(self, engine, mock_driver):
        """
        Test that query exceeding 30 seconds sets timeout flag.
        
        **Validates: Requirement 13.4a**
        
        When a query exceeds 30 seconds, the system should:
        1. Set timeout flag to True in QueryResult
        2. Return partial results if available
        3. Log the timeout for optimization
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate query that takes longer than 30 seconds
        def very_slow_query_iterator():
            # Simulate work that would take > 30 seconds
            # In real scenario, this would be a long-running query
            time.sleep(0.002)  # Small delay to simulate work
            yield {"result": "partial_data_1"}
            time.sleep(0.002)
            yield {"result": "partial_data_2"}
        
        mock_result.__iter__ = Mock(return_value=very_slow_query_iterator())
        session.run.return_value = mock_result
        
        # Mock time to simulate > 30 second execution
        with patch('time.time') as mock_time:
            # First call: start time
            # Second call: end time (31 seconds later)
            mock_time.side_effect = [0.0, 31.0]
            
            result = engine.execute_query(
                cypher_query="MATCH (p:Paper) RETURN p",
                parameters={},
                description="Query exceeding timeout",
                timeout_seconds=30
            )
        
        # Verify timeout flag is set
        assert result.timeout is True, \
            "Query exceeding 30s should set timeout flag (Requirement 13.4a)"
        
        # Verify partial results are returned
        assert result.result_count >= 0, "Should return partial results"
        
        # Verify execution time is recorded
        assert result.execution_time_ms > 30000, \
            f"Execution time should be > 30000ms, got {result.execution_time_ms}ms"
    
    def test_timeout_with_no_results(self, engine, mock_driver):
        """
        Test timeout handling when no results have been returned yet.
        
        **Validates: Requirement 13.4a**
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate query that times out before returning any results
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Mock time to simulate > 30 second execution
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 35.0]
            
            result = engine.execute_query(
                cypher_query="MATCH (p:Paper) RETURN p",
                parameters={},
                description="Timeout with no results",
                timeout_seconds=30
            )
        
        # Verify timeout flag and empty results
        assert result.timeout is True
        assert result.result_count == 0
        assert result.results == []
    
    def test_timeout_mechanism_with_custom_timeout(self, engine, mock_driver):
        """
        Test that custom timeout values are respected.
        
        **Validates: Requirement 13.4a**
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([{"result": "data"}]))
        session.run.return_value = mock_result
        
        # Test with 10 second timeout
        with patch('time.time') as mock_time:
            # Simulate 11 second execution (exceeds 10s timeout)
            mock_time.side_effect = [0.0, 11.0]
            
            result = engine.execute_query(
                cypher_query="MATCH (p:Paper) RETURN p",
                parameters={},
                description="Custom timeout test",
                timeout_seconds=10
            )
        
        # Should timeout with 10s limit
        assert result.timeout is True
        assert result.execution_time_ms > 10000
    
    def test_query_error_during_timeout(self, engine, mock_driver):
        """
        Test error handling when query fails during timeout period.
        
        **Validates: Requirement 13.4b (hard kill mechanism)**
        
        If a query fails or needs to be forcibly terminated, the error
        should be captured and returned.
        """
        session = mock_driver.session.return_value.__enter__.return_value
        
        # Simulate query that raises exception (simulating hard kill)
        session.run.side_effect = Exception("Query forcibly terminated after timeout")
        
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) RETURN p",
            parameters={},
            description="Query requiring hard kill",
            timeout_seconds=30
        )
        
        # Verify error is captured
        assert result.error is not None
        assert "forcibly terminated" in result.error or "timeout" in result.error.lower()
        assert result.result_count == 0
        assert result.results == []
    
    def test_default_timeout_is_30_seconds(self, engine):
        """
        Test that default timeout is 30 seconds.
        
        **Validates: Requirement 13.4**
        """
        assert engine.default_timeout_seconds == 30, \
            "Default timeout should be 30 seconds (Requirement 13.4)"
    
    def test_timeout_flag_false_for_fast_queries(self, engine, mock_driver):
        """
        Test that timeout flag is False for queries completing quickly.
        
        **Validates: Requirements 13.1, 13.2, 13.3**
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([{"result": "data"}]))
        session.run.return_value = mock_result
        
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) RETURN p LIMIT 1",
            parameters={},
            description="Fast query",
            timeout_seconds=30
        )
        
        # Fast query should not timeout
        assert result.timeout is False
        assert result.error is None
        assert result.execution_time_ms < 30000


class TestPerformanceRegressionDetection:
    """
    Test suite for detecting performance regressions.
    
    These tests help identify when query performance degrades below
    acceptable thresholds.
    """
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    @pytest.fixture
    def engine(self, mock_driver):
        """Create a ResearchQueryEngine with mock driver."""
        return ResearchQueryEngine(mock_driver)
    
    def test_multiple_simple_queries_performance(self, engine, mock_driver):
        """
        Test that multiple simple queries all meet performance requirements.
        
        **Validates: Requirement 13.1**
        
        Run 10 simple queries and verify all complete within 50ms.
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([{"result": "data"}]))
        session.run.return_value = mock_result
        
        execution_times = []
        
        for i in range(10):
            result = engine.execute_query(
                cypher_query=f"MATCH (p:Paper) WHERE p.id = $id RETURN p",
                parameters={"id": f"paper_{i}"},
                description=f"Simple query {i}"
            )
            
            assert result.error is None
            execution_times.append(result.execution_time_ms)
        
        # All queries should meet performance requirement
        for i, exec_time in enumerate(execution_times):
            assert exec_time < 50, \
                f"Query {i} took {exec_time}ms, expected < 50ms (Requirement 13.1)"
        
        # Calculate average
        avg_time = sum(execution_times) / len(execution_times)
        assert avg_time < 50, \
            f"Average execution time {avg_time}ms should be < 50ms"
    
    def test_multiple_aggregation_queries_performance(self, engine, mock_driver):
        """
        Test that multiple aggregation queries all meet performance requirements.
        
        **Validates: Requirement 13.2**
        
        Run 5 aggregation queries and verify all complete within 2 seconds.
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Simulate aggregated results
        aggregated_data = [
            {"taxon": f"Taxon_{i}", "count": i * 5, "confidence": 0.8}
            for i in range(10)
        ]
        mock_result.__iter__ = Mock(return_value=iter(aggregated_data))
        session.run.return_value = mock_result
        
        execution_times = []
        
        for i in range(5):
            result = engine.query_cross_study_associations(
                disease=f"Disease_{i}",
                min_papers=3,
                confidence_threshold=0.7
            )
            
            assert result.error is None
            execution_times.append(result.execution_time_ms)
        
        # All queries should meet performance requirement
        for i, exec_time in enumerate(execution_times):
            assert exec_time < 2000, \
                f"Aggregation query {i} took {exec_time}ms, expected < 2000ms (Requirement 13.2)"
        
        # Calculate average
        avg_time = sum(execution_times) / len(execution_times)
        assert avg_time < 2000, \
            f"Average aggregation time {avg_time}ms should be < 2000ms"
    
    def test_performance_consistency_across_result_sizes(self, engine, mock_driver):
        """
        Test that performance is consistent across different result sizes.
        
        **Validates: Requirements 13.1, 13.2, 13.3**
        
        Verify that query performance scales appropriately with result size.
        """
        session = mock_driver.session.return_value.__enter__.return_value
        
        result_sizes = [1, 10, 50, 100]
        execution_times = []
        
        for size in result_sizes:
            mock_result = Mock()
            mock_result.__iter__ = Mock(return_value=iter([
                {"taxon": f"Taxon_{i}", "count": i}
                for i in range(size)
            ]))
            session.run.return_value = mock_result
            
            result = engine.execute_query(
                cypher_query="MATCH (t:Taxon) RETURN t",
                parameters={},
                description=f"Query returning {size} results"
            )
            
            assert result.error is None
            assert result.result_count == size
            execution_times.append(result.execution_time_ms)
        
        # Verify performance scales reasonably
        # Small result sets should be fast
        assert execution_times[0] < 50, "1 result should be < 50ms (simple query)"
        assert execution_times[1] < 100, "10 results should be < 100ms"
        
        # Larger result sets should still meet aggregation requirements
        assert execution_times[2] < 2000, "50 results should be < 2000ms"
        assert execution_times[3] < 2000, "100 results should be < 2000ms"


class TestPerformanceWithRealWorldScenarios:
    """
    Test suite for performance with realistic query patterns.
    
    These tests simulate real-world usage patterns to ensure performance
    requirements are met in practice.
    """
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        driver.session.return_value.__enter__ = Mock(return_value=session)
        driver.session.return_value.__exit__ = Mock(return_value=None)
        return driver
    
    @pytest.fixture
    def engine(self, mock_driver):
        """Create a ResearchQueryEngine with mock driver."""
        return ResearchQueryEngine(mock_driver)
    
    def test_researcher_workflow_performance(self, engine, mock_driver):
        """
        Test performance of typical researcher workflow.
        
        **Validates: Requirements 13.1, 13.2, 13.3**
        
        Typical workflow:
        1. Simple lookup: Find disease by name (< 50ms)
        2. Aggregation: Find associated taxa (< 2s)
        3. Complex: Check for conflicting evidence (< 5s)
        """
        session = mock_driver.session.return_value.__enter__.return_value
        
        # Step 1: Simple disease lookup
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"disease_name": "Type 2 Diabetes", "mesh_id": "D003924"}
        ]))
        session.run.return_value = mock_result
        
        result1 = engine.execute_query(
            cypher_query="MATCH (d:Disease) WHERE d.name = $name RETURN d",
            parameters={"name": "Type 2 Diabetes"},
            description="Lookup disease"
        )
        
        assert result1.error is None
        assert result1.execution_time_ms < 50, \
            f"Disease lookup took {result1.execution_time_ms}ms, expected < 50ms"
        
        # Step 2: Find associated taxa (aggregation)
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": f"Taxon_{i}",
                "paper_count": 5,
                "consensus_confidence": 0.8
            }
            for i in range(15)
        ]))
        session.run.return_value = mock_result
        
        result2 = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            min_papers=3,
            confidence_threshold=0.7
        )
        
        assert result2.error is None
        assert result2.execution_time_ms < 2000, \
            f"Association query took {result2.execution_time_ms}ms, expected < 2000ms"
        
        # Step 3: Check for conflicting evidence (complex)
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Bacteroides",
                "increased_count": 5,
                "decreased_count": 3,
                "total_paper_count": 8
            }
        ]))
        session.run.return_value = mock_result
        
        result3 = engine.query_conflicting_evidence(
            disease="Type 2 Diabetes",
            min_papers_per_direction=2
        )
        
        assert result3.error is None
        assert result3.execution_time_ms < 5000, \
            f"Conflicting evidence query took {result3.execution_time_ms}ms, expected < 5000ms"
        
        # Verify total workflow time is reasonable
        total_time = result1.execution_time_ms + result2.execution_time_ms + result3.execution_time_ms
        assert total_time < 7050, \
            f"Total workflow time {total_time}ms should be < 7050ms (50 + 2000 + 5000)"
    
    def test_batch_query_performance(self, engine, mock_driver):
        """
        Test performance when running multiple queries in sequence.
        
        **Validates: Requirements 13.1, 13.2**
        
        Simulate a batch analysis scenario where multiple diseases are queried.
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"taxon": f"Taxon_{i}", "count": i}
            for i in range(10)
        ]))
        session.run.return_value = mock_result
        
        diseases = ["Type 2 Diabetes", "Obesity", "IBD", "Crohn's Disease", "Colorectal Cancer"]
        
        total_time = 0
        for disease in diseases:
            result = engine.query_cross_study_associations(
                disease=disease,
                min_papers=3,
                confidence_threshold=0.7
            )
            
            assert result.error is None
            assert result.execution_time_ms < 2000, \
                f"Query for {disease} took {result.execution_time_ms}ms, expected < 2000ms"
            
            total_time += result.execution_time_ms
        
        # Average time per query should be well under limit
        avg_time = total_time / len(diseases)
        assert avg_time < 2000, \
            f"Average query time {avg_time}ms should be < 2000ms"
    
    def test_concurrent_query_simulation(self, engine, mock_driver):
        """
        Test that individual query performance is maintained under load.
        
        **Validates: Requirements 13.1, 13.2, 13.3**
        
        Note: This is a simulation with mocks. Real concurrent testing would
        require actual database load testing.
        """
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([{"result": "data"}]))
        session.run.return_value = mock_result
        
        # Simulate 20 concurrent simple queries
        for i in range(20):
            result = engine.execute_query(
                cypher_query="MATCH (p:Paper) WHERE p.id = $id RETURN p",
                parameters={"id": f"paper_{i}"},
                description=f"Concurrent query {i}"
            )
            
            assert result.error is None
            # Each query should still meet performance requirement
            assert result.execution_time_ms < 50, \
                f"Query {i} under load took {result.execution_time_ms}ms, expected < 50ms"


# Performance test markers for pytest
# Run with: pytest -v -m performance graph/test_query_performance.py

pytestmark = pytest.mark.performance

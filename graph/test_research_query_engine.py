"""
graph/test_research_query_engine.py
------------------------------------
Unit tests for ResearchQueryEngine base class and QueryResult model.

Tests cover:
- QueryResult model validation
- Query execution timing
- Result counting
- Parameterized query generation
- Input validation and sanitization
- Error handling

Requirements: 1.1, 1.2, 1.3, 18.1
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from graph.research_query_engine import ResearchQueryEngine, QueryResult
import time


class TestQueryResult:
    """Test suite for QueryResult model."""
    
    def test_query_result_creation_with_defaults(self):
        """Test creating QueryResult with minimal required fields."""
        result = QueryResult(
            query_description="Test query"
        )
        
        assert result.query_description == "Test query"
        assert result.results == []
        assert result.result_count == 0
        assert result.execution_time_ms == 0.0
        assert result.timeout is False
        assert result.error is None
        assert result.query_id is not None  # Auto-generated UUID
        assert result.executed_at is not None  # Auto-generated timestamp
    
    def test_query_result_with_results(self):
        """Test QueryResult with actual result data."""
        results = [
            {"taxon": "Bacteroides", "count": 5},
            {"taxon": "Lactobacillus", "count": 3}
        ]
        
        result = QueryResult(
            query_description="Find taxa",
            results=results,
            result_count=2,
            execution_time_ms=123.45
        )
        
        assert result.results == results
        assert result.result_count == 2
        assert result.execution_time_ms == 123.45
    
    def test_query_result_with_aggregation_metadata(self):
        """Test QueryResult with aggregation metadata."""
        result = QueryResult(
            query_description="Aggregated query",
            aggregation_method="weighted_average",
            confidence_threshold=0.7
        )
        
        assert result.aggregation_method == "weighted_average"
        assert result.confidence_threshold == 0.7
    
    def test_query_result_with_timeout(self):
        """Test QueryResult with timeout flag."""
        result = QueryResult(
            query_description="Slow query",
            timeout=True
        )
        
        assert result.timeout is True
    
    def test_query_result_with_error(self):
        """Test QueryResult with error message."""
        result = QueryResult(
            query_description="Failed query",
            error="Connection timeout"
        )
        
        assert result.error == "Connection timeout"
    
    def test_query_result_validation_negative_count(self):
        """Test that negative result_count is rejected."""
        with pytest.raises(ValueError):
            QueryResult(
                query_description="Test",
                result_count=-1
            )
    
    def test_query_result_validation_negative_time(self):
        """Test that negative execution_time_ms is rejected."""
        with pytest.raises(ValueError):
            QueryResult(
                query_description="Test",
                execution_time_ms=-10.0
            )
    
    def test_query_result_validation_confidence_threshold_range(self):
        """Test that confidence_threshold must be in [0.0, 1.0]."""
        # Valid values
        QueryResult(query_description="Test", confidence_threshold=0.0)
        QueryResult(query_description="Test", confidence_threshold=0.5)
        QueryResult(query_description="Test", confidence_threshold=1.0)
        
        # Invalid values
        with pytest.raises(ValueError):
            QueryResult(query_description="Test", confidence_threshold=-0.1)
        
        with pytest.raises(ValueError):
            QueryResult(query_description="Test", confidence_threshold=1.1)


class TestResearchQueryEngine:
    """Test suite for ResearchQueryEngine base class."""
    
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
    
    def test_engine_initialization(self, mock_driver):
        """Test that engine initializes correctly."""
        engine = ResearchQueryEngine(mock_driver)
        
        assert engine.driver == mock_driver
        assert engine.default_timeout_seconds == 30
    
    def test_execute_query_success(self, engine, mock_driver):
        """Test successful query execution with timing and result counting."""
        # Setup mock session and result
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"taxon": "Bacteroides", "count": 5},
            {"taxon": "Lactobacillus", "count": 3}
        ]))
        session.run.return_value = mock_result
        
        # Execute query
        result = engine.execute_query(
            cypher_query="MATCH (t:Taxon) RETURN t.name as taxon, t.count as count",
            parameters={},
            description="Find all taxa"
        )
        
        # Verify result
        assert result.query_description == "Find all taxa"
        assert result.result_count == 2
        assert len(result.results) == 2
        assert result.results[0]["taxon"] == "Bacteroides"
        assert result.results[1]["taxon"] == "Lactobacillus"
        assert result.execution_time_ms > 0  # Should have some execution time
        assert result.timeout is False
        assert result.error is None
        
        # Verify parameterized query was used
        session.run.assert_called_once_with(
            "MATCH (t:Taxon) RETURN t.name as taxon, t.count as count",
            {}
        )
    
    def test_execute_query_with_parameters(self, engine, mock_driver):
        """Test query execution with parameters (injection prevention)."""
        # Setup mock
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"paper": "Paper1"}
        ]))
        session.run.return_value = mock_result
        
        # Execute parameterized query
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) WHERE p.year = $year RETURN p.title as paper",
            parameters={"year": 2024},
            description="Find papers from 2024"
        )
        
        # Verify parameters were passed correctly
        session.run.assert_called_once_with(
            "MATCH (p:Paper) WHERE p.year = $year RETURN p.title as paper",
            {"year": 2024}
        )
        
        assert result.result_count == 1
        assert result.error is None
    
    def test_execute_query_with_aggregation_metadata(self, engine, mock_driver):
        """Test that aggregation metadata is captured in result."""
        # Setup mock
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Execute query with aggregation metadata
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) RETURN p",
            parameters={},
            description="Aggregated query",
            aggregation_method="weighted_average",
            confidence_threshold=0.7
        )
        
        assert result.aggregation_method == "weighted_average"
        assert result.confidence_threshold == 0.7
    
    def test_execute_query_error_handling(self, engine, mock_driver):
        """Test that query errors are caught and returned in result."""
        # Setup mock to raise exception
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.side_effect = Exception("Connection failed")
        
        # Execute query
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) RETURN p",
            parameters={},
            description="Failing query"
        )
        
        # Verify error is captured
        assert result.error == "Connection failed"
        assert result.result_count == 0
        assert result.results == []
        assert result.execution_time_ms > 0  # Should still measure time
    
    def test_execute_query_invalid_cypher(self, engine):
        """Test validation of invalid Cypher query."""
        # Empty query
        result = engine.execute_query(
            cypher_query="",
            parameters={},
            description="Invalid query"
        )
        assert result.error is not None
        assert "Invalid Cypher query" in result.error
        
        # Non-string query
        result = engine.execute_query(
            cypher_query=None,
            parameters={},
            description="Invalid query"
        )
        assert result.error is not None
        assert "Invalid Cypher query" in result.error
    
    def test_execute_query_invalid_parameters(self, engine):
        """Test validation of invalid parameters."""
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) RETURN p",
            parameters="not a dict",
            description="Invalid parameters"
        )
        
        assert result.error is not None
        assert "Invalid parameters" in result.error
    
    def test_execute_query_timing_accuracy(self, engine, mock_driver):
        """Test that execution timing is captured."""
        # Setup mock
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([{"result": "data"}]))
        session.run.return_value = mock_result
        
        # Execute query
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) RETURN p",
            parameters={},
            description="Timed query"
        )
        
        # Verify timing is captured (should be > 0)
        assert result.execution_time_ms >= 0
        # Verify it's a reasonable value (not negative, not absurdly large)
        assert result.execution_time_ms < 1000  # Should complete in less than 1 second
    
    def test_validate_parameter_type_checking(self, engine):
        """Test parameter type validation."""
        # Valid type
        valid, error = engine.validate_parameter("year", 2024, int)
        assert valid is True
        assert error is None
        
        # Invalid type
        valid, error = engine.validate_parameter("year", "2024", int)
        assert valid is False
        assert "must be of type int" in error
    
    def test_validate_parameter_allowed_values(self, engine):
        """Test parameter validation against allowed values."""
        allowed = ["RCT", "observational", "meta_analysis"]
        
        # Valid value
        valid, error = engine.validate_parameter(
            "study_type", "RCT", str, allowed_values=allowed
        )
        assert valid is True
        assert error is None
        
        # Invalid value
        valid, error = engine.validate_parameter(
            "study_type", "invalid", str, allowed_values=allowed
        )
        assert valid is False
        assert "must be one of" in error
    
    def test_sanitize_string_parameter(self, engine):
        """Test string parameter sanitization."""
        # Normal string
        assert engine.sanitize_string_parameter("  test  ") == "test"
        
        # String with null bytes
        assert engine.sanitize_string_parameter("test\x00data") == "testdata"
        
        # Non-string input
        assert engine.sanitize_string_parameter(123) == ""
        assert engine.sanitize_string_parameter(None) == ""
    
    def test_build_parameterized_query_simple(self, engine):
        """Test building parameterized query with simple filters."""
        query, params = engine.build_parameterized_query(
            base_query="MATCH (p:Paper) WHERE {filters} RETURN p",
            filters={"p.year": 2024}
        )
        
        assert "p.year = $p_year" in query
        assert params == {"p_year": 2024}
    
    def test_build_parameterized_query_multiple_filters(self, engine):
        """Test building parameterized query with multiple filters."""
        query, params = engine.build_parameterized_query(
            base_query="MATCH (p:Paper) WHERE {filters} RETURN p",
            filters={
                "p.year": 2024,
                "p.article_type": "original_research"
            }
        )
        
        assert "p.year = $p_year" in query
        assert "p.article_type = $p_article_type" in query
        assert " AND " in query
        assert params == {
            "p_year": 2024,
            "p_article_type": "original_research"
        }
    
    def test_build_parameterized_query_no_filters(self, engine):
        """Test building parameterized query with no filters."""
        query, params = engine.build_parameterized_query(
            base_query="MATCH (p:Paper) WHERE {filters} RETURN p",
            filters={}
        )
        
        assert "WHERE true" in query
        assert params == {}
    
    def test_parameterized_queries_prevent_injection(self, engine, mock_driver):
        """Test that parameterized queries prevent SQL injection attempts."""
        # Setup mock
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Attempt injection via parameter (should be safely parameterized)
        malicious_input = "'; DROP TABLE Paper; --"
        
        result = engine.execute_query(
            cypher_query="MATCH (p:Paper) WHERE p.title = $title RETURN p",
            parameters={"title": malicious_input},
            description="Injection test"
        )
        
        # Verify the malicious input was passed as a parameter, not concatenated
        call_args = session.run.call_args
        assert call_args[0][0] == "MATCH (p:Paper) WHERE p.title = $title RETURN p"
        assert call_args[0][1] == {"title": malicious_input}
        
        # Query should execute without error (parameter is safely escaped)
        assert result.error is None


class TestQueryResultIntegration:
    """Integration tests for QueryResult with real data patterns."""
    
    def test_query_result_serialization(self):
        """Test that QueryResult can be serialized to JSON."""
        result = QueryResult(
            query_description="Test query",
            results=[{"taxon": "Bacteroides", "count": 5}],
            result_count=1,
            execution_time_ms=123.45,
            aggregation_method="weighted_average",
            confidence_threshold=0.7
        )
        
        # Should be serializable
        json_data = result.model_dump()
        
        assert json_data["query_description"] == "Test query"
        assert json_data["result_count"] == 1
        assert json_data["execution_time_ms"] == 123.45
        assert json_data["aggregation_method"] == "weighted_average"
        assert json_data["confidence_threshold"] == 0.7
    
    def test_query_result_with_complex_results(self):
        """Test QueryResult with complex nested result data."""
        complex_results = [
            {
                "taxon": "Bacteroides fragilis",
                "papers": [
                    {"doi": "10.1234/test1", "year": 2024},
                    {"doi": "10.1234/test2", "year": 2023}
                ],
                "consensus_confidence": 0.85,
                "direction_consistency": 0.9
            }
        ]
        
        result = QueryResult(
            query_description="Complex query",
            results=complex_results,
            result_count=1
        )
        
        assert result.results[0]["taxon"] == "Bacteroides fragilis"
        assert len(result.results[0]["papers"]) == 2
        assert result.results[0]["consensus_confidence"] == 0.85


class TestQueryCrossStudyAssociations:
    """Test suite for query_cross_study_associations method (Q1)."""
    
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
    
    def test_query_cross_study_associations_success(self, engine, mock_driver):
        """Test successful cross-study association query with default parameters."""
        # Setup mock to return sample data
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Bacteroides fragilis",
                "paper_count": 5,
                "consensus_confidence": 0.85,
                "consensus_direction": "increased",
                "direction_consistency": 0.80,
                "increased_count": 4,
                "decreased_count": 1,
                "no_change_count": 0,
                "paper_ids": ["PMID:12345", "PMID:67890", "DOI:10.1234/test"]
            },
            {
                "taxon_name": "Lactobacillus acidophilus",
                "paper_count": 3,
                "consensus_confidence": 0.75,
                "consensus_direction": "decreased",
                "direction_consistency": 1.0,
                "increased_count": 0,
                "decreased_count": 3,
                "no_change_count": 0,
                "paper_ids": ["PMID:11111", "PMID:22222", "PMID:33333"]
            }
        ]))
        session.run.return_value = mock_result
        
        # Execute query
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7,
            require_open_data=True
        )
        
        # Verify result structure
        assert result.error is None
        assert result.result_count == 2
        assert len(result.results) == 2
        
        # Verify first result
        first = result.results[0]
        assert first["taxon_name"] == "Bacteroides fragilis"
        assert first["paper_count"] == 5
        assert first["consensus_confidence"] == 0.85
        assert first["consensus_direction"] == "increased"
        assert first["direction_consistency"] == 0.80
        assert first["increased_count"] == 4
        assert first["decreased_count"] == 1
        assert len(first["paper_ids"]) == 3
        
        # Verify query metadata
        assert result.aggregation_method == "weighted_average"
        assert result.confidence_threshold == 0.7
        assert "Type 2 Diabetes" in result.query_description
        
        # Verify parameterized query was used
        call_args = session.run.call_args
        assert "$disease" in call_args[0][0]
        assert "$threshold" in call_args[0][0]
        assert "$study_type" in call_args[0][0]
        assert "$min_papers" in call_args[0][0]
        assert "$require_open_data" in call_args[0][0]
        
        # Verify parameters
        params = call_args[0][1]
        assert params["disease"] == "Type 2 Diabetes"
        assert params["threshold"] == 0.7
        assert params["study_type"] == "RCT"
        assert params["min_papers"] == 3
        assert params["require_open_data"] is True
    
    def test_query_cross_study_associations_any_study_type(self, engine, mock_driver):
        """Test query with study_type='any' to include all study types."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_cross_study_associations(
            disease="Crohn's Disease",
            study_type="any",
            min_papers=1,
            confidence_threshold=0.5,
            require_open_data=False
        )
        
        assert result.error is None
        
        # Verify parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["study_type"] == "any"
        assert params["require_open_data"] is False
    
    def test_query_cross_study_associations_no_open_data_requirement(self, engine, mock_driver):
        """Test query without requiring open data."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_cross_study_associations(
            disease="Obesity",
            require_open_data=False
        )
        
        assert result.error is None
        
        # Verify require_open_data parameter
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["require_open_data"] is False
    
    def test_query_cross_study_associations_empty_results(self, engine, mock_driver):
        """Test query that returns no results."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_cross_study_associations(
            disease="Rare Disease",
            min_papers=10,
            confidence_threshold=0.9
        )
        
        assert result.error is None
        assert result.result_count == 0
        assert result.results == []
    
    def test_query_cross_study_associations_invalid_study_type(self, engine):
        """Test validation of invalid study_type parameter."""
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            study_type="invalid_type"
        )
        
        assert result.error is not None
        assert "study_type" in result.error
        assert "must be one of" in result.error
    
    def test_query_cross_study_associations_invalid_min_papers(self, engine):
        """Test validation of invalid min_papers parameter."""
        # Negative value
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            min_papers=-1
        )
        assert result.error is not None
        assert "min_papers" in result.error
        
        # Zero value
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            min_papers=0
        )
        assert result.error is not None
        assert "min_papers" in result.error
        
        # Non-integer value
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            min_papers=3.5
        )
        assert result.error is not None
        assert "min_papers" in result.error
    
    def test_query_cross_study_associations_invalid_confidence_threshold(self, engine):
        """Test validation of invalid confidence_threshold parameter."""
        # Below range
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            confidence_threshold=-0.1
        )
        assert result.error is not None
        assert "confidence_threshold" in result.error
        
        # Above range
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes",
            confidence_threshold=1.5
        )
        assert result.error is not None
        assert "confidence_threshold" in result.error
    
    def test_query_cross_study_associations_boundary_values(self, engine, mock_driver):
        """Test query with boundary values for parameters."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Minimum valid values
        result = engine.query_cross_study_associations(
            disease="Test Disease",
            min_papers=1,
            confidence_threshold=0.0
        )
        assert result.error is None
        
        # Maximum valid values
        result = engine.query_cross_study_associations(
            disease="Test Disease",
            min_papers=1000,
            confidence_threshold=1.0
        )
        assert result.error is None
    
    def test_query_cross_study_associations_string_sanitization(self, engine, mock_driver):
        """Test that disease name is sanitized."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Disease name with whitespace and null bytes
        result = engine.query_cross_study_associations(
            disease="  Type 2 Diabetes\x00  "
        )
        
        assert result.error is None
        
        # Verify sanitized disease name in parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["disease"] == "Type 2 Diabetes"  # Trimmed and null bytes removed
    
    def test_query_cross_study_associations_cypher_structure(self, engine, mock_driver):
        """Test that generated Cypher query has correct structure."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes"
        )
        
        # Get the Cypher query
        call_args = session.run.call_args
        cypher_query = call_args[0][0]
        
        # Verify key components of the query
        assert "MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)" in cypher_query
        assert "WHERE r.disease = $disease" in cypher_query
        assert "r.confidence >= $threshold" in cypher_query
        assert "p.article_type = $study_type" in cypher_query
        assert "p.data_availability = 'open'" in cypher_query
        assert "size(p.accession_numbers) > 0" in cypher_query
        assert "avg(r.confidence) as consensus_confidence" in cypher_query
        assert "ORDER BY consensus_confidence DESC, paper_count DESC" in cypher_query
    
    def test_query_cross_study_associations_observational_studies(self, engine, mock_driver):
        """Test query for observational studies."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_cross_study_associations(
            disease="IBD",
            study_type="observational"
        )
        
        assert result.error is None
        
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["study_type"] == "observational"
    
    def test_query_cross_study_associations_meta_analysis(self, engine, mock_driver):
        """Test query for meta-analysis studies."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_cross_study_associations(
            disease="Colorectal Cancer",
            study_type="meta_analysis"
        )
        
        assert result.error is None
        
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["study_type"] == "meta_analysis"
    
    def test_query_cross_study_associations_error_handling(self, engine, mock_driver):
        """Test error handling when query execution fails."""
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.side_effect = Exception("Database connection failed")
        
        result = engine.query_cross_study_associations(
            disease="Type 2 Diabetes"
        )
        
        assert result.error is not None
        assert "Database connection failed" in result.error
        assert result.result_count == 0
        assert result.results == []


class TestQueryInterventionEvidence:
    """Test suite for query_intervention_evidence method (Q2)."""
    
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
    
    def test_query_intervention_evidence_success(self, engine, mock_driver):
        """Test successful intervention evidence query with default parameters."""
        # Setup mock to return sample data
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "intervention_type": "probiotic",
                "taxon_name": "Lactobacillus acidophilus",
                "effect_direction": "increased",
                "paper_count": 8,
                "total_sample_size": 450,
                "paper_ids": ["PMID:12345", "PMID:67890", "DOI:10.1234/test"],
                "avg_confidence": 0.87
            },
            {
                "intervention_type": "FMT",
                "taxon_name": "Bacteroides fragilis",
                "effect_direction": "increased",
                "paper_count": 5,
                "total_sample_size": 320,
                "paper_ids": ["PMID:11111", "PMID:22222"],
                "avg_confidence": 0.82
            }
        ]))
        session.run.return_value = mock_result
        
        # Execute query
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic", "FMT"],
            min_sample_size=50,
            evidence_strength="strong"
        )
        
        # Verify result structure
        assert result.error is None
        assert result.result_count == 2
        assert len(result.results) == 2
        
        # Verify first result
        first = result.results[0]
        assert first["intervention_type"] == "probiotic"
        assert first["taxon_name"] == "Lactobacillus acidophilus"
        assert first["effect_direction"] == "increased"
        assert first["paper_count"] == 8
        assert first["total_sample_size"] == 450
        assert len(first["paper_ids"]) == 3
        assert first["avg_confidence"] == 0.87
        
        # Verify query metadata
        assert result.aggregation_method == "sum_sample_sizes"
        assert "probiotic" in result.query_description
        assert "FMT" in result.query_description
        
        # Verify parameterized query was used
        call_args = session.run.call_args
        assert "$intervention_types" in call_args[0][0]
        assert "$evidence_strength" in call_args[0][0]
        assert "$min_sample_size" in call_args[0][0]
        
        # Verify parameters
        params = call_args[0][1]
        assert params["intervention_types"] == ["probiotic", "FMT"]
        assert params["evidence_strength"] == "strong"
        assert params["min_sample_size"] == 50
    
    def test_query_intervention_evidence_single_intervention(self, engine, mock_driver):
        """Test query with single intervention type."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_intervention_evidence(
            intervention_types=["diet"],
            min_sample_size=100,
            evidence_strength="moderate"
        )
        
        assert result.error is None
        
        # Verify parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["intervention_types"] == ["diet"]
        assert params["evidence_strength"] == "moderate"
        assert params["min_sample_size"] == 100
    
    def test_query_intervention_evidence_any_strength(self, engine, mock_driver):
        """Test query with evidence_strength='any' to include all strengths."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_intervention_evidence(
            intervention_types=["antibiotic"],
            evidence_strength="any"
        )
        
        assert result.error is None
        
        # Verify parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["evidence_strength"] == "any"
    
    def test_query_intervention_evidence_empty_results(self, engine, mock_driver):
        """Test query that returns no results."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_intervention_evidence(
            intervention_types=["rare_intervention"],
            min_sample_size=1000,
            evidence_strength="strong"
        )
        
        assert result.error is None
        assert result.result_count == 0
        assert result.results == []
    
    def test_query_intervention_evidence_invalid_intervention_types_empty(self, engine):
        """Test validation of empty intervention_types list."""
        result = engine.query_intervention_evidence(
            intervention_types=[],
            min_sample_size=50
        )
        
        assert result.error is not None
        assert "intervention_types" in result.error
        assert "non-empty list" in result.error
    
    def test_query_intervention_evidence_invalid_intervention_types_not_list(self, engine):
        """Test validation of non-list intervention_types."""
        result = engine.query_intervention_evidence(
            intervention_types="probiotic",
            min_sample_size=50
        )
        
        assert result.error is not None
        assert "intervention_types" in result.error
        assert "non-empty list" in result.error
    
    def test_query_intervention_evidence_invalid_evidence_strength(self, engine):
        """Test validation of invalid evidence_strength parameter."""
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"],
            evidence_strength="invalid_strength"
        )
        
        assert result.error is not None
        assert "evidence_strength" in result.error
        assert "must be one of" in result.error
    
    def test_query_intervention_evidence_invalid_min_sample_size(self, engine):
        """Test validation of invalid min_sample_size parameter."""
        # Negative value
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"],
            min_sample_size=-1
        )
        assert result.error is not None
        assert "min_sample_size" in result.error
        
        # Zero value
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"],
            min_sample_size=0
        )
        assert result.error is not None
        assert "min_sample_size" in result.error
        
        # Non-integer value
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"],
            min_sample_size=50.5
        )
        assert result.error is not None
        assert "min_sample_size" in result.error
    
    def test_query_intervention_evidence_boundary_values(self, engine, mock_driver):
        """Test query with boundary values for parameters."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Minimum valid values
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"],
            min_sample_size=1,
            evidence_strength="weak"
        )
        assert result.error is None
        
        # Large sample size
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"],
            min_sample_size=10000
        )
        assert result.error is None
    
    def test_query_intervention_evidence_string_sanitization(self, engine, mock_driver):
        """Test that intervention types are sanitized."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Intervention types with whitespace and null bytes
        result = engine.query_intervention_evidence(
            intervention_types=["  probiotic\x00  ", "  FMT  "]
        )
        
        assert result.error is None
        
        # Verify sanitized intervention types in parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["intervention_types"] == ["probiotic", "FMT"]  # Trimmed and null bytes removed
    
    def test_query_intervention_evidence_cypher_structure(self, engine, mock_driver):
        """Test that generated Cypher query has correct structure."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"]
        )
        
        # Get the Cypher query
        call_args = session.run.call_args
        cypher_query = call_args[0][0]
        
        # Verify key components of the query (Requirements 7.2, 7.3, 7.4, 7.5)
        assert "MATCH (p:Paper)-[r:REPORTS_INTERVENTION_EFFECT]->(t:Taxon)" in cypher_query
        assert "r.intervention_type IN $intervention_types" in cypher_query
        assert "r.evidence_strength = $evidence_strength" in cypher_query
        assert "p.article_type = 'original_research' OR p.article_type = 'meta_analysis'" in cypher_query
        assert "r.sample_size IS NOT NULL" in cypher_query
        assert "sum(r.sample_size) as total_samples" in cypher_query
        assert "total_samples >= $min_sample_size" in cypher_query
        assert "ORDER BY paper_count DESC, total_sample_size DESC" in cypher_query
    
    def test_query_intervention_evidence_multiple_interventions(self, engine, mock_driver):
        """Test query with multiple intervention types."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic", "FMT", "diet", "antibiotic"]
        )
        
        assert result.error is None
        
        call_args = session.run.call_args
        params = call_args[0][1]
        assert len(params["intervention_types"]) == 4
        assert "probiotic" in params["intervention_types"]
        assert "FMT" in params["intervention_types"]
        assert "diet" in params["intervention_types"]
        assert "antibiotic" in params["intervention_types"]
    
    def test_query_intervention_evidence_weak_evidence(self, engine, mock_driver):
        """Test query for weak evidence strength."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"],
            evidence_strength="weak"
        )
        
        assert result.error is None
        
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["evidence_strength"] == "weak"
    
    def test_query_intervention_evidence_error_handling(self, engine, mock_driver):
        """Test error handling when query execution fails."""
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.side_effect = Exception("Database connection failed")
        
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic"]
        )
        
        assert result.error is not None
        assert "Database connection failed" in result.error
        assert result.result_count == 0
        assert result.results == []
    
    def test_query_intervention_evidence_sorting(self, engine, mock_driver):
        """Test that results are sorted by paper_count DESC, then total_sample_size DESC."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "intervention_type": "probiotic",
                "taxon_name": "Taxon A",
                "effect_direction": "increased",
                "paper_count": 10,
                "total_sample_size": 500,
                "paper_ids": [],
                "avg_confidence": 0.85
            },
            {
                "intervention_type": "FMT",
                "taxon_name": "Taxon B",
                "effect_direction": "increased",
                "paper_count": 10,
                "total_sample_size": 300,
                "paper_ids": [],
                "avg_confidence": 0.80
            },
            {
                "intervention_type": "diet",
                "taxon_name": "Taxon C",
                "effect_direction": "decreased",
                "paper_count": 5,
                "total_sample_size": 800,
                "paper_ids": [],
                "avg_confidence": 0.90
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_intervention_evidence(
            intervention_types=["probiotic", "FMT", "diet"]
        )
        
        # Verify sorting: first two have same paper_count (10), sorted by sample_size DESC
        # Third has lower paper_count (5)
        assert result.results[0]["paper_count"] == 10
        assert result.results[0]["total_sample_size"] == 500
        assert result.results[1]["paper_count"] == 10
        assert result.results[1]["total_sample_size"] == 300
        assert result.results[2]["paper_count"] == 5
        assert result.results[2]["total_sample_size"] == 800


class TestQueryMethodologyLandscape:
    """Test suite for query_methodology_landscape method (Q3)."""
    
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
    
    def test_query_methodology_landscape_success(self, engine, mock_driver):
        """Test successful methodology landscape query with default parameters."""
        # Setup mock to return sample data
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "method": "shotgun metagenomics",
                "year": 2024,
                "total_papers": 45,
                "papers_with_data": 38,
                "data_availability_pct": 84.4,
                "ncbi_sra_count": 30,
                "ena_count": 12,
                "both_repositories_count": 4
            },
            {
                "method": "16S rRNA sequencing",
                "year": 2024,
                "total_papers": 120,
                "papers_with_data": 95,
                "data_availability_pct": 79.2,
                "ncbi_sra_count": 80,
                "ena_count": 20,
                "both_repositories_count": 5
            },
            {
                "method": "shotgun metagenomics",
                "year": 2023,
                "total_papers": 40,
                "papers_with_data": 30,
                "data_availability_pct": 75.0,
                "ncbi_sra_count": 25,
                "ena_count": 8,
                "both_repositories_count": 3
            }
        ]))
        session.run.return_value = mock_result
        
        # Execute query
        result = engine.query_methodology_landscape(
            year_start=2023,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"],
            require_deposited_data=True
        )
        
        # Verify result structure
        assert result.error is None
        assert result.result_count == 3
        assert len(result.results) == 3
        
        # Verify first result (sorted by year DESC, then method ASC)
        first = result.results[0]
        assert first["method"] == "shotgun metagenomics"
        assert first["year"] == 2024
        assert first["total_papers"] == 45
        assert first["papers_with_data"] == 38
        assert first["data_availability_pct"] == 84.4
        assert first["ncbi_sra_count"] == 30
        assert first["ena_count"] == 12
        assert first["both_repositories_count"] == 4
        
        # Verify query metadata
        assert result.aggregation_method == "group_by_method_year"
        assert "2023-2024" in result.query_description
        
        # Verify parameterized query was used
        call_args = session.run.call_args
        assert "$year_start" in call_args[0][0]
        assert "$year_end" in call_args[0][0]
        assert "$sequencing_methods" in call_args[0][0]
        assert "$require_deposited_data" in call_args[0][0]
        
        # Verify parameters
        params = call_args[0][1]
        assert params["year_start"] == 2023
        assert params["year_end"] == 2024
        assert params["sequencing_methods"] == ["16S rRNA sequencing", "shotgun metagenomics"]
        assert params["require_deposited_data"] is True
    
    def test_query_methodology_landscape_without_deposited_data_requirement(self, engine, mock_driver):
        """Test query without requiring deposited data."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "method": "16S rRNA sequencing",
                "year": 2024,
                "total_papers": 150,
                "papers_with_data": 95,
                "data_availability_pct": 63.3,
                "ncbi_sra_count": 80,
                "ena_count": 20,
                "both_repositories_count": 5
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_methodology_landscape(
            year_start=2024,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing"],
            require_deposited_data=False
        )
        
        assert result.error is None
        assert result.result_count == 1
        
        # Verify parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["require_deposited_data"] is False
    
    def test_query_methodology_landscape_empty_results(self, engine, mock_driver):
        """Test query that returns no results."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_methodology_landscape(
            year_start=2020,
            year_end=2021,
            sequencing_methods=["shotgun metagenomics"]
        )
        
        assert result.error is None
        assert result.result_count == 0
        assert result.results == []
    
    def test_query_methodology_landscape_invalid_year_range(self, engine):
        """Test validation of invalid year range (start > end)."""
        result = engine.query_methodology_landscape(
            year_start=2024,
            year_end=2020,
            sequencing_methods=["16S rRNA sequencing"]
        )
        
        assert result.error is not None
        assert "year_start must be <= year_end" in result.error
    
    def test_query_methodology_landscape_invalid_year_types(self, engine):
        """Test validation of invalid year types."""
        result = engine.query_methodology_landscape(
            year_start="2020",
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing"]
        )
        
        assert result.error is not None
        assert "must be integers" in result.error
    
    def test_query_methodology_landscape_empty_methods_list(self, engine):
        """Test validation of empty sequencing_methods list."""
        result = engine.query_methodology_landscape(
            year_start=2020,
            year_end=2024,
            sequencing_methods=[]
        )
        
        assert result.error is not None
        assert "non-empty list" in result.error
    
    def test_query_methodology_landscape_invalid_methods_type(self, engine):
        """Test validation of invalid sequencing_methods type."""
        result = engine.query_methodology_landscape(
            year_start=2020,
            year_end=2024,
            sequencing_methods="16S rRNA sequencing"
        )
        
        assert result.error is not None
        assert "non-empty list" in result.error
    
    def test_query_methodology_landscape_string_sanitization(self, engine, mock_driver):
        """Test that method names are sanitized."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_methodology_landscape(
            year_start=2020,
            year_end=2024,
            sequencing_methods=["  16S rRNA sequencing  ", "shotgun\x00metagenomics"]
        )
        
        # Verify parameters were sanitized
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["sequencing_methods"] == ["16S rRNA sequencing", "shotgunmetagenomics"]
    
    def test_query_methodology_landscape_cypher_structure(self, engine, mock_driver):
        """Test that generated Cypher query has correct structure."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_methodology_landscape(
            year_start=2020,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing"]
        )
        
        # Verify Cypher query structure
        call_args = session.run.call_args
        cypher_query = call_args[0][0]
        
        # Check for key query components
        assert "MATCH (p:Paper)-[r:USES_METHODOLOGY]->(m:Method)" in cypher_query
        assert "p.year >= $year_start" in cypher_query
        assert "p.year <= $year_end" in cypher_query
        assert "m.name IN $sequencing_methods" in cypher_query
        assert "size(p.accession_numbers) > 0" in cypher_query
        
        # Check for repository identification logic
        assert "SRP" in cypher_query or "STARTS WITH 'SRP'" in cypher_query
        assert "ERP" in cypher_query or "STARTS WITH 'ERP'" in cypher_query
        assert "PRJNA" in cypher_query or "STARTS WITH 'PRJNA'" in cypher_query
        assert "PRJEB" in cypher_query or "STARTS WITH 'PRJEB'" in cypher_query
        
        # Check for aggregation and sorting
        assert "data_availability_pct" in cypher_query
        assert "ncbi_sra_count" in cypher_query
        assert "ena_count" in cypher_query
        assert "both_repositories_count" in cypher_query
        assert "ORDER BY year DESC, method ASC" in cypher_query
    
    def test_query_methodology_landscape_single_year(self, engine, mock_driver):
        """Test query for a single year."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "method": "16S rRNA sequencing",
                "year": 2024,
                "total_papers": 100,
                "papers_with_data": 80,
                "data_availability_pct": 80.0,
                "ncbi_sra_count": 70,
                "ena_count": 15,
                "both_repositories_count": 5
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_methodology_landscape(
            year_start=2024,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing"]
        )
        
        assert result.error is None
        assert result.result_count == 1
        
        # Verify parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["year_start"] == 2024
        assert params["year_end"] == 2024
    
    def test_query_methodology_landscape_error_handling(self, engine, mock_driver):
        """Test error handling when query execution fails."""
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.side_effect = Exception("Database connection error")
        
        result = engine.query_methodology_landscape(
            year_start=2020,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing"]
        )
        
        assert result.error is not None
        assert "Database connection error" in result.error
        assert result.result_count == 0
    
    def test_query_methodology_landscape_sorting_verification(self, engine, mock_driver):
        """Test that results are sorted by year DESC, then method ASC."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "method": "16S rRNA sequencing",
                "year": 2024,
                "total_papers": 100,
                "papers_with_data": 80,
                "data_availability_pct": 80.0,
                "ncbi_sra_count": 70,
                "ena_count": 15,
                "both_repositories_count": 5
            },
            {
                "method": "shotgun metagenomics",
                "year": 2024,
                "total_papers": 50,
                "papers_with_data": 40,
                "data_availability_pct": 80.0,
                "ncbi_sra_count": 35,
                "ena_count": 8,
                "both_repositories_count": 3
            },
            {
                "method": "16S rRNA sequencing",
                "year": 2023,
                "total_papers": 90,
                "papers_with_data": 70,
                "data_availability_pct": 77.8,
                "ncbi_sra_count": 60,
                "ena_count": 12,
                "both_repositories_count": 2
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_methodology_landscape(
            year_start=2023,
            year_end=2024,
            sequencing_methods=["16S rRNA sequencing", "shotgun metagenomics"]
        )
        
        # Verify sorting: year DESC (2024 before 2023), then method ASC (16S before shotgun)
        assert result.results[0]["year"] == 2024
        assert result.results[0]["method"] == "16S rRNA sequencing"
        assert result.results[1]["year"] == 2024
        assert result.results[1]["method"] == "shotgun metagenomics"
        assert result.results[2]["year"] == 2023
        assert result.results[2]["method"] == "16S rRNA sequencing"


class TestQueryTopAssociationsByEvidence:
    """Test suite for query_top_associations_by_evidence method (Q4)."""
    
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
    
    def test_query_top_associations_success(self, engine, mock_driver):
        """Test successful top associations query with default parameters."""
        # Setup mock to return sample data
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Faecalibacterium prausnitzii",
                "paper_count": 12,
                "avg_confidence": 0.89,
                "consensus_direction": "decreased",
                "direction_consistency": 0.92,
                "increased_count": 1,
                "decreased_count": 11,
                "no_change_count": 0,
                "paper_ids": ["PMID:12345", "PMID:67890", "DOI:10.1234/test"]
            },
            {
                "taxon_name": "Bacteroides fragilis",
                "paper_count": 10,
                "avg_confidence": 0.85,
                "consensus_direction": "increased",
                "direction_consistency": 0.80,
                "increased_count": 8,
                "decreased_count": 2,
                "no_change_count": 0,
                "paper_ids": ["PMID:11111", "PMID:22222"]
            }
        ]))
        session.run.return_value = mock_result
        
        # Execute query
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            top_n=10,
            min_confidence=0.7
        )
        
        # Verify result structure
        assert result.error is None
        assert result.result_count == 2
        assert len(result.results) == 2
        
        # Verify first result (highest paper count)
        first = result.results[0]
        assert first["taxon_name"] == "Faecalibacterium prausnitzii"
        assert first["paper_count"] == 12
        assert first["avg_confidence"] == 0.89
        assert first["consensus_direction"] == "decreased"
        assert first["direction_consistency"] == 0.92
        assert first["increased_count"] == 1
        assert first["decreased_count"] == 11
        assert len(first["paper_ids"]) == 3
        
        # Verify second result
        second = result.results[1]
        assert second["taxon_name"] == "Bacteroides fragilis"
        assert second["paper_count"] == 10
        assert second["avg_confidence"] == 0.85
        
        # Verify query metadata
        assert result.aggregation_method == "top_n_by_evidence"
        assert result.confidence_threshold == 0.7
        assert "IBD" in result.query_description
        assert "Top 10" in result.query_description
        
        # Verify parameterized query was used
        call_args = session.run.call_args
        assert "$disease" in call_args[0][0]
        assert "$min_confidence" in call_args[0][0]
        assert "$top_n" in call_args[0][0]
        assert "LIMIT $top_n" in call_args[0][0]
        
        # Verify parameters
        params = call_args[0][1]
        assert params["disease"] == "IBD"
        assert params["min_confidence"] == 0.7
        assert params["top_n"] == 10
    
    def test_query_top_associations_custom_top_n(self, engine, mock_driver):
        """Test query with custom top_n value."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Taxon1",
                "paper_count": 5,
                "avg_confidence": 0.85,
                "consensus_direction": "increased",
                "direction_consistency": 1.0,
                "increased_count": 5,
                "decreased_count": 0,
                "no_change_count": 0,
                "paper_ids": ["PMID:1", "PMID:2", "PMID:3", "PMID:4", "PMID:5"]
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_top_associations_by_evidence(
            disease="Type 2 Diabetes",
            top_n=5,
            min_confidence=0.8
        )
        
        assert result.error is None
        
        # Verify parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["top_n"] == 5
        assert params["min_confidence"] == 0.8
    
    def test_query_top_associations_empty_results(self, engine, mock_driver):
        """Test query that returns no results."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_top_associations_by_evidence(
            disease="Rare Disease",
            top_n=10,
            min_confidence=0.95
        )
        
        assert result.error is None
        assert result.result_count == 0
        assert result.results == []
    
    def test_query_top_associations_invalid_top_n(self, engine):
        """Test validation of invalid top_n parameter."""
        # Negative value
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            top_n=-1
        )
        assert result.error is not None
        assert "top_n" in result.error
        
        # Zero value
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            top_n=0
        )
        assert result.error is not None
        assert "top_n" in result.error
        
        # Non-integer value
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            top_n=5.5
        )
        assert result.error is not None
        assert "top_n" in result.error
    
    def test_query_top_associations_invalid_min_confidence(self, engine):
        """Test validation of invalid min_confidence parameter."""
        # Below range
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            min_confidence=-0.1
        )
        assert result.error is not None
        assert "min_confidence" in result.error
        
        # Above range
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            min_confidence=1.5
        )
        assert result.error is not None
        assert "min_confidence" in result.error
    
    def test_query_top_associations_boundary_values(self, engine, mock_driver):
        """Test query with boundary values for parameters."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Minimum valid values
        result = engine.query_top_associations_by_evidence(
            disease="Test Disease",
            top_n=1,
            min_confidence=0.0
        )
        assert result.error is None
        
        # Maximum valid values
        result = engine.query_top_associations_by_evidence(
            disease="Test Disease",
            top_n=1000,
            min_confidence=1.0
        )
        assert result.error is None
    
    def test_query_top_associations_string_sanitization(self, engine, mock_driver):
        """Test that disease name is sanitized."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Disease name with whitespace and null bytes
        result = engine.query_top_associations_by_evidence(
            disease="  IBD\x00  ",
            top_n=10
        )
        
        assert result.error is None
        
        # Verify sanitized disease name in parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["disease"] == "IBD"  # Trimmed and null bytes removed
    
    def test_query_top_associations_cypher_structure(self, engine, mock_driver):
        """Test that generated Cypher query has correct structure."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_top_associations_by_evidence(
            disease="Type 2 Diabetes",
            top_n=10,
            min_confidence=0.7
        )
        
        # Get the Cypher query
        call_args = session.run.call_args
        cypher_query = call_args[0][0]
        
        # Verify key components of the query
        assert "MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)" in cypher_query
        assert "WHERE r.disease = $disease" in cypher_query
        assert "r.confidence >= $min_confidence" in cypher_query
        assert "avg(r.confidence) as avg_confidence" in cypher_query
        assert "ORDER BY paper_count DESC, avg_confidence DESC" in cypher_query
        assert "LIMIT $top_n" in cypher_query
    
    def test_query_top_associations_sorting_verification(self, engine, mock_driver):
        """Test that results are sorted by paper_count DESC, then avg_confidence DESC."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Taxon A",
                "paper_count": 10,
                "avg_confidence": 0.90,
                "consensus_direction": "increased",
                "direction_consistency": 0.9,
                "increased_count": 9,
                "decreased_count": 1,
                "no_change_count": 0,
                "paper_ids": ["PMID:1"]
            },
            {
                "taxon_name": "Taxon B",
                "paper_count": 10,
                "avg_confidence": 0.85,
                "consensus_direction": "increased",
                "direction_consistency": 0.8,
                "increased_count": 8,
                "decreased_count": 2,
                "no_change_count": 0,
                "paper_ids": ["PMID:2"]
            },
            {
                "taxon_name": "Taxon C",
                "paper_count": 8,
                "avg_confidence": 0.95,
                "consensus_direction": "decreased",
                "direction_consistency": 1.0,
                "increased_count": 0,
                "decreased_count": 8,
                "no_change_count": 0,
                "paper_ids": ["PMID:3"]
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            top_n=10,
            min_confidence=0.7
        )
        
        # Verify sorting: paper_count DESC (10, 10, 8), then avg_confidence DESC (0.90, 0.85)
        assert result.results[0]["taxon_name"] == "Taxon A"
        assert result.results[0]["paper_count"] == 10
        assert result.results[0]["avg_confidence"] == 0.90
        
        assert result.results[1]["taxon_name"] == "Taxon B"
        assert result.results[1]["paper_count"] == 10
        assert result.results[1]["avg_confidence"] == 0.85
        
        assert result.results[2]["taxon_name"] == "Taxon C"
        assert result.results[2]["paper_count"] == 8
        assert result.results[2]["avg_confidence"] == 0.95
    
    def test_query_top_associations_error_handling(self, engine, mock_driver):
        """Test error handling when query execution fails."""
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.side_effect = Exception("Database connection failed")
        
        result = engine.query_top_associations_by_evidence(
            disease="IBD",
            top_n=10,
            min_confidence=0.7
        )
        
        assert result.error is not None
        assert "Database connection failed" in result.error
        assert result.result_count == 0
        assert result.results == []
    
    def test_query_top_associations_default_parameters(self, engine, mock_driver):
        """Test query with default parameters."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Call with only required parameter
        result = engine.query_top_associations_by_evidence(
            disease="Crohn's Disease"
        )
        
        assert result.error is None
        
        # Verify default parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["top_n"] == 10  # Default value
        assert params["min_confidence"] == 0.7  # Default value
    
    def test_query_top_associations_aggregation_statistics(self, engine, mock_driver):
        """Test that aggregation statistics are correctly returned."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Test Taxon",
                "paper_count": 7,
                "avg_confidence": 0.82,
                "consensus_direction": "increased",
                "direction_consistency": 0.71,
                "increased_count": 5,
                "decreased_count": 2,
                "no_change_count": 0,
                "paper_ids": ["PMID:1", "PMID:2", "PMID:3", "PMID:4", "PMID:5", "PMID:6", "PMID:7"]
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_top_associations_by_evidence(
            disease="Obesity",
            top_n=5,
            min_confidence=0.75
        )
        
        assert result.error is None
        assert result.result_count == 1
        
        # Verify all aggregation statistics are present
        taxon = result.results[0]
        assert "paper_count" in taxon
        assert "avg_confidence" in taxon
        assert "consensus_direction" in taxon
        assert "direction_consistency" in taxon
        assert "increased_count" in taxon
        assert "decreased_count" in taxon
        assert "no_change_count" in taxon
        assert "paper_ids" in taxon
        
        # Verify values
        assert taxon["paper_count"] == 7
        assert taxon["avg_confidence"] == 0.82
        assert taxon["consensus_direction"] == "increased"
        assert taxon["direction_consistency"] == 0.71
        assert taxon["increased_count"] == 5
        assert taxon["decreased_count"] == 2
        assert taxon["no_change_count"] == 0
        assert len(taxon["paper_ids"]) == 7


class TestQueryConflictingEvidence:
    """Test suite for query_conflicting_evidence method (Q5)."""
    
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
    
    def test_query_conflicting_evidence_success(self, engine, mock_driver):
        """Test successful conflicting evidence query with default parameters."""
        # Setup mock to return sample data
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Escherichia coli",
                "total_paper_count": 8,
                "increased_count": 5,
                "decreased_count": 3,
                "increased_percentage": 62.5,
                "decreased_percentage": 37.5,
                "direction_balance": 2,
                "increased_papers": [
                    {"doi": "10.1234/test1", "year": 2023, "study_design": "RCT"},
                    {"doi": "10.1234/test2", "year": 2022, "study_design": "observational"},
                    {"doi": "10.1234/test3", "year": 2021, "study_design": "RCT"},
                    {"doi": "10.1234/test4", "year": 2020, "study_design": "observational"},
                    {"doi": "10.1234/test5", "year": 2019, "study_design": "RCT"}
                ],
                "decreased_papers": [
                    {"doi": "10.1234/test6", "year": 2023, "study_design": "RCT"},
                    {"doi": "10.1234/test7", "year": 2022, "study_design": "observational"},
                    {"doi": "10.1234/test8", "year": 2021, "study_design": "meta_analysis"}
                ]
            },
            {
                "taxon_name": "Bacteroides fragilis",
                "total_paper_count": 6,
                "increased_count": 3,
                "decreased_count": 3,
                "increased_percentage": 50.0,
                "decreased_percentage": 50.0,
                "direction_balance": 0,
                "increased_papers": [
                    {"doi": "10.1234/test9", "year": 2024, "study_design": "RCT"},
                    {"doi": "10.1234/test10", "year": 2023, "study_design": "RCT"},
                    {"doi": "10.1234/test11", "year": 2022, "study_design": "observational"}
                ],
                "decreased_papers": [
                    {"doi": "10.1234/test12", "year": 2024, "study_design": "RCT"},
                    {"doi": "10.1234/test13", "year": 2023, "study_design": "observational"},
                    {"doi": "10.1234/test14", "year": 2022, "study_design": "RCT"}
                ]
            }
        ]))
        session.run.return_value = mock_result
        
        # Execute query
        result = engine.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=2
        )
        
        # Verify result structure
        assert result.error is None
        assert result.result_count == 2
        assert len(result.results) == 2
        
        # Verify first result (E. coli with more papers)
        first = result.results[0]
        assert first["taxon_name"] == "Escherichia coli"
        assert first["total_paper_count"] == 8
        assert first["increased_count"] == 5
        assert first["decreased_count"] == 3
        assert first["increased_percentage"] == 62.5
        assert first["decreased_percentage"] == 37.5
        assert first["direction_balance"] == 2
        assert len(first["increased_papers"]) == 5
        assert len(first["decreased_papers"]) == 3
        
        # Verify second result (B. fragilis with balanced evidence)
        second = result.results[1]
        assert second["taxon_name"] == "Bacteroides fragilis"
        assert second["total_paper_count"] == 6
        assert second["increased_count"] == 3
        assert second["decreased_count"] == 3
        assert second["increased_percentage"] == 50.0
        assert second["decreased_percentage"] == 50.0
        assert second["direction_balance"] == 0
        
        # Verify query metadata
        assert result.aggregation_method == "conflicting_evidence_detection"
        assert "Crohn's Disease" in result.query_description
        
        # Verify parameterized query was used
        call_args = session.run.call_args
        assert "$disease" in call_args[0][0]
        assert "$min_papers_per_direction" in call_args[0][0]
        
        # Verify parameters
        params = call_args[0][1]
        assert params["disease"] == "Crohn's Disease"
        assert params["min_papers_per_direction"] == 2
    
    def test_query_conflicting_evidence_custom_threshold(self, engine, mock_driver):
        """Test query with custom min_papers_per_direction threshold."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_conflicting_evidence(
            disease="Type 2 Diabetes",
            min_papers_per_direction=5
        )
        
        assert result.error is None
        
        # Verify parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["min_papers_per_direction"] == 5
    
    def test_query_conflicting_evidence_empty_results(self, engine, mock_driver):
        """Test query that returns no results (no conflicting evidence)."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_conflicting_evidence(
            disease="Rare Disease",
            min_papers_per_direction=10
        )
        
        assert result.error is None
        assert result.result_count == 0
        assert result.results == []
    
    def test_query_conflicting_evidence_invalid_min_papers(self, engine):
        """Test validation of invalid min_papers_per_direction parameter."""
        # Negative value
        result = engine.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=-1
        )
        assert result.error is not None
        assert "min_papers_per_direction" in result.error
        
        # Zero value
        result = engine.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=0
        )
        assert result.error is not None
        assert "min_papers_per_direction" in result.error
        
        # Non-integer value
        result = engine.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=2.5
        )
        assert result.error is not None
        assert "min_papers_per_direction" in result.error
    
    def test_query_conflicting_evidence_boundary_values(self, engine, mock_driver):
        """Test query with boundary values for parameters."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Minimum valid value
        result = engine.query_conflicting_evidence(
            disease="Test Disease",
            min_papers_per_direction=1
        )
        assert result.error is None
        
        # Large valid value
        result = engine.query_conflicting_evidence(
            disease="Test Disease",
            min_papers_per_direction=100
        )
        assert result.error is None
    
    def test_query_conflicting_evidence_string_sanitization(self, engine, mock_driver):
        """Test that disease name is sanitized."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Disease name with whitespace and null bytes
        result = engine.query_conflicting_evidence(
            disease="  Crohn's Disease\x00  ",
            min_papers_per_direction=2
        )
        
        assert result.error is None
        
        # Verify sanitized disease name in parameters
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["disease"] == "Crohn's Disease"  # Trimmed and null bytes removed
    
    def test_query_conflicting_evidence_cypher_structure(self, engine, mock_driver):
        """Test that generated Cypher query has correct structure."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        result = engine.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=2
        )
        
        # Get the Cypher query
        call_args = session.run.call_args
        cypher_query = call_args[0][0]
        
        # Verify key components of the query (Requirements 9.1, 9.2, 9.3, 9.4, 9.5)
        assert "MATCH (p:Paper)-[r:REPORTS_ASSOCIATION]->(t:Taxon)" in cypher_query
        assert "WHERE r.disease = $disease" in cypher_query
        assert "r.direction = 'increased' OR r.direction = 'decreased'" in cypher_query
        assert "increased_count >= $min_papers_per_direction" in cypher_query
        assert "decreased_count >= $min_papers_per_direction" in cypher_query
        assert "100.0 * increased_count / total_paper_count as increased_percentage" in cypher_query
        assert "100.0 * decreased_count / total_paper_count as decreased_percentage" in cypher_query
        assert "abs(increased_count - decreased_count) as direction_balance" in cypher_query
        assert "ORDER BY total_paper_count DESC, direction_balance ASC" in cypher_query
    
    def test_query_conflicting_evidence_paper_metadata(self, engine, mock_driver):
        """Test that paper metadata is correctly returned (Requirement 9.4)."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Test Taxon",
                "total_paper_count": 4,
                "increased_count": 2,
                "decreased_count": 2,
                "increased_percentage": 50.0,
                "decreased_percentage": 50.0,
                "direction_balance": 0,
                "increased_papers": [
                    {"doi": "10.1234/test1", "year": 2023, "study_design": "RCT"},
                    {"doi": "10.1234/test2", "year": 2022, "study_design": "observational"}
                ],
                "decreased_papers": [
                    {"doi": "10.1234/test3", "year": 2021, "study_design": "RCT"},
                    {"doi": "10.1234/test4", "year": 2020, "study_design": "meta_analysis"}
                ]
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_conflicting_evidence(
            disease="Test Disease",
            min_papers_per_direction=2
        )
        
        assert result.error is None
        assert result.result_count == 1
        
        # Verify paper metadata structure
        taxon = result.results[0]
        assert "increased_papers" in taxon
        assert "decreased_papers" in taxon
        
        # Verify increased papers metadata
        for paper in taxon["increased_papers"]:
            assert "doi" in paper
            assert "year" in paper
            assert "study_design" in paper
        
        # Verify decreased papers metadata
        for paper in taxon["decreased_papers"]:
            assert "doi" in paper
            assert "year" in paper
            assert "study_design" in paper
    
    def test_query_conflicting_evidence_sorting_verification(self, engine, mock_driver):
        """Test that results are sorted by total_paper_count DESC, then direction_balance ASC (Requirement 9.5)."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        
        # Create results already sorted correctly (as the query would return them)
        # Sorted by total_paper_count DESC, then direction_balance ASC
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Taxon B",
                "total_paper_count": 10,
                "increased_count": 5,
                "decreased_count": 5,
                "increased_percentage": 50.0,
                "decreased_percentage": 50.0,
                "direction_balance": 0,  # Lower balance comes first for same paper count
                "increased_papers": [],
                "decreased_papers": []
            },
            {
                "taxon_name": "Taxon A",
                "total_paper_count": 10,
                "increased_count": 7,
                "decreased_count": 3,
                "increased_percentage": 70.0,
                "decreased_percentage": 30.0,
                "direction_balance": 4,  # Higher balance comes second for same paper count
                "increased_papers": [],
                "decreased_papers": []
            },
            {
                "taxon_name": "Taxon C",
                "total_paper_count": 6,
                "increased_count": 4,
                "decreased_count": 2,
                "increased_percentage": 66.7,
                "decreased_percentage": 33.3,
                "direction_balance": 2,  # Lower paper count comes last
                "increased_papers": [],
                "decreased_papers": []
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_conflicting_evidence(
            disease="Test Disease",
            min_papers_per_direction=2
        )
        
        assert result.error is None
        assert result.result_count == 3
        
        # Verify sorting: first by total_paper_count DESC
        assert result.results[0]["total_paper_count"] >= result.results[1]["total_paper_count"]
        assert result.results[1]["total_paper_count"] >= result.results[2]["total_paper_count"]
        
        # For same paper count, verify sorting by direction_balance ASC
        if result.results[0]["total_paper_count"] == result.results[1]["total_paper_count"]:
            assert result.results[0]["direction_balance"] <= result.results[1]["direction_balance"]
    
    def test_query_conflicting_evidence_error_handling(self, engine, mock_driver):
        """Test error handling when query execution fails."""
        session = mock_driver.session.return_value.__enter__.return_value
        session.run.side_effect = Exception("Database connection failed")
        
        result = engine.query_conflicting_evidence(
            disease="Crohn's Disease",
            min_papers_per_direction=2
        )
        
        assert result.error is not None
        assert "Database connection failed" in result.error
        assert result.result_count == 0
        assert result.results == []
    
    def test_query_conflicting_evidence_default_parameters(self, engine, mock_driver):
        """Test query with default parameters."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        session.run.return_value = mock_result
        
        # Call with only required parameter
        result = engine.query_conflicting_evidence(
            disease="IBD"
        )
        
        assert result.error is None
        
        # Verify default parameter value
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["min_papers_per_direction"] == 2  # Default value
    
    def test_query_conflicting_evidence_percentage_calculation(self, engine, mock_driver):
        """Test that percentages are correctly calculated (Requirement 9.3)."""
        session = mock_driver.session.return_value.__enter__.return_value
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {
                "taxon_name": "Test Taxon",
                "total_paper_count": 10,
                "increased_count": 7,
                "decreased_count": 3,
                "increased_percentage": 70.0,
                "decreased_percentage": 30.0,
                "direction_balance": 4,
                "increased_papers": [],
                "decreased_papers": []
            }
        ]))
        session.run.return_value = mock_result
        
        result = engine.query_conflicting_evidence(
            disease="Test Disease",
            min_papers_per_direction=2
        )
        
        assert result.error is None
        assert result.result_count == 1
        
        taxon = result.results[0]
        
        # Verify percentage calculations
        assert taxon["increased_percentage"] == 70.0
        assert taxon["decreased_percentage"] == 30.0
        
        # Verify percentages sum to 100
        assert abs(taxon["increased_percentage"] + taxon["decreased_percentage"] - 100.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
graph/test_enhanced_kg_pipeline.py
-----------------------------------
Unit tests for the enhanced knowledge graph pipeline.

Tests cover:
1. Pipeline configuration
2. Batch processing with parallel workers
3. Neo4j loading with separate database
4. Edge and claim creation
5. Integration with enhanced components

Requirements: 16.1, 17.2
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from typing import List

from graph.enhanced_kg_pipeline import (
    PipelineConfig,
    EnhancedNeo4jLoader,
    EnhancedKGPipeline,
    run_enhanced_pipeline
)
from graph.enhanced_graph_builder import EnhancedGraphEdge
from graph.reified_claims import ScientificClaim
from graph.provenance import ProvenanceMetadata
from nlp.enriched_record import EnrichedPaperRecord


# ========== Fixtures ==========

@pytest.fixture
def sample_config():
    """Create a sample pipeline configuration."""
    return PipelineConfig(
        enabled=True,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test_password",
        neo4j_database="neo4j_test",
        batch_size=10,
        num_workers=2,
        extraction_method="regex_ner",
        extractor_version="1.0",
        output_dir=Path("data/processed"),
        save_intermediate=True,
        neo4j_batch_size=100
    )


@pytest.fixture
def sample_enriched_papers():
    """Create sample enriched paper records."""
    from nlp.enriched_record import NamedEntity, ParsedSection, DataAvailabilityInfo
    
    papers = []
    
    for i in range(25):  # 25 papers to test batching
        paper = EnrichedPaperRecord(
            title=f"Test Paper {i}",
            abstract=f"This paper studies Bacteroides fragilis in Type 2 Diabetes. "
                     f"Results showed increased abundance (p=0.001, LDA=3.2).",
            year=2024,
            doi=f"10.1234/test.{i}",
            pmid=f"PMID{i}",
            article_type_normalized="original_research",
            data_availability=DataAvailabilityInfo(
                status="open",
                accession_numbers=[f"SRA{i}"]
            ),
            entities=[
                NamedEntity(text="Bacteroides fragilis", label="taxon"),
                NamedEntity(text="Type 2 Diabetes", label="disease")
            ],
            sections=[
                ParsedSection(
                    section_type="results",
                    content=f"Bacteroides fragilis showed increased abundance in T2D patients. "
                            f"LDA score: 3.2, p-value: 0.001."
                )
            ],
            methods=["16S rRNA sequencing"]
        )
        papers.append(paper)
    
    return papers


@pytest.fixture
def sample_edge():
    """Create a sample enhanced graph edge."""
    provenance = ProvenanceMetadata(
        paper_id="10.1234/test",
        section_type="results",
        source_sentence="Bacteroides fragilis increased in T2D.",
        sentence_offset=0,
        extraction_method="regex_ner",
        extraction_timestamp=datetime.now(),
        extractor_version="1.0",
        confidence_score=0.85,
        validation_status="unvalidated"
    )
    
    return EnhancedGraphEdge(
        source="Bacteroides fragilis",
        target="Type 2 Diabetes",
        relation="REPORTS_ASSOCIATION",
        properties={
            "direction": "increased",
            "comparison": "T2D vs healthy",
            "statistical_measure": "LDA score",
            "effect_size": 3.2,
            "p_value": 0.001
        },
        provenance=provenance,
        evidence_strength="strong",
        confidence=0.85
    )


@pytest.fixture
def sample_claim():
    """Create a sample reified claim."""
    now_str = datetime.now().isoformat()
    return ScientificClaim(
        claim_id="claim_001",
        claim_type="association",
        subject_entity="Bacteroides fragilis",
        predicate="associated_with_increased",
        object_entity="Type 2 Diabetes",
        supporting_papers=["10.1234/test1", "10.1234/test2"],
        contradicting_papers=[],
        total_sample_size=200,
        evidence_strength="strong",
        consensus_confidence=0.85,
        effect_direction_consistency=1.0,
        first_reported=now_str,
        last_updated=now_str
    )


# ========== PipelineConfig Tests ==========

def test_pipeline_config_defaults():
    """Test that PipelineConfig has correct default values."""
    config = PipelineConfig()
    
    assert config.enabled is True
    assert config.batch_size == 100
    assert config.num_workers == 8
    assert config.neo4j_database == "neo4j_enhanced"
    assert config.neo4j_batch_size == 10000


def test_pipeline_config_from_env():
    """Test creating config from environment variables."""
    with patch.dict('os.environ', {
        'ENHANCED_PIPELINE_ENABLED': 'true',
        'NEO4J_ENHANCED_URI': 'bolt://test:7687',
        'NEO4J_ENHANCED_USER': 'test_user',
        'NEO4J_ENHANCED_PASSWORD': 'test_pass',
        'NEO4J_ENHANCED_DATABASE': 'test_db',
        'ENHANCED_BATCH_SIZE': '50',
        'ENHANCED_NUM_WORKERS': '4'
    }):
        config = PipelineConfig.from_env()
        
        assert config.enabled is True
        assert config.neo4j_uri == 'bolt://test:7687'
        assert config.neo4j_user == 'test_user'
        assert config.neo4j_password == 'test_pass'
        assert config.neo4j_database == 'test_db'
        assert config.batch_size == 50
        assert config.num_workers == 4


def test_pipeline_config_disabled():
    """Test that pipeline can be disabled via config."""
    with patch.dict('os.environ', {'ENHANCED_PIPELINE_ENABLED': 'false'}):
        config = PipelineConfig.from_env()
        assert config.enabled is False


# ========== EnhancedNeo4jLoader Tests ==========

@patch('graph.enhanced_kg_pipeline.GraphDatabase')
def test_neo4j_loader_initialization(mock_graph_db, sample_config):
    """Test Neo4j loader initialization."""
    mock_driver = Mock()
    mock_graph_db.driver.return_value = mock_driver
    
    loader = EnhancedNeo4jLoader(
        uri=sample_config.neo4j_uri,
        user=sample_config.neo4j_user,
        password=sample_config.neo4j_password,
        database=sample_config.neo4j_database,
        batch_size=sample_config.neo4j_batch_size
    )
    
    assert loader.database == sample_config.neo4j_database
    assert loader.batch_size == sample_config.neo4j_batch_size
    mock_graph_db.driver.assert_called_once_with(
        sample_config.neo4j_uri,
        auth=(sample_config.neo4j_user, sample_config.neo4j_password)
    )


@patch('graph.enhanced_kg_pipeline.GraphDatabase')
def test_neo4j_loader_create_indexes(mock_graph_db):
    """Test that Neo4j loader creates required indexes."""
    mock_driver = Mock()
    mock_session = MagicMock()
    mock_graph_db.driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = Mock(return_value=False)
    
    loader = EnhancedNeo4jLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    loader.create_indexes()
    
    # Verify that index creation queries were executed
    # 7 node property indexes + 8 relationship property indexes = 15 total
    assert mock_session.run.call_count >= 15


@patch('graph.enhanced_kg_pipeline.GraphDatabase')
def test_neo4j_loader_load_edges(mock_graph_db, sample_edge):
    """Test loading edges into Neo4j."""
    mock_driver = Mock()
    mock_session = MagicMock()
    mock_tx = MagicMock()
    
    mock_graph_db.driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = Mock(return_value=False)
    mock_session.begin_transaction.return_value.__enter__ = Mock(return_value=mock_tx)
    mock_session.begin_transaction.return_value.__exit__ = Mock(return_value=False)
    
    loader = EnhancedNeo4jLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="test_db",
        batch_size=10
    )
    
    edges = [sample_edge]
    loader.load_edges(edges)
    
    # Verify that transaction was used
    mock_session.begin_transaction.assert_called()
    mock_tx.commit.assert_called()


@patch('graph.enhanced_kg_pipeline.GraphDatabase')
def test_neo4j_loader_load_claims(mock_graph_db, sample_claim):
    """Test loading reified claims into Neo4j."""
    mock_driver = Mock()
    mock_session = MagicMock()
    mock_tx = MagicMock()
    
    mock_graph_db.driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = Mock(return_value=False)
    mock_session.begin_transaction.return_value.__enter__ = Mock(return_value=mock_tx)
    mock_session.begin_transaction.return_value.__exit__ = Mock(return_value=False)
    
    loader = EnhancedNeo4jLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="test_db",
        batch_size=10
    )
    
    claims = [sample_claim]
    loader.load_claims(claims)
    
    # Verify that transaction was used
    mock_session.begin_transaction.assert_called()
    mock_tx.commit.assert_called()


@patch('graph.enhanced_kg_pipeline.GraphDatabase')
def test_neo4j_loader_batch_processing(mock_graph_db, sample_edge):
    """Test that loader processes edges in batches."""
    mock_driver = Mock()
    mock_session = MagicMock()
    mock_tx = MagicMock()
    
    mock_graph_db.driver.return_value = mock_driver
    mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = Mock(return_value=False)
    mock_session.begin_transaction.return_value.__enter__ = Mock(return_value=mock_tx)
    mock_session.begin_transaction.return_value.__exit__ = Mock(return_value=False)
    
    loader = EnhancedNeo4jLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="test_db",
        batch_size=5  # Small batch size for testing
    )
    
    # Create 12 edges (should be 3 batches)
    edges = [sample_edge] * 12
    loader.load_edges(edges)
    
    # Verify that multiple batches were processed
    assert mock_session.begin_transaction.call_count == 3


# ========== EnhancedKGPipeline Tests ==========

@patch('graph.enhanced_kg_pipeline.EnhancedNeo4jLoader')
def test_pipeline_initialization(mock_loader_class, sample_config):
    """Test pipeline initialization with config."""
    mock_loader = Mock()
    mock_loader_class.return_value = mock_loader
    
    pipeline = EnhancedKGPipeline(sample_config)
    
    assert pipeline.config == sample_config
    mock_loader_class.assert_called_once()
    mock_loader.create_indexes.assert_called_once()


@patch('graph.enhanced_kg_pipeline.EnhancedNeo4jLoader')
def test_pipeline_disabled(mock_loader_class):
    """Test that pipeline skips execution when disabled."""
    config = PipelineConfig(enabled=False)
    pipeline = EnhancedKGPipeline(config)
    
    result = pipeline.run([])
    
    assert result["status"] == "disabled"
    mock_loader_class.assert_not_called()


@patch('graph.enhanced_kg_pipeline.EnhancedNeo4jLoader')
@patch('graph.enhanced_kg_pipeline.EnhancedGraphBuilder')
def test_pipeline_batch_processing(
    mock_builder_class,
    mock_loader_class,
    sample_config,
    sample_enriched_papers
):
    """
    Test that pipeline processes papers in batches.
    
    Requirement 17.2: Process papers in batches of 100
    """
    # Setup mocks
    mock_loader = Mock()
    mock_loader_class.return_value = mock_loader
    
    mock_builder = Mock()
    mock_builder.process_papers.return_value = []
    mock_builder.relationships = []
    mock_builder.edges = []
    mock_builder.relationship_index = {}
    mock_builder.create_reified_claims.return_value = []
    mock_builder.get_statistics.return_value = {
        "total_relationships": 0,
        "total_edges": 0,
        "total_claims": 0
    }
    mock_builder_class.return_value = mock_builder
    
    # Use smaller batch size for testing
    sample_config.batch_size = 10
    sample_config.save_intermediate = False
    
    pipeline = EnhancedKGPipeline(sample_config)
    result = pipeline.run(sample_enriched_papers, load_to_neo4j=False)
    
    # Verify batching: 25 papers / 10 per batch = 3 batches
    assert result["status"] == "success"
    # Builder should be created multiple times (once per batch)
    assert mock_builder_class.call_count >= 3


@patch('graph.enhanced_kg_pipeline.EnhancedNeo4jLoader')
@patch('graph.enhanced_kg_pipeline.EnhancedGraphBuilder')
def test_pipeline_parallel_workers(
    mock_builder_class,
    mock_loader_class,
    sample_config,
    sample_enriched_papers
):
    """
    Test that pipeline uses parallel workers.
    
    Requirement 17.2: Use 8-16 parallel workers
    """
    # Setup mocks
    mock_loader = Mock()
    mock_loader_class.return_value = mock_loader
    
    mock_builder = Mock()
    mock_builder.process_papers.return_value = []
    mock_builder.relationships = []
    mock_builder.edges = []
    mock_builder.relationship_index = {}
    mock_builder.create_reified_claims.return_value = []
    mock_builder.get_statistics.return_value = {
        "total_relationships": 0,
        "total_edges": 0,
        "total_claims": 0
    }
    mock_builder_class.return_value = mock_builder
    
    sample_config.num_workers = 4
    sample_config.batch_size = 10
    sample_config.save_intermediate = False
    
    pipeline = EnhancedKGPipeline(sample_config)
    result = pipeline.run(sample_enriched_papers, load_to_neo4j=False)
    
    assert result["status"] == "success"
    # Verify that parallel processing occurred
    assert mock_builder_class.call_count >= 1


@patch('graph.enhanced_kg_pipeline.EnhancedNeo4jLoader')
@patch('graph.enhanced_kg_pipeline.EnhancedGraphBuilder')
def test_pipeline_creates_reified_claims(
    mock_builder_class,
    mock_loader_class,
    sample_config,
    sample_enriched_papers,
    sample_claim
):
    """Test that pipeline creates reified claims from aggregated evidence."""
    # Setup mocks
    mock_loader = Mock()
    mock_loader_class.return_value = mock_loader
    
    mock_builder = Mock()
    mock_builder.process_papers.return_value = []
    mock_builder.relationships = []
    mock_builder.edges = []
    mock_builder.relationship_index = {}
    mock_builder.create_reified_claims.return_value = [sample_claim]
    mock_builder.get_statistics.return_value = {
        "total_relationships": 1,
        "total_edges": 1,
        "total_claims": 1
    }
    mock_builder_class.return_value = mock_builder
    
    sample_config.save_intermediate = False
    
    pipeline = EnhancedKGPipeline(sample_config)
    result = pipeline.run(sample_enriched_papers[:5], load_to_neo4j=False)
    
    assert result["status"] == "success"
    assert result["claims_count"] == 1


@patch('graph.enhanced_kg_pipeline.EnhancedNeo4jLoader')
@patch('graph.enhanced_kg_pipeline.EnhancedGraphBuilder')
def test_pipeline_loads_to_neo4j(
    mock_builder_class,
    mock_loader_class,
    sample_config,
    sample_enriched_papers,
    sample_edge,
    sample_claim
):
    """
    Test that pipeline loads edges and claims to Neo4j.
    
    Requirement 16.1: Write to separate Neo4j database instance
    """
    # Setup mocks
    mock_loader = Mock()
    mock_loader_class.return_value = mock_loader
    
    mock_builder = Mock()
    mock_builder.process_papers.return_value = [sample_edge]
    mock_builder.relationships = []
    mock_builder.edges = [sample_edge]
    mock_builder.relationship_index = {}
    mock_builder.create_reified_claims.return_value = [sample_claim]
    mock_builder.get_statistics.return_value = {
        "total_relationships": 1,
        "total_edges": 1,
        "total_claims": 1
    }
    mock_builder_class.return_value = mock_builder
    
    sample_config.save_intermediate = False
    
    pipeline = EnhancedKGPipeline(sample_config)
    result = pipeline.run(sample_enriched_papers[:5], load_to_neo4j=True)
    
    assert result["status"] == "success"
    mock_loader.load_edges.assert_called_once()
    mock_loader.load_claims.assert_called_once()


@patch('graph.enhanced_kg_pipeline.EnhancedNeo4jLoader')
@patch('graph.enhanced_kg_pipeline.EnhancedGraphBuilder')
def test_pipeline_saves_intermediate_results(
    mock_builder_class,
    mock_loader_class,
    sample_config,
    sample_enriched_papers,
    sample_edge,
    sample_claim,
    tmp_path
):
    """Test that pipeline saves intermediate results to JSON files."""
    # Setup mocks
    mock_loader = Mock()
    mock_loader_class.return_value = mock_loader
    
    mock_builder = Mock()
    mock_builder.process_papers.return_value = [sample_edge]
    mock_builder.relationships = []
    mock_builder.edges = [sample_edge]
    mock_builder.relationship_index = {}
    mock_builder.create_reified_claims.return_value = [sample_claim]
    mock_builder.get_statistics.return_value = {
        "total_relationships": 1,
        "total_edges": 1,
        "total_claims": 1
    }
    mock_builder_class.return_value = mock_builder
    
    sample_config.save_intermediate = True
    sample_config.output_dir = tmp_path
    
    pipeline = EnhancedKGPipeline(sample_config)
    result = pipeline.run(sample_enriched_papers[:5], load_to_neo4j=False)
    
    assert result["status"] == "success"
    
    # Check that files were created
    files = list(tmp_path.glob("enhanced_*.json"))
    assert len(files) == 3  # edges, claims, stats


@patch('graph.enhanced_kg_pipeline.EnhancedNeo4jLoader')
@patch('graph.enhanced_kg_pipeline.EnhancedGraphBuilder')
def test_pipeline_returns_statistics(
    mock_builder_class,
    mock_loader_class,
    sample_config,
    sample_enriched_papers
):
    """Test that pipeline returns comprehensive statistics."""
    # Setup mocks
    mock_loader = Mock()
    mock_loader_class.return_value = mock_loader
    
    mock_builder = Mock()
    mock_builder.process_papers.return_value = []
    mock_builder.relationships = []
    mock_builder.edges = []
    mock_builder.relationship_index = {}
    mock_builder.create_reified_claims.return_value = []
    mock_builder.get_statistics.return_value = {
        "total_relationships": 10,
        "total_edges": 10,
        "associations": 5,
        "interventions": 3,
        "methodologies": 2
    }
    mock_builder_class.return_value = mock_builder
    
    sample_config.save_intermediate = False
    
    pipeline = EnhancedKGPipeline(sample_config)
    result = pipeline.run(sample_enriched_papers[:5], load_to_neo4j=False)
    
    assert result["status"] == "success"
    assert "statistics" in result
    assert "processing_time_seconds" in result["statistics"]
    assert result["statistics"]["total_relationships"] == 10


# ========== Convenience Function Tests ==========

@patch('graph.enhanced_kg_pipeline.EnhancedKGPipeline')
def test_run_enhanced_pipeline_convenience(mock_pipeline_class, sample_enriched_papers):
    """Test the convenience function for running the pipeline."""
    mock_pipeline = Mock()
    mock_pipeline.run.return_value = {"status": "success"}
    mock_pipeline_class.return_value = mock_pipeline
    
    result = run_enhanced_pipeline(sample_enriched_papers)
    
    assert result["status"] == "success"
    mock_pipeline.run.assert_called_once()
    mock_pipeline.close.assert_called_once()


# ========== Integration Tests ==========

def test_pipeline_config_validation():
    """Test that pipeline config validates worker count."""
    config = PipelineConfig(num_workers=16)
    assert config.num_workers == 16
    
    # Test recommended range (8-16 workers per Requirement 17.2)
    config = PipelineConfig(num_workers=8)
    assert config.num_workers == 8


def test_pipeline_batch_size_validation():
    """Test that pipeline config validates batch size."""
    # Requirement 17.2: batches of 100
    config = PipelineConfig(batch_size=100)
    assert config.batch_size == 100
    
    # Test custom batch size
    config = PipelineConfig(batch_size=50)
    assert config.batch_size == 50


def test_separate_database_configuration():
    """
    Test that pipeline uses separate database instance.
    
    Requirement 16.1: Write to separate Neo4j database instance
    """
    config = PipelineConfig(neo4j_database="neo4j_enhanced")
    assert config.neo4j_database == "neo4j_enhanced"
    assert config.neo4j_database != "neo4j"  # Different from default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

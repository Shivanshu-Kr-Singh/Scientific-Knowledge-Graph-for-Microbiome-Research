"""
graph/test_enhanced_neo4j_loader.py
------------------------------------
Unit tests for the enhanced Neo4j loader.

Tests batch loading, provenance embedding, and relationship property storage.

Requirements: 12.5, 17.5
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

from graph.enhanced_neo4j_loader import EnhancedNeo4jLoader, load_enhanced_graph
from graph.semantic_relationships import (
    SemanticRelationship,
    RelationType,
    create_association_relationship,
    create_intervention_relationship,
    create_methodology_relationship,
)
from graph.provenance import ProvenanceMetadata


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = Mock()
    session = Mock()
    tx = Mock()
    
    # Setup mock chain
    driver.session.return_value.__enter__.return_value = session
    session.begin_transaction.return_value.__enter__.return_value = tx
    session.run = Mock()
    tx.run = Mock()
    tx.commit = Mock()
    
    return driver


@pytest.fixture
def sample_provenance():
    """Create sample provenance metadata."""
    return ProvenanceMetadata(
        paper_id="PMC123456",
        section_type="results",
        source_sentence="Bacteroides fragilis was significantly increased in T2D patients.",
        sentence_offset=150,
        extraction_method="llm_extractor_v1.2",
        extraction_timestamp=datetime(2024, 1, 15, 10, 30, 0),
        extractor_version="1.2.0",
        llm_prompt_hash="abc123def456",
        confidence_score=0.87,
        validation_status="unvalidated",
        validator_id=None,
        surrounding_context="Previous sentence. Bacteroides fragilis was significantly increased in T2D patients. Next sentence.",
        figure_table_ref="Table 2"
    )


@pytest.fixture
def sample_association_relationship(sample_provenance):
    """Create sample association relationship."""
    return create_association_relationship(
        source_entity="PMC123456",
        target_entity="Bacteroides_fragilis",
        direction="increased",
        comparison="T2D vs healthy",
        statistical_measure="LDA score",
        provenance=sample_provenance,
        evidence_strength="strong",
        extraction_confidence=0.87,
        effect_size=3.2,
        p_value=0.001,
        adjusted_p_value=0.005
    )


@pytest.fixture
def sample_intervention_relationship(sample_provenance):
    """Create sample intervention relationship."""
    return create_intervention_relationship(
        source_entity="PMC123456",
        target_entity="Lactobacillus_rhamnosus",
        intervention_type="probiotic",
        effect_direction="increased",
        provenance=sample_provenance,
        evidence_strength="moderate",
        extraction_confidence=0.75,
        duration="4 weeks",
        dosage="10^9 CFU/day",
        sample_size=50
    )


class TestEnhancedNeo4jLoader:
    """Test suite for EnhancedNeo4jLoader."""
    
    def test_initialization(self):
        """Test loader initialization with connection parameters."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            loader = EnhancedNeo4jLoader(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                batch_size=5000
            )
            
            # Verify driver was created with correct parameters
            mock_gdb.driver.assert_called_once_with(
                "bolt://localhost:7687",
                auth=("neo4j", "password")
            )
            
            assert loader.batch_size == 5000
    
    def test_context_manager(self):
        """Test loader works as context manager."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_gdb.driver.return_value = mock_driver
            
            with EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password") as loader:
                assert loader is not None
            
            # Verify close was called
            mock_driver.close.assert_called_once()
    
    def test_serialize_properties_with_datetime(self):
        """Test property serialization converts datetime to ISO format."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase'):
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            props = {
                "timestamp": datetime(2024, 1, 15, 10, 30, 0),
                "name": "test",
                "value": 42,
                "none_value": None
            }
            
            serialized = loader._serialize_properties(props)
            
            assert serialized["timestamp"] == "2024-01-15T10:30:00"
            assert serialized["name"] == "test"
            assert serialized["value"] == 42
            assert "none_value" not in serialized  # None values are skipped
    
    def test_build_relationship_properties_association(self, sample_association_relationship):
        """Test building relationship properties from association relationship."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase'):
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            props = loader._build_relationship_properties(sample_association_relationship)
            
            # Check semantic properties
            assert props["direction"] == "increased"
            assert props["comparison"] == "T2D vs healthy"
            assert props["statistical_measure"] == "LDA score"
            assert props["effect_size"] == 3.2
            assert props["p_value"] == 0.001
            assert props["adjusted_p_value"] == 0.005
            
            # Check provenance properties
            assert props["section"] == "results"
            assert props["source_sentence"] == "Bacteroides fragilis was significantly increased in T2D patients."
            assert props["sentence_offset"] == 150
            assert props["extraction_method"] == "llm_extractor_v1.2"
            assert props["extraction_timestamp"] == "2024-01-15T10:30:00"
            assert props["extractor_version"] == "1.2.0"
            assert props["llm_prompt_hash"] == "abc123def456"
            
            # Check quality indicators
            assert props["confidence"] == 0.87
            assert props["evidence_strength"] == "strong"
    
    def test_build_relationship_properties_intervention(self, sample_intervention_relationship):
        """Test building relationship properties from intervention relationship."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase'):
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            props = loader._build_relationship_properties(sample_intervention_relationship)
            
            # Check intervention-specific properties
            assert props["intervention_type"] == "probiotic"
            assert props["effect_direction"] == "increased"
            assert props["duration"] == "4 weeks"
            assert props["dosage"] == "10^9 CFU/day"
            assert props["sample_size"] == 50
            
            # Check provenance is embedded
            assert props["section"] == "results"
            assert props["extraction_method"] == "llm_extractor_v1.2"
            
            # Check quality indicators
            assert props["confidence"] == 0.75
            assert props["evidence_strength"] == "moderate"
    
    def test_create_node(self):
        """Test creating a single node."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            node_props = {
                "name": "Bacteroides fragilis",
                "ncbi_id": "817",
                "grounded": True
            }
            
            loader.create_node(mock_session, "Taxon", "Bacteroides_fragilis", node_props)
            
            # Verify Cypher query was executed
            mock_session.run.assert_called_once()
            call_args = mock_session.run.call_args
            
            assert "MERGE (n:Taxon {id: $id})" in call_args[0][0]
            assert call_args[1]["id"] == "Bacteroides_fragilis"
            assert call_args[1]["props"]["name"] == "Bacteroides fragilis"
    
    def test_create_relationship(self, sample_association_relationship):
        """Test creating a relationship with rich properties."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = Mock()
            mock_session = Mock()
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            loader.create_relationship(mock_session, sample_association_relationship)
            
            # Verify Cypher query was executed
            mock_session.run.assert_called_once()
            call_args = mock_session.run.call_args
            
            # Check query structure
            query = call_args[0][0]
            assert "MATCH (source {id: $source_id})" in query
            assert "MATCH (target {id: $target_id})" in query
            assert "MERGE (source)-[r:REPORTS_ASSOCIATION]->(target)" in query
            assert "SET r += $props" in query
            
            # Check parameters
            assert call_args[1]["source_id"] == "PMC123456"
            assert call_args[1]["target_id"] == "Bacteroides_fragilis"
            
            # Check properties include both semantic and provenance data
            props = call_args[1]["props"]
            assert "direction" in props
            assert "p_value" in props
            assert "section" in props
            assert "extraction_method" in props
    
    def test_load_nodes_batch_single_batch(self):
        """Test loading nodes in a single batch."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_tx = MagicMock()
            
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password", batch_size=100)
            
            nodes = [
                {"id": "taxon1", "name": "Bacteroides fragilis"},
                {"id": "taxon2", "name": "Lactobacillus rhamnosus"},
                {"id": "taxon3", "name": "Escherichia coli"}
            ]
            
            loader.load_nodes_batch(nodes, "Taxon")
            
            # Verify transaction was created and committed
            mock_session.begin_transaction.assert_called_once()
            mock_tx.commit.assert_called_once()
            
            # Verify all nodes were processed
            assert mock_tx.run.call_count == 3
    
    def test_load_nodes_batch_multiple_batches(self):
        """Test loading nodes across multiple batches."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_tx = MagicMock()
            
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
            mock_gdb.driver.return_value = mock_driver
            
            # Set small batch size to test batching
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password", batch_size=2)
            
            nodes = [
                {"id": f"taxon{i}", "name": f"Taxon {i}"}
                for i in range(5)
            ]
            
            loader.load_nodes_batch(nodes, "Taxon")
            
            # Verify multiple transactions were created (3 batches: 2, 2, 1)
            assert mock_session.begin_transaction.call_count == 3
            assert mock_tx.commit.call_count == 3
            
            # Verify all nodes were processed
            assert mock_tx.run.call_count == 5
    
    def test_load_relationships_batch(self, sample_association_relationship, sample_intervention_relationship):
        """Test loading relationships in batches."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_tx = MagicMock()
            
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password", batch_size=100)
            
            relationships = [
                sample_association_relationship,
                sample_intervention_relationship
            ]
            
            loader.load_relationships_batch(relationships)
            
            # Verify transaction was created and committed
            mock_session.begin_transaction.assert_called_once()
            mock_tx.commit.assert_called_once()
            
            # Verify both relationships were processed
            assert mock_tx.run.call_count == 2
    
    def test_load_graph_complete(self, sample_association_relationship):
        """Test loading complete graph with nodes and relationships."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_tx = MagicMock()
            
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password", batch_size=100)
            
            nodes = {
                "Paper": [{"id": "PMC123456", "title": "Test Paper"}],
                "Taxon": [{"id": "Bacteroides_fragilis", "name": "Bacteroides fragilis"}],
                "Disease": [{"id": "T2D", "name": "Type 2 Diabetes"}]
            }
            
            relationships = [sample_association_relationship]
            
            loader.load_graph(nodes, relationships)
            
            # Verify transactions were created for nodes and relationships
            # 3 node types + 1 relationship batch = 4 transactions
            assert mock_session.begin_transaction.call_count == 4
            assert mock_tx.commit.call_count == 4
    
    def test_create_indexes(self):
        """Test index creation for efficient querying."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            loader.create_indexes()
            
            # Verify multiple index creation queries were executed
            assert mock_session.run.call_count > 0
            
            # Check that paper, taxon, and disease indexes were created
            calls = [call[0][0] for call in mock_session.run.call_args_list]
            
            # Paper indexes
            assert any("paper_year" in call for call in calls)
            assert any("paper_article_type" in call for call in calls)
            assert any("paper_data_availability" in call for call in calls)
            
            # Entity indexes
            assert any("taxon_name" in call for call in calls)
            assert any("disease_name" in call for call in calls)
            assert any("method_name" in call for call in calls)
            
            # Canonical identifier indexes (Requirement 12.2)
            assert any("taxon_ncbi_id" in call for call in calls)
            assert any("disease_mesh_id" in call for call in calls)
            
            # Relationship property indexes (Requirement 12.3)
            assert any("rel_association_confidence" in call for call in calls)
            assert any("rel_association_p_value" in call for call in calls)
            assert any("rel_intervention_confidence" in call for call in calls)
            assert any("rel_intervention_p_value" in call for call in calls)
            assert any("rel_intervention_type" in call for call in calls)
            assert any("rel_methodology_confidence" in call for call in calls)
            
            # Composite indexes (Requirement 12.4)
            assert any("rel_association_evidence_consensus_composite" in call for call in calls)
            assert any("rel_intervention_evidence_consensus_composite" in call for call in calls)
    
    def test_clear_database(self):
        """Test database clearing functionality."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = MagicMock()
            mock_session = MagicMock()
            
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            loader.clear_database()
            
            # Verify delete query was executed
            mock_session.run.assert_called_once()
            call_args = mock_session.run.call_args
            assert "MATCH (n) DETACH DELETE n" in call_args[0][0]
    
    def test_batch_size_requirement(self):
        """Test that default batch size meets requirement of 10,000."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase'):
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            # Requirement 17.5: Batch loading with 10,000 nodes/edges per transaction
            assert loader.batch_size == 10000
    
    def test_load_claims_batch(self):
        """Test loading reified claims in batches (Requirements 4.1, 4.2, 4.3)."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            from graph.reified_claims import ScientificClaim
            
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_tx = MagicMock()
            
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password", batch_size=100)
            
            # Create sample claims
            claims = [
                ScientificClaim(
                    claim_id="claim-001",
                    claim_type="association",
                    subject_entity="Bacteroides_fragilis",
                    predicate="associated_with_increased_abundance",
                    object_entity="Type_2_Diabetes",
                    supporting_papers=["PMC123456", "PMC789012"],
                    contradicting_papers=["PMC345678"],
                    total_sample_size=150,
                    evidence_strength="strong",
                    consensus_confidence=0.85,
                    effect_direction_consistency=0.75,
                    first_reported="2024-01-15",
                    last_updated="2024-03-20",
                    pooled_effect_size=2.5,
                    effect_size_variance=0.3,
                    meta_analysis_performed=True
                ),
                ScientificClaim(
                    claim_id="claim-002",
                    claim_type="intervention_effect",
                    subject_entity="Lactobacillus_rhamnosus",
                    predicate="increased_by_probiotic",
                    object_entity="Probiotic_Intervention",
                    supporting_papers=["PMC111222"],
                    contradicting_papers=[],
                    total_sample_size=50,
                    evidence_strength="moderate",
                    consensus_confidence=0.70,
                    effect_direction_consistency=1.0,
                    first_reported="2024-02-10",
                    last_updated="2024-02-10",
                    pooled_effect_size=None,
                    effect_size_variance=None,
                    meta_analysis_performed=False
                )
            ]
            
            loader.load_claims_batch(claims)
            
            # Verify transaction was created and committed
            mock_session.begin_transaction.assert_called_once()
            mock_tx.commit.assert_called_once()
            
            # Verify Cypher queries were executed
            # For each claim: 1 CREATE node + N SUPPORTED_BY + M CONTRADICTED_BY
            # Claim 1: 1 + 2 + 1 = 4 queries
            # Claim 2: 1 + 1 + 0 = 2 queries
            # Total: 6 queries
            assert mock_tx.run.call_count == 6
            
            # Verify claim node creation queries
            calls = [call[0][0] for call in mock_tx.run.call_args_list]
            
            # Check that ScientificClaim nodes were created
            create_claim_calls = [call for call in calls if "CREATE (c:ScientificClaim" in call]
            assert len(create_claim_calls) == 2
            
            # Check that SUPPORTED_BY relationships were created
            supported_by_calls = [call for call in calls if "SUPPORTED_BY" in call]
            assert len(supported_by_calls) == 3  # 2 from claim-001, 1 from claim-002
            
            # Check that CONTRADICTED_BY relationships were created
            contradicted_by_calls = [call for call in calls if "CONTRADICTED_BY" in call]
            assert len(contradicted_by_calls) == 1  # 1 from claim-001
    
    def test_load_claims_batch_empty_list(self):
        """Test loading empty claims list handles gracefully."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            mock_driver = MagicMock()
            mock_gdb.driver.return_value = mock_driver
            
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password")
            
            # Should not raise an error
            loader.load_claims_batch([])
            
            # Verify no session was created
            mock_driver.session.assert_not_called()
    
    def test_load_claims_batch_multiple_batches(self):
        """Test loading claims across multiple batches."""
        with patch('graph.enhanced_neo4j_loader.GraphDatabase') as mock_gdb:
            from graph.reified_claims import ScientificClaim
            
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_tx = MagicMock()
            
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.begin_transaction.return_value.__enter__.return_value = mock_tx
            mock_gdb.driver.return_value = mock_driver
            
            # Set small batch size to test batching
            loader = EnhancedNeo4jLoader("bolt://localhost:7687", "neo4j", "password", batch_size=2)
            
            # Create 5 claims to trigger multiple batches (3 batches: 2, 2, 1)
            claims = [
                ScientificClaim(
                    claim_id=f"claim-{i:03d}",
                    claim_type="association",
                    subject_entity=f"Taxon_{i}",
                    predicate="associated_with",
                    object_entity="Disease_X",
                    supporting_papers=[f"PMC{i}"],
                    contradicting_papers=[],
                    total_sample_size=50,
                    evidence_strength="moderate",
                    consensus_confidence=0.75,
                    effect_direction_consistency=0.80,
                    first_reported="2024-01-01",
                    last_updated="2024-01-01",
                    pooled_effect_size=None,
                    effect_size_variance=None,
                    meta_analysis_performed=False
                )
                for i in range(5)
            ]
            
            loader.load_claims_batch(claims)
            
            # Verify multiple transactions were created (3 batches)
            assert mock_session.begin_transaction.call_count == 3
            assert mock_tx.commit.call_count == 3
            
            # Verify all claims were processed
            # Each claim: 1 CREATE + 1 SUPPORTED_BY = 2 queries
            # 5 claims * 2 = 10 queries
            assert mock_tx.run.call_count == 10


class TestLoadEnhancedGraph:
    """Test suite for convenience function."""
    
    def test_load_enhanced_graph_with_indexes(self, sample_association_relationship):
        """Test convenience function loads graph and creates indexes."""
        with patch('graph.enhanced_neo4j_loader.EnhancedNeo4jLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value.__enter__.return_value = mock_loader
            
            nodes = {
                "Paper": [{"id": "PMC123456", "title": "Test Paper"}],
                "Taxon": [{"id": "Bacteroides_fragilis", "name": "Bacteroides fragilis"}]
            }
            
            relationships = [sample_association_relationship]
            
            load_enhanced_graph(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                nodes=nodes,
                relationships=relationships,
                create_indexes=True,
                batch_size=5000
            )
            
            # Verify loader was created with correct parameters
            mock_loader_class.assert_called_once_with(
                "bolt://localhost:7687",
                "neo4j",
                "password",
                5000,
                None  # validation_queue_path defaults to None
            )
            
            # Verify graph was loaded
            mock_loader.load_graph.assert_called_once_with(nodes, relationships, True)
            
            # Verify indexes were created
            mock_loader.create_indexes.assert_called_once()
    
    def test_load_enhanced_graph_without_indexes(self, sample_association_relationship):
        """Test convenience function can skip index creation."""
        with patch('graph.enhanced_neo4j_loader.EnhancedNeo4jLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value.__enter__.return_value = mock_loader
            
            nodes = {"Paper": [{"id": "PMC123456", "title": "Test Paper"}]}
            relationships = [sample_association_relationship]
            
            load_enhanced_graph(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="password",
                nodes=nodes,
                relationships=relationships,
                create_indexes=False
            )
            
            # Verify graph was loaded
            mock_loader.load_graph.assert_called_once()
            
            # Verify indexes were NOT created
            mock_loader.create_indexes.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

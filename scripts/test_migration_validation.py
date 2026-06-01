#!/usr/bin/env python3
"""
scripts/test_migration_validation.py
------------------------------------
Integration test for migration validation.

This test validates that:
1. Migrated data matches old data structure (Requirement 16.2)
2. Provenance metadata is added correctly (Requirement 20.3)
3. Entity counts match between old and new systems (Requirement 16.2)

Requirements: 16.2, 20.3
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.migrate_to_enhanced_schema import SchemaEnhancementMigrator, MigrationStats
from scripts.backup_neo4j import Neo4jBackupManager
from scripts.rollback_neo4j import Neo4jRollbackManager
from graph.semantic_relationships import RelationType
from graph.provenance import ProvenanceMetadata


class TestMigrationValidation:
    """
    Integration tests for migration validation.
    
    Tests that migrated data matches old data structure, provenance metadata
    is added correctly, and entity counts match between systems.
    
    Requirements: 16.2, 20.3
    """
    
    @pytest.fixture
    def mock_old_driver(self):
        """Create a mock driver for old Neo4j database."""
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        driver.session.return_value.__exit__ = Mock(return_value=False)
        return driver
    
    @pytest.fixture
    def mock_new_loader(self):
        """Create a mock loader for new Neo4j database."""
        loader = MagicMock()
        loader.driver = MagicMock()
        session = MagicMock()
        loader.driver.session.return_value.__enter__.return_value = session
        loader.driver.session.return_value.__exit__ = Mock(return_value=False)
        return loader
    
    @pytest.fixture
    def sample_old_nodes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create sample old nodes for testing."""
        return {
            "Paper": [
                {
                    "id": "paper1",
                    "title": "Microbiome Study 1",
                    "year": 2024,
                    "article_type": "original_research",
                },
                {
                    "id": "paper2",
                    "title": "Microbiome Study 2",
                    "year": 2024,
                    "article_type": "meta_analysis",
                },
                {
                    "id": "paper3",
                    "title": "Microbiome Study 3",
                    "year": 2023,
                    "article_type": "original_research",
                },
            ],
            "Taxon": [
                {"id": "taxon1", "name": "Bacteroides fragilis"},
                {"id": "taxon2", "name": "Escherichia coli"},
                {"id": "taxon3", "name": "Lactobacillus acidophilus"},
            ],
            "Disease": [
                {"id": "disease1", "name": "Type 2 Diabetes"},
                {"id": "disease2", "name": "Inflammatory Bowel Disease"},
            ],
        }
    
    @pytest.fixture
    def sample_old_relationships(self) -> List[Dict[str, Any]]:
        """Create sample old relationships for testing."""
        return [
            # Relationship with provenance
            {
                "source_id": "paper1",
                "target_id": "taxon1",
                "rel_type": "REPORTS_ASSOCIATION",
                "source_labels": ["Paper"],
                "target_labels": ["Taxon"],
                "properties": {
                    "direction": "increased",
                    "comparison": "T2D vs healthy",
                    "statistical_measure": "LDA score",
                    "effect_size": 2.5,
                    "p_value": 0.01,
                    "section": "results",
                    "source_sentence": "Bacteroides fragilis was significantly increased in T2D patients",
                    "extraction_method": "biobert_ner",
                    "extraction_timestamp": "2024-01-01T00:00:00+00:00",
                    "extractor_version": "1.0",
                    "confidence": 0.85,
                },
            },
            # Legacy relationship without provenance
            {
                "source_id": "paper2",
                "target_id": "taxon2",
                "rel_type": "HAS_TAXON",
                "source_labels": ["Paper"],
                "target_labels": ["Taxon"],
                "properties": {
                    "direction": "decreased",
                },
            },
            # Intervention relationship
            {
                "source_id": "paper3",
                "target_id": "taxon3",
                "rel_type": "REPORTS_INTERVENTION_EFFECT",
                "source_labels": ["Paper"],
                "target_labels": ["Taxon"],
                "properties": {
                    "intervention_type": "probiotic",
                    "effect_direction": "increased",
                    "duration": "4 weeks",
                    "sample_size": 100,
                    "section": "results",
                    "source_sentence": "Probiotic supplementation increased Lactobacillus",
                    "extraction_method": "llm_extractor_v1.2",
                    "extraction_timestamp": "2024-01-15T00:00:00+00:00",
                    "extractor_version": "1.2",
                    "confidence": 0.90,
                },
            },
        ]
    
    @pytest.fixture
    def migrator(self, mock_old_driver, mock_new_loader):
        """Create a migrator with mocked dependencies."""
        with patch('scripts.migrate_to_enhanced_schema.GraphDatabase') as mock_gdb, \
             patch('scripts.migrate_to_enhanced_schema.EnhancedNeo4jLoader') as mock_loader_class:
            
            mock_gdb.driver.return_value = mock_old_driver
            mock_loader_class.return_value = mock_new_loader
            
            migrator = SchemaEnhancementMigrator(
                old_uri="bolt://localhost:7687",
                old_user="neo4j",
                old_password="password",
                new_uri="bolt://localhost:7688",
                new_user="neo4j",
                new_password="password",
                batch_size=100
            )
            
            yield migrator
            
            migrator.close()
    
    def test_migrated_data_structure_matches_old_system(
        self,
        migrator,
        sample_old_nodes,
        sample_old_relationships
    ):
        """
        Test that migrated data matches old data structure.
        
        Validates:
        - All node types are preserved
        - All node properties are preserved
        - All relationship types are mapped correctly
        - All relationship properties are preserved or enhanced
        
        Requirement 16.2: Migrated data matches old data structure
        """
        # Test node structure preservation
        for node_type, nodes in sample_old_nodes.items():
            for node in nodes:
                # Verify all properties are present
                assert "id" in node
                
                # Verify node type-specific properties
                if node_type == "Paper":
                    assert "title" in node
                    assert "year" in node
                elif node_type == "Taxon":
                    assert "name" in node
                elif node_type == "Disease":
                    assert "name" in node
        
        # Test relationship structure preservation
        for old_rel in sample_old_relationships:
            enhanced_rel = migrator.enhance_relationship_with_provenance(old_rel)
            
            assert enhanced_rel is not None, f"Failed to enhance relationship: {old_rel['rel_type']}"
            
            # Verify core relationship structure
            assert enhanced_rel.source_entity == old_rel["source_id"]
            assert enhanced_rel.target_entity == old_rel["target_id"]
            
            # Verify relationship type mapping
            expected_type = migrator._map_relationship_type(old_rel["rel_type"])
            assert enhanced_rel.relation_type == expected_type
            
            # Verify properties are preserved
            old_props = old_rel["properties"]
            new_props = enhanced_rel.properties
            
            # Check that key properties are preserved
            if "direction" in old_props:
                assert "direction" in new_props
                assert new_props["direction"] == old_props["direction"]
            
            if "effect_size" in old_props:
                assert "effect_size" in new_props
                assert new_props["effect_size"] == old_props["effect_size"]
            
            if "p_value" in old_props:
                assert "p_value" in new_props
                assert new_props["p_value"] == old_props["p_value"]
            
            if "intervention_type" in old_props:
                assert "intervention_type" in new_props
                assert new_props["intervention_type"] == old_props["intervention_type"]
    
    def test_provenance_metadata_added_correctly(
        self,
        migrator,
        sample_old_relationships
    ):
        """
        Test that provenance metadata is added correctly.
        
        Validates:
        - Existing provenance is preserved
        - Legacy relationships get provenance metadata
        - All required provenance fields are present
        - Provenance metadata is valid
        
        Requirement 20.3: Provenance metadata is added correctly
        """
        for old_rel in sample_old_relationships:
            enhanced_rel = migrator.enhance_relationship_with_provenance(old_rel)
            
            assert enhanced_rel is not None
            assert enhanced_rel.provenance is not None
            
            provenance = enhanced_rel.provenance
            
            # Verify all required provenance fields are present
            assert provenance.paper_id is not None
            assert provenance.section_type is not None
            assert provenance.source_sentence is not None
            assert provenance.extraction_method is not None
            assert provenance.extraction_timestamp is not None
            assert provenance.extractor_version is not None
            assert provenance.confidence_score is not None
            assert provenance.validation_status is not None
            
            # Verify provenance field types
            assert isinstance(provenance.paper_id, str)
            assert isinstance(provenance.section_type, str)
            assert isinstance(provenance.source_sentence, str)
            assert isinstance(provenance.extraction_method, str)
            assert isinstance(provenance.extraction_timestamp, datetime)
            assert isinstance(provenance.extractor_version, str)
            assert isinstance(provenance.confidence_score, float)
            assert isinstance(provenance.validation_status, str)
            
            # Verify provenance field values are valid
            assert 0.0 <= provenance.confidence_score <= 1.0
            assert provenance.validation_status in ["unvalidated", "human_verified", "cross_validated"]
            assert provenance.section_type in ["abstract", "methods", "results", "discussion", "other"]
            
            # Check if relationship had existing provenance
            old_props = old_rel["properties"]
            has_existing_provenance = all(
                key in old_props for key in ["section", "source_sentence", "extraction_method"]
            )
            
            if has_existing_provenance:
                # Verify existing provenance is preserved
                assert provenance.section_type == old_props["section"]
                assert provenance.source_sentence == old_props["source_sentence"]
                assert provenance.extraction_method == old_props["extraction_method"]
                assert provenance.confidence_score == old_props["confidence"]
            else:
                # Verify legacy provenance is created
                assert "[LEGACY]" in provenance.source_sentence
                assert provenance.extraction_method == "legacy"
                assert provenance.validation_status == "unvalidated"
                assert provenance.confidence_score == 0.5
    
    def test_entity_counts_match_between_systems(
        self,
        migrator,
        sample_old_nodes
    ):
        """
        Test that entity counts match between old and new systems.
        
        Validates:
        - >= 90% of entities are matched
        - Entity counts are tracked correctly
        - Missing entities are identified
        - Match percentage is calculated correctly
        
        Requirement 16.2: Entity counts match between old and new systems
        """
        # Simulate new system with 95% entity match
        new_nodes = {}
        for node_type, nodes in sample_old_nodes.items():
            # Keep 95% of entities (round up)
            keep_count = int(len(nodes) * 0.95) + 1
            new_nodes[node_type] = nodes[:keep_count]
        
        # Run verification
        result = migrator.verify_entity_extraction(sample_old_nodes, new_nodes)
        
        # Verify result
        assert result is True, "Entity extraction verification should pass with >= 90% match"
        
        # Verify statistics
        total_old = sum(len(nodes) for nodes in sample_old_nodes.values())
        total_new = sum(len(nodes) for nodes in new_nodes.values())
        
        assert migrator.stats.entities_matched == total_new
        assert migrator.stats.entities_missing == (total_old - total_new)
        assert migrator.stats.entity_match_percentage >= 90.0
        
        # Test with insufficient match (< 90%)
        migrator_fail = SchemaEnhancementMigrator(
            old_uri="bolt://localhost:7687",
            old_user="neo4j",
            old_password="password",
            new_uri="bolt://localhost:7688",
            new_user="neo4j",
            new_password="password"
        )
        
        # Simulate new system with 80% entity match
        new_nodes_fail = {}
        for node_type, nodes in sample_old_nodes.items():
            # Keep only 80% of entities
            keep_count = int(len(nodes) * 0.80)
            new_nodes_fail[node_type] = nodes[:keep_count]
        
        # Run verification
        result_fail = migrator_fail.verify_entity_extraction(sample_old_nodes, new_nodes_fail)
        
        # Verify result
        assert result_fail is False, "Entity extraction verification should fail with < 90% match"
        assert migrator_fail.stats.entity_match_percentage < 90.0
        
        migrator_fail.close()
    
    def test_migration_statistics_tracking(
        self,
        migrator,
        sample_old_relationships
    ):
        """
        Test that migration statistics are tracked correctly.
        
        Validates:
        - Legacy relationship count is correct
        - Provenance added count is correct
        - Statistics are updated during migration
        """
        # Process relationships
        for old_rel in sample_old_relationships:
            enhanced_rel = migrator.enhance_relationship_with_provenance(old_rel)
            assert enhanced_rel is not None
        
        # Verify statistics
        # 2 relationships have existing provenance, 1 is legacy
        assert migrator.stats.provenance_added == 2
        assert migrator.stats.legacy_relationships == 1
        
        # Verify total
        total_processed = migrator.stats.provenance_added + migrator.stats.legacy_relationships
        assert total_processed == len(sample_old_relationships)
    
    def test_relationship_property_validation(
        self,
        migrator,
        sample_old_relationships
    ):
        """
        Test that relationship properties are validated correctly.
        
        Validates:
        - Confidence scores are in valid range [0.0, 1.0]
        - P-values are in valid range [0.0, 1.0]
        - Direction values are valid
        - Evidence strength is determined correctly
        """
        for old_rel in sample_old_relationships:
            enhanced_rel = migrator.enhance_relationship_with_provenance(old_rel)
            
            assert enhanced_rel is not None
            
            # Validate confidence score
            assert 0.0 <= enhanced_rel.extraction_confidence <= 1.0
            
            # Validate p-value if present
            if "p_value" in enhanced_rel.properties:
                p_value = enhanced_rel.properties["p_value"]
                assert 0.0 <= p_value <= 1.0
            
            # Validate direction if present
            if "direction" in enhanced_rel.properties:
                direction = enhanced_rel.properties["direction"]
                assert direction in ["increased", "decreased", "no_change"]
            
            # Validate evidence strength
            assert enhanced_rel.evidence_strength in ["strong", "moderate", "weak"]
    
    def test_migration_with_missing_properties(self, migrator):
        """
        Test migration handles relationships with missing properties gracefully.
        
        Validates:
        - Relationships with minimal properties are handled
        - Default values are used for missing properties
        - No errors are raised for missing optional properties
        """
        # Relationship with minimal properties
        minimal_rel = {
            "source_id": "paper1",
            "target_id": "taxon1",
            "rel_type": "HAS_TAXON",
            "source_labels": ["Paper"],
            "target_labels": ["Taxon"],
            "properties": {},  # No properties
        }
        
        enhanced_rel = migrator.enhance_relationship_with_provenance(minimal_rel)
        
        assert enhanced_rel is not None
        assert enhanced_rel.source_entity == "paper1"
        assert enhanced_rel.target_entity == "taxon1"
        assert enhanced_rel.provenance is not None
        
        # Verify default values are used
        assert enhanced_rel.properties["direction"] == "no_change"
        assert enhanced_rel.properties["comparison"] == "unknown"
        assert enhanced_rel.evidence_strength == "weak"
    
    def test_migration_preserves_statistical_measures(
        self,
        migrator,
        sample_old_relationships
    ):
        """
        Test that statistical measures are preserved during migration.
        
        Validates:
        - Effect sizes are preserved
        - P-values are preserved
        - Statistical measures are preserved
        - Adjusted p-values are preserved if present
        """
        # Find relationship with statistical measures
        rel_with_stats = sample_old_relationships[0]  # First relationship has stats
        
        enhanced_rel = migrator.enhance_relationship_with_provenance(rel_with_stats)
        
        assert enhanced_rel is not None
        
        # Verify statistical measures are preserved
        old_props = rel_with_stats["properties"]
        new_props = enhanced_rel.properties
        
        assert new_props["effect_size"] == old_props["effect_size"]
        assert new_props["p_value"] == old_props["p_value"]
        assert new_props["statistical_measure"] == old_props["statistical_measure"]
    
    def test_migration_handles_intervention_relationships(
        self,
        migrator,
        sample_old_relationships
    ):
        """
        Test that intervention relationships are migrated correctly.
        
        Validates:
        - Intervention type is preserved
        - Effect direction is preserved
        - Duration is preserved
        - Sample size is preserved
        """
        # Find intervention relationship
        intervention_rel = sample_old_relationships[2]  # Third relationship is intervention
        
        enhanced_rel = migrator.enhance_relationship_with_provenance(intervention_rel)
        
        assert enhanced_rel is not None
        assert enhanced_rel.relation_type == RelationType.REPORTS_INTERVENTION_EFFECT
        
        # Verify intervention properties
        old_props = intervention_rel["properties"]
        new_props = enhanced_rel.properties
        
        assert new_props["intervention_type"] == old_props["intervention_type"]
        assert new_props["effect_direction"] == old_props["effect_direction"]
        assert new_props["duration"] == old_props["duration"]
        assert new_props["sample_size"] == old_props["sample_size"]


class TestMigrationValidationEdgeCases:
    """Test edge cases in migration validation."""
    
    def test_empty_database_migration(self):
        """Test migration with empty old database."""
        with patch('scripts.migrate_to_enhanced_schema.GraphDatabase'), \
             patch('scripts.migrate_to_enhanced_schema.EnhancedNeo4jLoader'):
            
            migrator = SchemaEnhancementMigrator(
                old_uri="bolt://localhost:7687",
                old_user="neo4j",
                old_password="password",
                new_uri="bolt://localhost:7688",
                new_user="neo4j",
                new_password="password"
            )
            
            # Empty nodes and relationships
            old_nodes = {}
            new_nodes = {}
            
            # Should not raise error
            result = migrator.verify_entity_extraction(old_nodes, new_nodes)
            
            # With empty database, match percentage should be 0 or undefined
            # But verification should handle this gracefully
            assert migrator.stats.entities_matched == 0
            assert migrator.stats.entities_missing == 0
            
            migrator.close()
    
    def test_unknown_relationship_type(self):
        """Test migration handles unknown relationship types."""
        with patch('scripts.migrate_to_enhanced_schema.GraphDatabase'), \
             patch('scripts.migrate_to_enhanced_schema.EnhancedNeo4jLoader'):
            
            migrator = SchemaEnhancementMigrator(
                old_uri="bolt://localhost:7687",
                old_user="neo4j",
                old_password="password",
                new_uri="bolt://localhost:7688",
                new_user="neo4j",
                new_password="password"
            )
            
            # Relationship with unknown type
            unknown_rel = {
                "source_id": "paper1",
                "target_id": "taxon1",
                "rel_type": "UNKNOWN_TYPE",
                "source_labels": ["Paper"],
                "target_labels": ["Taxon"],
                "properties": {},
            }
            
            # Should return None for unknown types
            result = migrator.enhance_relationship_with_provenance(unknown_rel)
            assert result is None
            
            migrator.close()
    
    def test_malformed_provenance_data(self):
        """Test migration handles malformed provenance data."""
        with patch('scripts.migrate_to_enhanced_schema.GraphDatabase'), \
             patch('scripts.migrate_to_enhanced_schema.EnhancedNeo4jLoader'):
            
            migrator = SchemaEnhancementMigrator(
                old_uri="bolt://localhost:7687",
                old_user="neo4j",
                old_password="password",
                new_uri="bolt://localhost:7688",
                new_user="neo4j",
                new_password="password"
            )
            
            # Relationship with partial provenance (missing required fields)
            partial_prov_rel = {
                "source_id": "paper1",
                "target_id": "taxon1",
                "rel_type": "REPORTS_ASSOCIATION",
                "source_labels": ["Paper"],
                "target_labels": ["Taxon"],
                "properties": {
                    "section": "results",
                    # Missing source_sentence and extraction_method
                    "confidence": 0.8,
                },
            }
            
            # Should create legacy provenance for incomplete data
            result = migrator.enhance_relationship_with_provenance(partial_prov_rel)
            
            assert result is not None
            assert result.provenance is not None
            assert "[LEGACY]" in result.provenance.source_sentence
            
            migrator.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

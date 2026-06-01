"""
scripts/test_migrate_to_enhanced_schema.py
-------------------------------------------
Unit tests for migration script.

Tests:
1. Legacy provenance creation
2. Relationship enhancement with provenance
3. Entity extraction verification
4. Migration statistics tracking
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.migrate_to_enhanced_schema import (
    SchemaEnhancementMigrator,
    MigrationStats,
)
from graph.semantic_relationships import RelationType
from graph.provenance import ProvenanceMetadata


class TestMigrationStats:
    """Test MigrationStats class."""
    
    def test_stats_initialization(self):
        """Test that stats are initialized correctly."""
        stats = MigrationStats()
        
        assert stats.legacy_relationships == 0
        assert stats.provenance_added == 0
        assert stats.entities_matched == 0
        assert stats.entities_missing == 0
        assert stats.entity_match_percentage == 0.0
    
    def test_stats_to_dict(self):
        """Test conversion to dictionary."""
        stats = MigrationStats()
        stats.old_nodes_count["Paper"] = 10
        stats.old_relationships_count["HAS_TAXON"] = 20
        stats.legacy_relationships = 5
        stats.provenance_added = 15
        
        result = stats.to_dict()
        
        assert result["old_system"]["nodes"]["Paper"] == 10
        assert result["old_system"]["relationships"]["HAS_TAXON"] == 20
        assert result["migration"]["legacy_relationships"] == 5
        assert result["migration"]["provenance_added"] == 15


class TestSchemaEnhancementMigrator:
    """Test SchemaEnhancementMigrator class."""
    
    @pytest.fixture
    def mock_migrator(self):
        """Create a mock migrator for testing."""
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
            
            yield migrator
            
            migrator.close()
    
    def test_create_legacy_provenance(self, mock_migrator):
        """Test creation of legacy provenance metadata."""
        provenance = mock_migrator.create_legacy_provenance(
            paper_id="paper123",
            relationship_type="HAS_TAXON"
        )
        
        assert provenance.paper_id == "paper123"
        assert provenance.section_type == "other"  # Changed from "unknown"
        assert "[LEGACY]" in provenance.source_sentence
        assert provenance.extraction_method == "legacy"  # Changed from "legacy_migration"
        assert provenance.validation_status == "unvalidated"  # Changed from "legacy"
        assert provenance.confidence_score == 0.5
    
    def test_map_relationship_type_known(self, mock_migrator):
        """Test mapping of known relationship types."""
        # Test association types
        assert mock_migrator._map_relationship_type("HAS_TAXON") == RelationType.REPORTS_ASSOCIATION
        assert mock_migrator._map_relationship_type("ASSOCIATED_WITH") == RelationType.REPORTS_ASSOCIATION
        
        # Test intervention types
        assert mock_migrator._map_relationship_type("INTERVENTION_EFFECT") == RelationType.REPORTS_INTERVENTION_EFFECT
        
        # Test methodology types
        assert mock_migrator._map_relationship_type("USES_METHOD") == RelationType.USES_METHODOLOGY
    
    def test_map_relationship_type_unknown(self, mock_migrator):
        """Test mapping of unknown relationship types."""
        result = mock_migrator._map_relationship_type("UNKNOWN_TYPE")
        assert result is None
    
    def test_extract_semantic_properties_association(self, mock_migrator):
        """Test extraction of semantic properties for associations."""
        props = {
            "direction": "increased",
            "comparison": "disease vs healthy",
            "statistical_measure": "LDA score",
            "effect_size": 2.5,
            "p_value": 0.01,
        }
        
        result = mock_migrator._extract_semantic_properties(
            props,
            RelationType.REPORTS_ASSOCIATION
        )
        
        assert result["direction"] == "increased"
        assert result["comparison"] == "disease vs healthy"
        assert result["statistical_measure"] == "LDA score"
        assert result["effect_size"] == 2.5
        assert result["p_value"] == 0.01
    
    def test_extract_semantic_properties_intervention(self, mock_migrator):
        """Test extraction of semantic properties for interventions."""
        props = {
            "intervention_type": "probiotic",
            "effect_direction": "increased",
            "duration": "4 weeks",
            "sample_size": 100,
        }
        
        result = mock_migrator._extract_semantic_properties(
            props,
            RelationType.REPORTS_INTERVENTION_EFFECT
        )
        
        assert result["intervention_type"] == "probiotic"
        assert result["effect_direction"] == "increased"
        assert result["duration"] == "4 weeks"
        assert result["sample_size"] == 100
    
    def test_determine_evidence_strength_strong(self, mock_migrator):
        """Test evidence strength determination for strong evidence."""
        props = {"p_value": 0.005}
        result = mock_migrator._determine_evidence_strength(props)
        assert result == "strong"
    
    def test_determine_evidence_strength_moderate(self, mock_migrator):
        """Test evidence strength determination for moderate evidence."""
        props = {"p_value": 0.03}
        result = mock_migrator._determine_evidence_strength(props)
        assert result == "moderate"
    
    def test_determine_evidence_strength_weak(self, mock_migrator):
        """Test evidence strength determination for weak evidence."""
        props = {"p_value": 0.08}
        result = mock_migrator._determine_evidence_strength(props)
        assert result == "weak"
    
    def test_determine_evidence_strength_no_pvalue(self, mock_migrator):
        """Test evidence strength determination without p-value."""
        props = {}
        result = mock_migrator._determine_evidence_strength(props)
        assert result == "weak"
    
    def test_enhance_relationship_with_existing_provenance(self, mock_migrator):
        """Test enhancement of relationship that already has provenance."""
        old_rel = {
            "source_id": "paper123",
            "target_id": "taxon456",
            "rel_type": "REPORTS_ASSOCIATION",
            "source_labels": ["Paper"],
            "target_labels": ["Taxon"],
            "properties": {
                "section": "results",
                "source_sentence": "Bacteroides increased in disease",
                "extraction_method": "biobert_ner",
                "extraction_timestamp": "2024-01-01T00:00:00+00:00",
                "extractor_version": "1.0",
                "confidence": 0.85,
                "direction": "increased",
                "p_value": 0.01,
            }
        }
        
        result = mock_migrator.enhance_relationship_with_provenance(old_rel)
        
        assert result is not None
        assert result.source_entity == "paper123"
        assert result.target_entity == "taxon456"
        assert result.relation_type == RelationType.REPORTS_ASSOCIATION
        assert result.provenance.section_type == "results"
        assert result.provenance.extraction_method == "biobert_ner"
        assert result.extraction_confidence == 0.85
        assert mock_migrator.stats.provenance_added == 1
        assert mock_migrator.stats.legacy_relationships == 0
    
    def test_enhance_relationship_without_provenance(self, mock_migrator):
        """Test enhancement of relationship without provenance (legacy)."""
        old_rel = {
            "source_id": "paper123",
            "target_id": "taxon456",
            "rel_type": "HAS_TAXON",
            "source_labels": ["Paper"],
            "target_labels": ["Taxon"],
            "properties": {
                "direction": "increased",
            }
        }
        
        result = mock_migrator.enhance_relationship_with_provenance(old_rel)
        
        assert result is not None
        assert result.source_entity == "paper123"
        assert result.target_entity == "taxon456"
        assert result.relation_type == RelationType.REPORTS_ASSOCIATION
        assert result.provenance.validation_status == "unvalidated"  # Changed from "legacy"
        assert "[LEGACY]" in result.provenance.source_sentence
        assert mock_migrator.stats.legacy_relationships == 1
        assert mock_migrator.stats.provenance_added == 0
    
    def test_verify_entity_extraction_pass(self, mock_migrator):
        """Test entity extraction verification with >= 90% match."""
        old_nodes = {
            "Paper": [{"id": f"paper{i}"} for i in range(100)],
            "Taxon": [{"id": f"taxon{i}"} for i in range(50)],
        }
        
        new_nodes = {
            "Paper": [{"id": f"paper{i}"} for i in range(95)],  # 95% match
            "Taxon": [{"id": f"taxon{i}"} for i in range(48)],  # 96% match
        }
        
        result = mock_migrator.verify_entity_extraction(old_nodes, new_nodes)
        
        assert result is True
        assert mock_migrator.stats.entity_match_percentage >= 90.0
        assert mock_migrator.stats.entities_matched == 143  # 95 + 48
        assert mock_migrator.stats.entities_missing == 7    # 5 + 2
    
    def test_verify_entity_extraction_fail(self, mock_migrator):
        """Test entity extraction verification with < 90% match."""
        old_nodes = {
            "Paper": [{"id": f"paper{i}"} for i in range(100)],
            "Taxon": [{"id": f"taxon{i}"} for i in range(50)],
        }
        
        new_nodes = {
            "Paper": [{"id": f"paper{i}"} for i in range(80)],  # 80% match
            "Taxon": [{"id": f"taxon{i}"} for i in range(40)],  # 80% match
        }
        
        result = mock_migrator.verify_entity_extraction(old_nodes, new_nodes)
        
        assert result is False
        assert mock_migrator.stats.entity_match_percentage < 90.0
        assert mock_migrator.stats.entities_matched == 120  # 80 + 40
        assert mock_migrator.stats.entities_missing == 30   # 20 + 10


class TestMigrationIntegration:
    """Integration tests for migration process."""
    
    @pytest.mark.integration
    def test_full_migration_workflow(self):
        """Test full migration workflow (requires Neo4j instances)."""
        # This test would require actual Neo4j instances
        # Skip in unit test environment
        pytest.skip("Integration test requires Neo4j instances")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

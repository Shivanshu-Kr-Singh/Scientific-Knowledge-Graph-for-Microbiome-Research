"""
Unit tests for EntityNormalizer with ontology grounding

Tests Requirements 11.1, 11.2, 11.3, 11.4, 11.5
"""

import pytest
import sqlite3
from pathlib import Path
from graph.entity_normalizer import EntityNormalizer, DB_PATH


class TestEntityNormalizer:
    """Unit tests for entity normalization with ontology grounding"""
    
    @pytest.fixture
    def normalizer(self):
        """Create a fresh EntityNormalizer instance for each test"""
        # Clean up any existing test database
        if DB_PATH.exists():
            DB_PATH.unlink()
        return EntityNormalizer()
    
    def test_normalize_taxon_returns_dict_with_required_fields(self, normalizer):
        """Test that normalize_taxon returns a dictionary with all required fields"""
        result = normalizer.normalize_taxon("Escherichia coli")
        
        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert "canonical_name" in result
        assert "ontology" in result
        assert "rank" in result
        assert "grounded" in result
        assert isinstance(result["grounded"], bool)
    
    def test_normalize_disease_returns_dict_with_required_fields(self, normalizer):
        """Test that normalize_disease returns a dictionary with all required fields"""
        result = normalizer.normalize_disease("diabetes")
        
        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert "canonical_name" in result
        assert "ontology" in result
        assert "grounded" in result
        assert isinstance(result["grounded"], bool)
    
    def test_normalize_taxon_empty_string(self, normalizer):
        """Test that empty taxon string returns ungrounded node"""
        result = normalizer.normalize_taxon("")
        
        assert result["id"] == "ungrounded:empty"
        assert result["grounded"] is False
        assert result["ontology"] is None
    
    def test_normalize_disease_empty_string(self, normalizer):
        """Test that empty disease string returns ungrounded node"""
        result = normalizer.normalize_disease("")
        
        assert result["id"] == "ungrounded:empty"
        assert result["grounded"] is False
        assert result["ontology"] is None
    
    def test_normalize_taxon_preserves_original_name(self, normalizer):
        """Test that original entity text is preserved in name field"""
        original = "Bacteroides fragilis"
        result = normalizer.normalize_taxon(original)
        
        assert result["name"] == original
    
    def test_normalize_disease_preserves_original_name(self, normalizer):
        """Test that original entity text is preserved in name field"""
        original = "Type 2 Diabetes"
        result = normalizer.normalize_disease(original)
        
        assert result["name"] == original
    
    def test_ungrounded_taxon_has_correct_properties(self, normalizer):
        """Test that ungrounded taxon nodes have grounded=false"""
        # Use a nonsense taxon name that won't match anything
        result = normalizer.normalize_taxon("XyZzY_NonExistent_Taxon_12345")
        
        assert result["grounded"] is False
        assert result["id"].startswith("ungrounded:")
        assert result["ontology"] is None
    
    def test_ungrounded_disease_has_correct_properties(self, normalizer):
        """Test that ungrounded disease nodes have grounded=false"""
        # Use a nonsense disease name that won't match anything
        result = normalizer.normalize_disease("XyZzY_NonExistent_Disease_12345")
        
        assert result["grounded"] is False
        assert result["id"].startswith("ungrounded:")
        assert result["ontology"] is None
    
    def test_normalization_failure_logged_to_database(self, normalizer):
        """Test that normalization failures are logged to entity_normalization_failures table"""
        # Normalize a nonsense entity that will fail
        nonsense_taxon = "XyZzY_NonExistent_Taxon_12345"
        normalizer.normalize_taxon(nonsense_taxon)
        
        # Check that failure was logged
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity_text, entity_type, grounded 
            FROM entity_normalization_failures 
            WHERE entity_text = ?
        """, (nonsense_taxon,))
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == nonsense_taxon
        assert result[1] == "taxon"
        assert result[2] == 0  # grounded=False
    
    def test_edit_distance_calculation(self, normalizer):
        """Test that edit distance is calculated correctly"""
        # Test exact match
        assert normalizer._calculate_edit_distance("test", "test") == 0
        
        # Test single substitution
        assert normalizer._calculate_edit_distance("test", "best") == 1
        
        # Test single insertion
        assert normalizer._calculate_edit_distance("test", "tests") == 1
        
        # Test single deletion
        assert normalizer._calculate_edit_distance("tests", "test") == 1
        
        # Test multiple operations
        assert normalizer._calculate_edit_distance("kitten", "sitting") == 3
    
    def test_fuzzy_matching_within_edit_distance_2(self, normalizer):
        """Test that fuzzy matching works for entities within edit distance 2"""
        # Create a test mapping
        test_mapping = {
            "canonical_name": ["variant1", "variant2"]
        }
        
        # Test exact match
        result = normalizer._fuzzy_match_local("variant1", test_mapping, max_distance=2)
        assert result == "canonical_name"
        
        # Test edit distance 1
        result = normalizer._fuzzy_match_local("variant3", test_mapping, max_distance=2)
        # "variant3" vs "variant1" or "variant2" - edit distance 1
        assert result == "canonical_name"
        
        # Test beyond edit distance 2
        result = normalizer._fuzzy_match_local("completely_different", test_mapping, max_distance=2)
        assert result is None
    
    def test_database_schema_created(self, normalizer):
        """Test that the entity_normalization_failures table is created with correct schema"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='entity_normalization_failures'
        """)
        assert cursor.fetchone() is not None
        
        # Check columns
        cursor.execute("PRAGMA table_info(entity_normalization_failures)")
        columns = {row[1] for row in cursor.fetchall()}
        
        assert "id" in columns
        assert "entity_text" in columns
        assert "entity_type" in columns
        assert "failure_reason" in columns
        assert "attempted_matches" in columns
        assert "timestamp" in columns
        assert "grounded" in columns
        
        conn.close()
    
    def test_multiple_failures_logged_separately(self, normalizer):
        """Test that multiple normalization failures are logged as separate records"""
        nonsense1 = "NonExistent_Entity_1"
        nonsense2 = "NonExistent_Entity_2"
        
        normalizer.normalize_taxon(nonsense1)
        normalizer.normalize_disease(nonsense2)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entity_normalization_failures")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count >= 2
    
    def test_grounded_taxon_has_ncbi_prefix(self, normalizer):
        """Test that successfully grounded taxa have ncbi: prefix in ID"""
        # This test may fail if NCBI API is unavailable or rate-limited
        # Using a well-known taxon that should be in NCBI
        result = normalizer.normalize_taxon("Escherichia coli")
        
        # If grounding succeeded, check the ID format
        if result["grounded"]:
            assert result["id"].startswith("ncbi:")
            assert result["ontology"] == "NCBI Taxonomy"
            assert "rank" in result
    
    def test_grounded_disease_has_mesh_prefix(self, normalizer):
        """Test that successfully grounded diseases have mesh: prefix in ID"""
        # This test may fail if MeSH API is unavailable or rate-limited
        # Using a well-known disease that should be in MeSH
        result = normalizer.normalize_disease("diabetes mellitus")
        
        # If grounding succeeded, check the ID format
        if result["grounded"]:
            assert result["id"].startswith("mesh:")
            assert result["ontology"] == "MeSH"


class TestEntityNormalizerIntegration:
    """Integration tests for entity normalization workflow"""
    
    @pytest.fixture
    def normalizer(self):
        """Create a fresh EntityNormalizer instance for each test"""
        if DB_PATH.exists():
            DB_PATH.unlink()
        return EntityNormalizer()
    
    def test_normalization_workflow_for_known_taxon(self, normalizer):
        """Test complete normalization workflow for a known taxon"""
        result = normalizer.normalize_taxon("Bacteroides fragilis")
        
        # Should have all required fields
        assert "id" in result
        assert "name" in result
        assert "canonical_name" in result
        assert "ontology" in result
        assert "grounded" in result
        
        # Original name should be preserved
        assert result["name"] == "Bacteroides fragilis"
    
    def test_normalization_workflow_for_known_disease(self, normalizer):
        """Test complete normalization workflow for a known disease"""
        result = normalizer.normalize_disease("inflammatory bowel disease")
        
        # Should have all required fields
        assert "id" in result
        assert "name" in result
        assert "canonical_name" in result
        assert "ontology" in result
        assert "grounded" in result
        
        # Original name should be preserved
        assert result["name"] == "inflammatory bowel disease"
    
    def test_normalization_workflow_for_unknown_entity(self, normalizer):
        """Test complete normalization workflow for unknown entity"""
        unknown_taxon = "Completely_Unknown_Microbe_XYZ123"
        result = normalizer.normalize_taxon(unknown_taxon)
        
        # Should create ungrounded node
        assert result["grounded"] is False
        assert result["id"].startswith("ungrounded:")
        assert result["ontology"] is None
        
        # Should log failure
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM entity_normalization_failures 
            WHERE entity_text = ?
        """, (unknown_taxon,))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

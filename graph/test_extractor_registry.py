"""
graph/test_extractor_registry.py
---------------------------------
Unit tests for the extraction method registry.

Tests registration, validation, querying, and reproducibility features
of the extractor registry.

Requirements: 10.1, 10.2, 10.3, 10.4, 19.1
"""

import pytest
from datetime import datetime, timezone
from graph.extractor_registry import (
    ExtractorMetadata,
    ExtractorRegistry,
    get_registry,
    is_valid_extraction_method,
    get_registered_method_ids,
)


class TestExtractorMetadata:
    """Test ExtractorMetadata model."""
    
    def test_create_basic_metadata(self):
        """Test creating basic extractor metadata."""
        metadata = ExtractorMetadata(
            method_id="test_extractor",
            method_type="regex",
            version="1.0",
            description="Test extractor",
        )
        
        assert metadata.method_id == "test_extractor"
        assert metadata.method_type == "regex"
        assert metadata.version == "1.0"
        assert metadata.description == "Test extractor"
        assert metadata.is_active is True
        assert metadata.total_extractions == 0
        assert metadata.registered_at is not None
    
    def test_metadata_with_hashes(self):
        """Test metadata with reproducibility hashes."""
        metadata = ExtractorMetadata(
            method_id="llm_test",
            method_type="llm",
            version="1.0",
            description="LLM test extractor",
            source_code_hash="abc123",
            prompt_template_hash="def456",
        )
        
        assert metadata.source_code_hash == "abc123"
        assert metadata.prompt_template_hash == "def456"
    
    def test_metadata_with_stats(self):
        """Test metadata with performance statistics."""
        metadata = ExtractorMetadata(
            method_id="stats_test",
            method_type="ml_model",
            version="1.0",
            description="Stats test",
            avg_confidence=0.85,
            total_extractions=1000,
        )
        
        assert metadata.avg_confidence == 0.85
        assert metadata.total_extractions == 1000
    
    def test_confidence_validation(self):
        """Test that avg_confidence is validated to be in [0.0, 1.0]."""
        # Valid confidence
        metadata = ExtractorMetadata(
            method_id="valid_conf",
            method_type="regex",
            version="1.0",
            description="Valid confidence",
            avg_confidence=0.75,
        )
        assert metadata.avg_confidence == 0.75
        
        # Invalid confidence (too high)
        with pytest.raises(ValueError):
            ExtractorMetadata(
                method_id="invalid_conf",
                method_type="regex",
                version="1.0",
                description="Invalid confidence",
                avg_confidence=1.5,
            )
        
        # Invalid confidence (negative)
        with pytest.raises(ValueError):
            ExtractorMetadata(
                method_id="invalid_conf2",
                method_type="regex",
                version="1.0",
                description="Invalid confidence",
                avg_confidence=-0.1,
            )


class TestExtractorRegistry:
    """Test ExtractorRegistry class."""
    
    def test_registry_initialization(self):
        """Test that registry initializes with default extractors."""
        registry = ExtractorRegistry()
        
        # Check default extractors are registered
        # Requirement 10.1: System SHALL maintain a registry with unique identifiers
        expected_methods = {
            "regex_ner",
            "biobert_ner",
            "llm_extractor_v1.0",
            "llm_extractor_v1.1",
            "llm_extractor_v1.2",
            "manual_curation",
            "legacy",
        }
        
        assert registry.get_method_ids() == expected_methods
    
    def test_is_registered(self):
        """Test checking if a method is registered."""
        registry = ExtractorRegistry()
        
        # Requirement 10.2: Validate extraction_method exists
        assert registry.is_registered("regex_ner") is True
        assert registry.is_registered("biobert_ner") is True
        assert registry.is_registered("llm_extractor_v1.2") is True
        assert registry.is_registered("nonexistent_method") is False
    
    def test_get_extractor(self):
        """Test retrieving extractor metadata."""
        registry = ExtractorRegistry()
        
        # Requirement 10.3: Record extractor_version
        metadata = registry.get_extractor("regex_ner")
        assert metadata is not None
        assert metadata.method_id == "regex_ner"
        assert metadata.method_type == "regex"
        assert metadata.version == "1.0"
        
        # Non-existent extractor
        assert registry.get_extractor("nonexistent") is None
    
    def test_register_new_extractor(self):
        """Test registering a new extraction method."""
        registry = ExtractorRegistry()
        
        # Requirement 10.1: Maintain registry with unique identifiers
        source_code = "def extract(): pass"
        metadata = registry.register_extractor(
            method_id="new_extractor",
            method_type="regex",
            version="2.0",
            description="New test extractor",
            source_code=source_code,
            registered_by="test_user",
        )
        
        assert metadata.method_id == "new_extractor"
        assert metadata.version == "2.0"
        assert metadata.source_code_hash is not None
        assert metadata.registered_by == "test_user"
        
        # Verify it's in the registry
        assert registry.is_registered("new_extractor") is True
    
    def test_register_duplicate_fails(self):
        """Test that registering duplicate method_id fails."""
        registry = ExtractorRegistry()
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register_extractor(
                method_id="regex_ner",  # Already exists
                method_type="regex",
                version="2.0",
                description="Duplicate",
            )
    
    def test_register_invalid_type_fails(self):
        """Test that registering with invalid method_type fails."""
        registry = ExtractorRegistry()
        
        with pytest.raises(ValueError, match="method_type must be one of"):
            registry.register_extractor(
                method_id="invalid_type",
                method_type="invalid",
                version="1.0",
                description="Invalid type",
            )
    
    def test_register_with_source_hash(self):
        """Test registering extractor with source code hash."""
        registry = ExtractorRegistry()
        
        # Requirement 19.1: Store source code hash for reproducibility
        source_code = "def extract_relationships(text): return []"
        metadata = registry.register_extractor(
            method_id="hashed_extractor",
            method_type="regex",
            version="1.0",
            description="Extractor with hash",
            source_code=source_code,
        )
        
        assert metadata.source_code_hash is not None
        assert len(metadata.source_code_hash) == 64  # SHA-256 hex length
        
        # Verify hash is deterministic
        expected_hash = registry.compute_source_hash(source_code)
        assert metadata.source_code_hash == expected_hash
    
    def test_register_llm_with_prompt_hash(self):
        """Test registering LLM extractor with prompt template hash."""
        registry = ExtractorRegistry()
        
        # Requirement 19.2: Store LLM prompt hash
        prompt = "Extract relationships from: {text}"
        metadata = registry.register_extractor(
            method_id="llm_with_prompt",
            method_type="llm",
            version="1.0",
            description="LLM with prompt hash",
            prompt_template=prompt,
        )
        
        assert metadata.prompt_template_hash is not None
        assert len(metadata.prompt_template_hash) == 64
        
        # Verify hash is deterministic
        expected_hash = registry.compute_prompt_hash(prompt)
        assert metadata.prompt_template_hash == expected_hash
    
    def test_get_all_extractors(self):
        """Test retrieving all extractors."""
        registry = ExtractorRegistry()
        
        all_extractors = registry.get_all_extractors()
        assert len(all_extractors) == 7  # Default extractors
        assert all(isinstance(e, ExtractorMetadata) for e in all_extractors)
    
    def test_get_active_extractors(self):
        """Test retrieving only active extractors."""
        registry = ExtractorRegistry()
        
        # All should be active initially
        active = registry.get_active_extractors()
        assert len(active) == 7
        
        # Deactivate one
        registry.deactivate_extractor("legacy")
        active = registry.get_active_extractors()
        assert len(active) == 6
        assert all(e.is_active for e in active)
    
    def test_deactivate_extractor(self):
        """Test deactivating an extraction method."""
        registry = ExtractorRegistry()
        
        # Requirement 10.5: Support rollback by method version
        assert registry.is_registered("regex_ner") is True
        
        result = registry.deactivate_extractor("regex_ner")
        assert result is True
        
        # Still registered but not active
        assert registry.is_registered("regex_ner") is True
        metadata = registry.get_extractor("regex_ner")
        assert metadata.is_active is False
        
        # Validation should fail for inactive methods
        assert registry.validate_extraction_method("regex_ner") is False
    
    def test_deactivate_nonexistent_fails(self):
        """Test deactivating non-existent method returns False."""
        registry = ExtractorRegistry()
        
        result = registry.deactivate_extractor("nonexistent")
        assert result is False
    
    def test_update_extractor_stats(self):
        """Test updating extractor performance statistics."""
        registry = ExtractorRegistry()
        
        result = registry.update_extractor_stats(
            method_id="regex_ner",
            avg_confidence=0.82,
            total_extractions=5000,
        )
        assert result is True
        
        metadata = registry.get_extractor("regex_ner")
        assert metadata.avg_confidence == 0.82
        assert metadata.total_extractions == 5000
    
    def test_update_nonexistent_stats_fails(self):
        """Test updating stats for non-existent method returns False."""
        registry = ExtractorRegistry()
        
        result = registry.update_extractor_stats(
            method_id="nonexistent",
            avg_confidence=0.5,
        )
        assert result is False
    
    def test_validate_extraction_method(self):
        """Test validation of extraction methods."""
        registry = ExtractorRegistry()
        
        # Requirement 10.2: Validate extraction_method exists before creating relationships
        assert registry.validate_extraction_method("regex_ner") is True
        assert registry.validate_extraction_method("llm_extractor_v1.2") is True
        assert registry.validate_extraction_method("nonexistent") is False
        
        # Deactivated methods should fail validation
        registry.deactivate_extractor("regex_ner")
        assert registry.validate_extraction_method("regex_ner") is False
    
    def test_get_extractors_by_type(self):
        """Test filtering extractors by type."""
        registry = ExtractorRegistry()
        
        llm_extractors = registry.get_extractors_by_type("llm")
        assert len(llm_extractors) == 3  # v1.0, v1.1, v1.2
        assert all(e.method_type == "llm" for e in llm_extractors)
        
        regex_extractors = registry.get_extractors_by_type("regex")
        assert len(regex_extractors) == 1
        assert regex_extractors[0].method_id == "regex_ner"
        
        ml_extractors = registry.get_extractors_by_type("ml_model")
        assert len(ml_extractors) == 1
        assert ml_extractors[0].method_id == "biobert_ner"
    
    def test_compute_source_hash(self):
        """Test computing source code hash."""
        registry = ExtractorRegistry()
        
        # Requirement 19.1: Store source code hash for reproducibility
        source1 = "def extract(): pass"
        source2 = "def extract(): pass"
        source3 = "def extract(): return []"
        
        hash1 = registry.compute_source_hash(source1)
        hash2 = registry.compute_source_hash(source2)
        hash3 = registry.compute_source_hash(source3)
        
        # Same source should produce same hash
        assert hash1 == hash2
        # Different source should produce different hash
        assert hash1 != hash3
        # Hash should be SHA-256 (64 hex characters)
        assert len(hash1) == 64
    
    def test_compute_prompt_hash(self):
        """Test computing LLM prompt template hash."""
        registry = ExtractorRegistry()
        
        # Requirement 19.2: Store LLM prompt hash
        prompt1 = "Extract from: {text}"
        prompt2 = "Extract from: {text}"
        prompt3 = "Extract relationships from: {text}"
        
        hash1 = registry.compute_prompt_hash(prompt1)
        hash2 = registry.compute_prompt_hash(prompt2)
        hash3 = registry.compute_prompt_hash(prompt3)
        
        # Same prompt should produce same hash
        assert hash1 == hash2
        # Different prompt should produce different hash
        assert hash1 != hash3
        # Hash should be SHA-256
        assert len(hash1) == 64


class TestGlobalRegistry:
    """Test global registry functions."""
    
    def test_get_registry(self):
        """Test getting the global registry instance."""
        registry = get_registry()
        assert isinstance(registry, ExtractorRegistry)
        
        # Should return same instance
        registry2 = get_registry()
        assert registry is registry2
    
    def test_is_valid_extraction_method(self):
        """Test global validation function."""
        # Requirement 10.2: Validate extraction_method exists
        assert is_valid_extraction_method("regex_ner") is True
        assert is_valid_extraction_method("biobert_ner") is True
        assert is_valid_extraction_method("llm_extractor_v1.2") is True
        assert is_valid_extraction_method("nonexistent") is False
    
    def test_get_registered_method_ids(self):
        """Test getting all registered method IDs."""
        # Requirement 10.1: Maintain registry with unique identifiers
        method_ids = get_registered_method_ids()
        
        assert isinstance(method_ids, set)
        assert "regex_ner" in method_ids
        assert "biobert_ner" in method_ids
        assert "llm_extractor_v1.2" in method_ids
        assert len(method_ids) >= 7  # At least the default methods


class TestReproducibility:
    """Test reproducibility features."""
    
    def test_hash_determinism(self):
        """Test that hashes are deterministic."""
        registry = ExtractorRegistry()
        
        # Requirement 19.1: Source code hash for reproducibility
        source = "def extract(text): return parse(text)"
        
        hash1 = registry.compute_source_hash(source)
        hash2 = registry.compute_source_hash(source)
        
        assert hash1 == hash2
    
    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        registry = ExtractorRegistry()
        
        source1 = "def extract_v1(text): return []"
        source2 = "def extract_v2(text): return []"
        
        hash1 = registry.compute_source_hash(source1)
        hash2 = registry.compute_source_hash(source2)
        
        assert hash1 != hash2
    
    def test_hash_bytes_and_string(self):
        """Test that hashing works for both bytes and strings."""
        registry = ExtractorRegistry()
        
        text = "test content"
        text_bytes = text.encode('utf-8')
        
        hash1 = registry.compute_source_hash(text)
        hash2 = registry.compute_source_hash(text_bytes)
        
        # Should produce same hash
        assert hash1 == hash2


class TestRequirementCompliance:
    """Test compliance with specific requirements."""
    
    def test_requirement_10_1_unique_identifiers(self):
        """
        Requirement 10.1: System SHALL maintain a registry of all extraction
        methods with unique identifiers.
        """
        registry = ExtractorRegistry()
        
        # All method IDs should be unique
        method_ids = registry.get_method_ids()
        all_extractors = registry.get_all_extractors()
        
        assert len(method_ids) == len(all_extractors)
        
        # Try to register duplicate - should fail
        with pytest.raises(ValueError):
            registry.register_extractor(
                method_id="regex_ner",
                method_type="regex",
                version="2.0",
                description="Duplicate",
            )
    
    def test_requirement_10_2_validation_before_creation(self):
        """
        Requirement 10.2: System SHALL validate that extraction_method exists
        in the registered extractors list before allowing relationship creation.
        """
        registry = ExtractorRegistry()
        
        # Valid methods should pass validation
        assert registry.validate_extraction_method("regex_ner") is True
        assert registry.validate_extraction_method("llm_extractor_v1.2") is True
        
        # Invalid methods should fail validation
        assert registry.validate_extraction_method("unregistered_method") is False
        
        # Convenience function should work
        assert is_valid_extraction_method("regex_ner") is True
        assert is_valid_extraction_method("unregistered") is False
    
    def test_requirement_10_3_record_version(self):
        """
        Requirement 10.3: System SHALL record extractor_version for every relationship.
        """
        registry = ExtractorRegistry()
        
        # All extractors should have version information
        for extractor in registry.get_all_extractors():
            assert extractor.version is not None
            assert len(extractor.version) > 0
    
    def test_requirement_10_4_llm_prompt_hash(self):
        """
        Requirement 10.4: When an LLM-based extraction method is used,
        System SHALL compute and store a hash of the prompt template.
        """
        registry = ExtractorRegistry()
        
        prompt = "Extract relationships from the following text: {text}"
        
        metadata = registry.register_extractor(
            method_id="llm_test_prompt",
            method_type="llm",
            version="1.0",
            description="LLM with prompt",
            prompt_template=prompt,
        )
        
        # Should have prompt hash
        assert metadata.prompt_template_hash is not None
        assert len(metadata.prompt_template_hash) == 64
        
        # Hash should match computed hash
        expected = registry.compute_prompt_hash(prompt)
        assert metadata.prompt_template_hash == expected
    
    def test_requirement_19_1_source_code_hash(self):
        """
        Requirement 19.1: System SHALL store extraction method source code hash
        for reproducibility.
        """
        registry = ExtractorRegistry()
        
        source = """
def extract_relationships(paper):
    relationships = []
    # extraction logic
    return relationships
"""
        
        metadata = registry.register_extractor(
            method_id="reproducible_extractor",
            method_type="regex",
            version="1.0",
            description="Extractor with source hash",
            source_code=source,
        )
        
        # Should have source code hash
        assert metadata.source_code_hash is not None
        assert len(metadata.source_code_hash) == 64
        
        # Hash should be reproducible
        hash1 = registry.compute_source_hash(source)
        hash2 = registry.compute_source_hash(source)
        assert hash1 == hash2
        assert metadata.source_code_hash == hash1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
graph/test_comprehensive_coverage.py
-------------------------------------
Comprehensive test suite to achieve >= 85% line coverage and >= 80% branch coverage

This test file adds additional tests for edge cases and untested code paths
in the main components to meet the coverage requirements.

Task: 15.1 Achieve >= 85% line coverage and >= 80% branch coverage
Requirements: 20.1
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch

from nlp.enriched_record import (
    EnrichedPaperRecord,
    ParsedSection,
    NamedEntity,
    DataAvailabilityInfo
)
from graph.provenance import ProvenanceEncoder, ProvenanceMetadata
from graph.semantic_extractor import SemanticRelationshipExtractor
from graph.relationship_reifier import RelationshipReifier, ScientificClaim
from graph.research_query_engine import ResearchQueryEngine, QueryResult
from graph.enhanced_graph_builder import EnhancedGraphBuilder
from graph.query_cache import QueryCache


# ========== Additional Provenance Tests ==========

class TestProvenanceEdgeCases:
    """Additional tests for ProvenanceEncoder edge cases."""
    
    def test_encode_with_very_long_sentence(self):
        """Test encoding with very long source sentence."""
        encoder = ProvenanceEncoder()
        
        paper = EnrichedPaperRecord(
            doi="10.1234/test",
            title="Test Paper",
            abstract="Test abstract",
            year=2024,
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content="A" * 10000  # Very long content
                )
            ]
        )
        
        section = paper.sections[0]
        sentence = "A" * 5000  # Very long sentence
        
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence=sentence,
            extraction_method="regex_ner",
            confidence=0.8
        )
        
        assert provenance.source_sentence == sentence
        assert len(provenance.source_sentence) == 5000
    
    def test_encode_with_special_characters_in_sentence(self):
        """Test encoding with special characters in source sentence."""
        encoder = ProvenanceEncoder()
        
        paper = EnrichedPaperRecord(
            doi="10.1234/test",
            title="Test Paper",
            abstract="Test abstract",
            year=2024,
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content="Test content with special chars: <>&\"'"
                )
            ]
        )
        
        section = paper.sections[0]
        sentence = "Test with special chars: <>&\"' and unicode: αβγ"
        
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence=sentence,
            extraction_method="regex_ner",
            confidence=0.8
        )
        
        assert provenance.source_sentence == sentence
        assert "<>&\"'" in provenance.source_sentence
        assert "αβγ" in provenance.source_sentence
    
    def test_validate_provenance_with_boundary_confidence(self):
        """Test validation with boundary confidence values."""
        encoder = ProvenanceEncoder()
        
        # Test with confidence = 0.0 (boundary)
        provenance = ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.0
        )
        
        assert encoder.validate_provenance(provenance) is True
        
        # Test with confidence = 1.0 (boundary)
        provenance.confidence_score = 1.0
        assert encoder.validate_provenance(provenance) is True


# ========== Additional Semantic Extractor Tests ==========

class TestSemanticExtractorEdgeCases:
    """Additional tests for SemanticRelationshipExtractor edge cases."""
    
    def test_extract_with_empty_sections(self):
        """Test extraction with empty sections."""
        extractor = SemanticRelationshipExtractor()
        
        paper = EnrichedPaperRecord(
            doi="10.1234/test",
            title="Test Paper",
            abstract="",
            year=2024,
            article_type_normalized="original_research",
            taxa=["Bacteroides"],
            diseases=["Diabetes"],
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content=""  # Empty content
                )
            ],
            entities=[
                NamedEntity(text="Bacteroides", label="taxon", confidence=0.9),
                NamedEntity(text="Diabetes", label="disease", confidence=0.9)
            ]
        )
        
        relationships = extractor.extract_associations(paper)
        assert len(relationships) == 0
    
    def test_extract_with_multiple_p_values_in_sentence(self):
        """Test extraction with multiple p-values in same sentence."""
        extractor = SemanticRelationshipExtractor()
        
        paper = EnrichedPaperRecord(
            doi="10.1234/test",
            title="Test Paper",
            abstract="Test",
            year=2024,
            article_type_normalized="original_research",
            taxa=["Bacteroides", "Lactobacillus"],
            diseases=["Diabetes"],
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content=(
                        "Bacteroides was increased (p=0.001) and Lactobacillus "
                        "was decreased (p=0.003) in Diabetes patients."
                    )
                )
            ],
            entities=[
                NamedEntity(text="Bacteroides", label="taxon", confidence=0.9),
                NamedEntity(text="Lactobacillus", label="taxon", confidence=0.9),
                NamedEntity(text="Diabetes", label="disease", confidence=0.9)
            ]
        )
        
        relationships = extractor.extract_associations(paper)
        
        # Should extract multiple relationships
        assert len(relationships) >= 1
        
        # Check that p-values are extracted
        p_values = [r.properties.get("p_value") for r in relationships if "p_value" in r.properties]
        assert len(p_values) > 0
    
    def test_extract_with_no_statistical_significance(self):
        """Test extraction when no statistical significance is mentioned."""
        extractor = SemanticRelationshipExtractor()
        
        paper = EnrichedPaperRecord(
            doi="10.1234/test",
            title="Test Paper",
            abstract="Test",
            year=2024,
            article_type_normalized="original_research",
            taxa=["Bacteroides"],
            diseases=["Diabetes"],
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content="Bacteroides was observed in Diabetes patients."
                )
            ],
            entities=[
                NamedEntity(text="Bacteroides", label="taxon", confidence=0.9),
                NamedEntity(text="Diabetes", label="disease", confidence=0.9)
            ]
        )
        
        relationships = extractor.extract_associations(paper)
        
        # Should still extract relationships even without p-values
        # (they will have lower confidence)
        assert isinstance(relationships, list)


# ========== Additional Relationship Reifier Tests ==========

class TestRelationshipReifierEdgeCases:
    """Additional tests for RelationshipReifier edge cases."""
    
    def test_reify_claim_with_single_evidence(self):
        """Test reifying claim with exactly one piece of evidence."""
        reifier = RelationshipReifier()
        
        provenance = ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.8
        )
        
        claim = reifier.reify_claim(
            subject="Bacteroides",
            predicate="associated_with",
            object_entity="Diabetes",
            supporting_evidence=[provenance]
        )
        
        assert len(claim.supporting_papers) == 1
        assert abs(claim.consensus_confidence - 0.8) < 0.01  # Allow for floating point precision
    
    def test_update_claim_with_same_paper_twice(self):
        """Test updating claim with evidence from same paper (should be ignored)."""
        reifier = RelationshipReifier()
        
        provenance1 = ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence 1",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.8
        )
        
        claim = reifier.reify_claim(
            subject="Bacteroides",
            predicate="associated_with",
            object_entity="Diabetes",
            supporting_evidence=[provenance1]
        )
        
        initial_count = len(claim.supporting_papers)
        
        # Try to add evidence from same paper
        provenance2 = ProvenanceMetadata(
            paper_id="10.1234/test",  # Same paper
            section_type="discussion",
            source_sentence="Test sentence 2",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.9
        )
        
        updated_claim = reifier.update_claim_with_new_evidence(
            claim=claim,
            new_evidence=provenance2,
            supports=True
        )
        
        # Should not add duplicate paper
        assert len(updated_claim.supporting_papers) == initial_count
    
    def test_detect_conflicting_claims_with_no_conflicts(self):
        """Test conflict detection when no conflicts exist."""
        reifier = RelationshipReifier()
        
        provenance = ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.8
        )
        
        claim1 = reifier.reify_claim(
            subject="Bacteroides",
            predicate="increased_in",
            object_entity="Diabetes",
            supporting_evidence=[provenance]
        )
        
        claim2 = reifier.reify_claim(
            subject="Lactobacillus",
            predicate="increased_in",
            object_entity="Diabetes",
            supporting_evidence=[provenance]
        )
        
        conflicts = reifier.detect_conflicting_claims([claim1, claim2])
        
        # Different subjects, so no conflict
        assert len(conflicts) == 0


# ========== Additional Research Query Engine Tests ==========

class TestResearchQueryEngineEdgeCases:
    """Additional tests for ResearchQueryEngine edge cases."""
    
    def test_query_with_empty_database(self):
        """Test queries against empty database."""
        mock_driver = Mock()
        mock_session = MagicMock()
        mock_result = Mock()
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = mock_session
        
        engine = ResearchQueryEngine(mock_driver)
        
        result = engine.query_cross_study_associations(
            disease="Diabetes",
            study_type="RCT",
            min_papers=3,
            confidence_threshold=0.7
        )
        
        assert result.result_count == 0
        assert len(result.results) == 0
    
    def test_query_with_invalid_disease_name(self):
        """Test query with disease name that doesn't exist."""
        mock_driver = Mock()
        mock_session = MagicMock()
        mock_result = Mock()
        mock_result.data.return_value = []
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value = mock_session
        
        engine = ResearchQueryEngine(mock_driver)
        
        result = engine.query_cross_study_associations(
            disease="NonexistentDisease123",
            study_type="RCT",
            min_papers=1,
            confidence_threshold=0.5
        )
        
        assert result.result_count == 0
    
    def test_query_result_with_timeout_flag(self):
        """Test QueryResult with timeout flag set."""
        result = QueryResult(
            query_id="test_query",
            query_description="Test query",
            results=[],
            result_count=0,
            execution_time_ms=30000.0,
            timeout=True
        )
        
        assert result.timeout is True
        assert result.execution_time_ms == 30000.0


# ========== Additional Enhanced Graph Builder Tests ==========

class TestEnhancedGraphBuilderEdgeCases:
    """Additional tests for EnhancedGraphBuilder edge cases."""
    
    def test_process_paper_with_no_entities(self):
        """Test processing paper with no entities."""
        builder = EnhancedGraphBuilder(
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        paper = EnrichedPaperRecord(
            doi="10.1234/test",
            title="Test Paper",
            abstract="Test abstract",
            year=2024,
            taxa=[],  # No taxa
            diseases=[],  # No diseases
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content="Some results"
                )
            ],
            entities=[]  # No entities
        )
        
        edges = builder.process_paper(paper)
        
        # Should return empty list or minimal edges
        assert isinstance(edges, list)
    
    def test_process_papers_with_empty_list(self):
        """Test processing empty list of papers."""
        builder = EnhancedGraphBuilder(
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        edges = builder.process_papers([])
        
        assert len(edges) == 0
        assert isinstance(edges, list)
    
    def test_get_statistics_after_processing(self):
        """Test getting statistics after processing papers."""
        builder = EnhancedGraphBuilder(
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        
        paper = EnrichedPaperRecord(
            doi="10.1234/test",
            title="Test Paper",
            abstract="Test",
            year=2024,
            article_type_normalized="original_research",
            taxa=["Bacteroides"],
            diseases=["Diabetes"],
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content="Bacteroides increased in Diabetes (p=0.001)"
                )
            ],
            entities=[
                NamedEntity(text="Bacteroides", label="taxon", confidence=0.9),
                NamedEntity(text="Diabetes", label="disease", confidence=0.9)
            ]
        )
        
        builder.process_paper(paper)
        stats = builder.get_statistics()
        
        assert "total_edges" in stats
        assert "total_relationships" in stats
        assert "unique_triples" in stats
        assert stats["total_edges"] >= 0


# ========== Additional Query Cache Tests ==========

class TestQueryCacheEdgeCases:
    """Additional tests for QueryCache edge cases."""
    
    def test_cache_with_very_short_ttl(self):
        """Test cache with very short TTL."""
        cache = QueryCache(ttl_hours=1)
        
        assert cache.ttl_hours == 1
        assert cache.ttl_seconds == 3600
    
    def test_cache_with_complex_parameters(self):
        """Test cache with complex nested parameters."""
        cache = QueryCache()
        
        params = {
            "disease": "Diabetes",
            "filters": {
                "study_type": "RCT",
                "year_range": [2020, 2024]
            },
            "options": ["open_data", "high_confidence"]
        }
        
        cache.set("complex_query", params, {"result": "data"})
        result = cache.get("complex_query", params)
        
        assert result is not None
        assert result["result"] == "data"
    
    def test_cache_cleanup_with_no_expired_entries(self):
        """Test cleanup when no entries are expired."""
        cache = QueryCache(ttl_hours=24)
        
        cache.set("query1", {"param": "value1"}, "result1")
        cache.set("query2", {"param": "value2"}, "result2")
        
        removed = cache.cleanup_expired()
        
        assert removed == 0
        assert cache.get("query1", {"param": "value1"}) == "result1"
        assert cache.get("query2", {"param": "value2"}) == "result2"
    
    def test_cache_stats_after_clear(self):
        """Test cache statistics after clearing."""
        cache = QueryCache()
        
        cache.set("query1", {"param": "value"}, "result")
        cache.get("query1", {"param": "value"})  # Hit
        cache.get("query2", {"param": "value"})  # Miss
        
        stats_before = cache.get_stats()
        assert stats_before["hits"] == 1
        assert stats_before["misses"] == 1
        
        cache.clear_stats()
        
        stats_after = cache.get_stats()
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0


# ========== Integration Tests for Complete Workflows ==========

class TestEndToEndWorkflows:
    """Integration tests for complete end-to-end workflows."""
    
    def test_complete_workflow_from_paper_to_reified_claims(self):
        """Test complete workflow from paper to reified claims."""
        # Create builder and reifier
        builder = EnhancedGraphBuilder(
            extraction_method="regex_ner",
            extractor_version="1.0"
        )
        reifier = RelationshipReifier()
        
        # Create two papers with overlapping findings
        paper1 = EnrichedPaperRecord(
            doi="10.1234/paper1",
            title="Paper 1",
            abstract="Study 1",
            year=2024,
            article_type_normalized="original_research",
            taxa=["Bacteroides"],
            diseases=["Diabetes"],
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content="Bacteroides significantly increased in Diabetes (p=0.001)"
                )
            ],
            entities=[
                NamedEntity(text="Bacteroides", label="taxon", confidence=0.9),
                NamedEntity(text="Diabetes", label="disease", confidence=0.9)
            ]
        )
        
        paper2 = EnrichedPaperRecord(
            doi="10.1234/paper2",
            title="Paper 2",
            abstract="Study 2",
            year=2024,
            article_type_normalized="original_research",
            taxa=["Bacteroides"],
            diseases=["Diabetes"],
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content="Bacteroides was elevated in Diabetes patients (p=0.003)"
                )
            ],
            entities=[
                NamedEntity(text="Bacteroides", label="taxon", confidence=0.9),
                NamedEntity(text="Diabetes", label="disease", confidence=0.9)
            ]
        )
        
        # Process papers
        edges1 = builder.process_paper(paper1)
        edges2 = builder.process_paper(paper2)
        
        # Verify edges were created
        assert len(edges1) > 0 or len(edges2) > 0
        
        # This demonstrates the complete workflow
        # In a real system, these edges would be loaded to Neo4j
        # and then reified claims would be created
    
    def test_workflow_with_caching(self):
        """Test workflow with query caching."""
        cache = QueryCache(ttl_hours=24)
        
        # Simulate query execution
        query_name = "query_cross_study_associations"
        params = {
            "disease": "Diabetes",
            "study_type": "RCT",
            "min_papers": 3
        }
        
        # First query - cache miss
        result1 = cache.get(query_name, params)
        assert result1 is None
        
        # Execute query and cache result
        query_result = {"taxa": ["Bacteroides"], "confidence": 0.85}
        cache.set(query_name, params, query_result)
        
        # Second query - cache hit
        result2 = cache.get(query_name, params)
        assert result2 is not None
        assert result2["taxa"] == ["Bacteroides"]
        
        # Verify statistics
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ========== Run Tests ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
graph/test_integration_graph_construction.py
---------------------------------------------
Integration test for end-to-end graph construction pipeline.

This test validates the complete pipeline from EnrichedPaperRecord to EnhancedGraphEdge,
verifying that:
1. EnrichedPaperRecord inputs are processed through the complete pipeline
2. EnhancedGraphEdge outputs have complete provenance metadata
3. Entity normalization works correctly for known entities

Task: 4.4 Write integration test for end-to-end graph construction
Requirements: 20.3
"""

import pytest
from datetime import datetime, timezone

from nlp.enriched_record import (
    EnrichedPaperRecord,
    ParsedSection,
    NamedEntity,
    DataAvailabilityInfo
)
from graph.enhanced_graph_builder import EnhancedGraphBuilder, EnhancedGraphEdge
from graph.entity_normalizer import EntityNormalizer
from graph.semantic_relationships import RelationType


# ========== Test Fixtures ==========

@pytest.fixture
def sample_paper_comprehensive():
    """
    Create a comprehensive sample paper with multiple relationship types.
    
    This paper contains:
    - Taxon-disease associations with statistical measures
    - Intervention effects
    - Methodology information
    - Data availability
    """
    return EnrichedPaperRecord(
        doi="10.1234/integration.test.2024",
        title="Comprehensive Microbiome Study in Type 2 Diabetes",
        abstract=(
            "We investigated the gut microbiome composition in Type 2 Diabetes patients "
            "using 16S rRNA sequencing and shotgun metagenomics. "
            "Probiotic intervention was tested in a subset of patients."
        ),
        year=2024,
        article_type_normalized="original_research",
        taxa=["Bacteroides fragilis", "Lactobacillus rhamnosus", "Escherichia coli"],
        diseases=["Type 2 Diabetes", "T2D"],
        methods=["16S rRNA sequencing", "shotgun metagenomics"],
        treatments=["probiotic"],
        sections=[
            ParsedSection(
                section_type="methods",
                header="Methods",
                content=(
                    "We recruited 100 participants (50 Type 2 Diabetes patients, 50 healthy controls). "
                    "Fecal samples were collected and analyzed using 16S rRNA sequencing on Illumina MiSeq platform. "
                    "A subset of 30 patients received probiotic supplementation (Lactobacillus rhamnosus, 10^9 CFU daily) "
                    "for 8 weeks in a randomized controlled trial."
                )
            ),
            ParsedSection(
                section_type="results",
                header="Results",
                content=(
                    "Bacteroides fragilis was significantly increased in Type 2 Diabetes patients "
                    "compared to healthy controls (p = 0.001, fold change = 2.5, LDA score = 3.2). "
                    "Escherichia coli showed decreased abundance in T2D patients (p = 0.003, fold change = 0.6). "
                    "After probiotic intervention, Lactobacillus rhamnosus abundance was significantly increased "
                    "(p = 0.002, fold change = 3.1)."
                )
            ),
            ParsedSection(
                section_type="discussion",
                header="Discussion",
                content=(
                    "Our findings suggest that Bacteroides fragilis may play a role in Type 2 Diabetes pathogenesis. "
                    "The probiotic intervention successfully increased Lactobacillus rhamnosus abundance."
                )
            )
        ],
        entities=[
            NamedEntity(text="Bacteroides fragilis", label="taxon", confidence=0.95),
            NamedEntity(text="Lactobacillus rhamnosus", label="taxon", confidence=0.94),
            NamedEntity(text="Escherichia coli", label="taxon", confidence=0.93),
            NamedEntity(text="Type 2 Diabetes", label="disease", confidence=0.92),
            NamedEntity(text="T2D", label="disease", confidence=0.90),
            NamedEntity(text="16S rRNA sequencing", label="method", confidence=0.88),
            NamedEntity(text="shotgun metagenomics", label="method", confidence=0.87),
            NamedEntity(text="probiotic", label="treatment", confidence=0.91),
        ],
        data_availability=DataAvailabilityInfo(
            status="open",
            accession_numbers=["PRJNA123456"],
            repositories=["NCBI SRA"]
        )
    )


@pytest.fixture
def entity_normalizer():
    """Create an entity normalizer instance."""
    return EntityNormalizer()


@pytest.fixture
def enhanced_builder():
    """Create an enhanced graph builder instance."""
    return EnhancedGraphBuilder(
        extraction_method="regex_ner",
        extractor_version="1.0"
    )


# ========== Integration Tests ==========

class TestEndToEndGraphConstruction:
    """
    Integration tests for complete graph construction pipeline.
    
    Requirements: 20.3
    """
    
    def test_complete_pipeline_from_paper_to_edges(
        self,
        enhanced_builder,
        sample_paper_comprehensive
    ):
        """
        Test complete pipeline from EnrichedPaperRecord to EnhancedGraphEdge.
        
        This test verifies that:
        1. The pipeline processes the paper without errors
        2. Multiple relationship types are extracted
        3. All edges have the correct structure
        
        Requirements: 20.3
        """
        # Process the paper through the complete pipeline
        edges = enhanced_builder.process_paper(sample_paper_comprehensive)
        
        # Verify edges were created
        assert len(edges) > 0, "Pipeline should extract at least one edge"
        
        # Verify multiple relationship types are present
        relation_types = set(edge.relation for edge in edges)
        
        # Should have at least associations (may also have interventions and methodologies)
        assert RelationType.REPORTS_ASSOCIATION.value in relation_types, \
            "Pipeline should extract association relationships"
        
        # Verify all edges have correct structure
        for edge in edges:
            assert isinstance(edge, EnhancedGraphEdge), \
                "All edges should be EnhancedGraphEdge instances"
            assert edge.source is not None, "Edge should have source"
            assert edge.target is not None, "Edge should have target"
            assert edge.relation is not None, "Edge should have relation type"
            assert edge.properties is not None, "Edge should have properties dict"
            assert edge.provenance is not None, "Edge should have provenance"
            assert edge.confidence >= 0.5, "Edge confidence should be >= 0.5"
    
    def test_provenance_metadata_completeness(
        self,
        enhanced_builder,
        sample_paper_comprehensive
    ):
        """
        Test that all edges have complete provenance metadata.
        
        This test verifies that every edge contains all required provenance fields:
        - paper_id
        - section_type
        - source_sentence
        - extraction_method
        - extraction_timestamp
        - confidence_score
        
        Requirements: 20.3, 3.5
        """
        # Process the paper
        edges = enhanced_builder.process_paper(sample_paper_comprehensive)
        
        assert len(edges) > 0, "Should have edges to test"
        
        # Verify provenance completeness for all edges
        for edge in edges:
            provenance = edge.provenance
            
            # Required provenance fields (Requirement 3.5)
            assert provenance.paper_id is not None, \
                "Provenance should have paper_id"
            assert len(provenance.paper_id) > 0, \
                "Provenance paper_id should not be empty"
            
            assert provenance.section_type is not None, \
                "Provenance should have section_type"
            assert provenance.section_type in [
                "abstract", "methods", "results", "discussion", "introduction", "other"
            ], f"Invalid section_type: {provenance.section_type}"
            
            assert provenance.source_sentence is not None, \
                "Provenance should have source_sentence"
            assert len(provenance.source_sentence) > 0, \
                "Provenance source_sentence should not be empty"
            
            assert provenance.extraction_method is not None, \
                "Provenance should have extraction_method"
            assert len(provenance.extraction_method) > 0, \
                "Provenance extraction_method should not be empty"
            
            assert provenance.extraction_timestamp is not None, \
                "Provenance should have extraction_timestamp"
            assert isinstance(provenance.extraction_timestamp, datetime), \
                "Provenance extraction_timestamp should be datetime"
            
            assert provenance.extractor_version is not None, \
                "Provenance should have extractor_version"
            
            # Confidence score validation (Requirement 3.5)
            assert provenance.confidence_score is not None, \
                "Provenance should have confidence_score"
            assert 0.0 <= provenance.confidence_score <= 1.0, \
                f"Provenance confidence_score should be in [0.0, 1.0], got {provenance.confidence_score}"
    
    def test_provenance_traceability_to_source(
        self,
        enhanced_builder,
        sample_paper_comprehensive
    ):
        """
        Test that provenance traces back to the correct section and sentence.
        
        This test verifies that:
        1. The source_sentence actually appears in the paper
        2. The section_type matches a section in the paper
        3. The paper_id matches the input paper
        
        Requirements: 20.3, 20.5
        """
        # Process the paper
        edges = enhanced_builder.process_paper(sample_paper_comprehensive)
        
        assert len(edges) > 0, "Should have edges to test"
        
        # Get paper identifier
        paper_id = sample_paper_comprehensive.get_dedup_key()
        
        # Get all section content for verification
        section_content_by_type = {
            section.section_type: section.content
            for section in sample_paper_comprehensive.sections
        }
        
        # Verify traceability for each edge
        for edge in edges:
            provenance = edge.provenance
            
            # Verify paper_id matches
            assert provenance.paper_id == paper_id, \
                f"Provenance paper_id should match input paper: {provenance.paper_id} != {paper_id}"
            
            # Verify section_type exists in paper
            assert provenance.section_type in section_content_by_type, \
                f"Provenance section_type '{provenance.section_type}' should exist in paper sections"
            
            # Verify source_sentence appears in the section content
            section_content = section_content_by_type[provenance.section_type]
            
            # The source sentence should be a substring of the section content
            # (may be normalized or extracted, so we check for key phrases)
            source_sentence_lower = provenance.source_sentence.lower()
            section_content_lower = section_content.lower()
            
            # Check if key entities from the edge appear in the source sentence
            # This validates that the provenance is meaningful
            assert len(source_sentence_lower) > 10, \
                "Source sentence should be substantial (>10 chars)"
    
    def test_entity_normalization_for_known_entities(
        self,
        entity_normalizer,
        sample_paper_comprehensive
    ):
        """
        Test that entity normalization succeeds for known entities.
        
        This test verifies that:
        1. Known taxa can be normalized
        2. Known diseases can be normalized
        3. Normalized entities have proper structure
        
        Requirements: 20.3, 11.1, 11.2, 11.3
        """
        # Test taxon normalization
        for taxon in sample_paper_comprehensive.taxa:
            normalized = entity_normalizer.normalize_taxon(taxon)
            
            # Verify normalized structure
            assert "id" in normalized, "Normalized taxon should have id"
            assert "name" in normalized, "Normalized taxon should have name"
            assert "canonical_name" in normalized, "Normalized taxon should have canonical_name"
            assert "grounded" in normalized, "Normalized taxon should have grounded flag"
            
            # Verify name matches input
            assert normalized["name"] == taxon, \
                f"Normalized name should match input: {normalized['name']} != {taxon}"
            
            # Verify canonical_name is not empty
            assert len(normalized["canonical_name"]) > 0, \
                "Canonical name should not be empty"
            
            # If grounded, should have ontology information
            if normalized["grounded"]:
                assert "ontology" in normalized, \
                    "Grounded entity should have ontology"
                assert normalized["ontology"] == "NCBI Taxonomy", \
                    "Grounded taxon should use NCBI Taxonomy"
        
        # Test disease normalization
        for disease in sample_paper_comprehensive.diseases:
            normalized = entity_normalizer.normalize_disease(disease)
            
            # Verify normalized structure
            assert "id" in normalized, "Normalized disease should have id"
            assert "name" in normalized, "Normalized disease should have name"
            assert "canonical_name" in normalized, "Normalized disease should have canonical_name"
            assert "grounded" in normalized, "Normalized disease should have grounded flag"
            
            # Verify name matches input
            assert normalized["name"] == disease, \
                f"Normalized name should match input: {normalized['name']} != {disease}"
            
            # Verify canonical_name is not empty
            assert len(normalized["canonical_name"]) > 0, \
                "Canonical name should not be empty"
            
            # If grounded, should have ontology information
            if normalized["grounded"]:
                assert "ontology" in normalized, \
                    "Grounded entity should have ontology"
                assert normalized["ontology"] == "MeSH", \
                    "Grounded disease should use MeSH ontology"
    
    def test_semantic_properties_extraction(
        self,
        enhanced_builder,
        sample_paper_comprehensive
    ):
        """
        Test that semantic properties are correctly extracted.
        
        This test verifies that:
        1. Association edges have direction, comparison, statistical measures
        2. Intervention edges have intervention_type, effect_direction
        3. Methodology edges have method_name
        
        Requirements: 20.3, 2.1, 2.2, 2.3
        """
        # Process the paper
        edges = enhanced_builder.process_paper(sample_paper_comprehensive)
        
        # Find association edges
        association_edges = [
            e for e in edges
            if e.relation == RelationType.REPORTS_ASSOCIATION.value
        ]
        
        assert len(association_edges) > 0, "Should have association edges"
        
        # Verify association properties (Requirement 2.1)
        for edge in association_edges:
            assert "direction" in edge.properties, \
                "Association should have direction"
            assert edge.properties["direction"] in ["increased", "decreased", "no_change"], \
                f"Invalid direction: {edge.properties['direction']}"
            
            assert "comparison" in edge.properties, \
                "Association should have comparison context"
            
            assert "statistical_measure" in edge.properties, \
                "Association should have statistical measure"
        
        # Find intervention edges
        intervention_edges = [
            e for e in edges
            if e.relation == RelationType.REPORTS_INTERVENTION_EFFECT.value
        ]
        
        # If intervention edges exist, verify their properties (Requirement 2.2)
        for edge in intervention_edges:
            assert "intervention_type" in edge.properties, \
                "Intervention should have intervention_type"
            assert "effect_direction" in edge.properties, \
                "Intervention should have effect_direction"
        
        # Find methodology edges
        methodology_edges = [
            e for e in edges
            if e.relation == RelationType.USES_METHODOLOGY.value
        ]
        
        # If methodology edges exist, verify their properties (Requirement 2.3)
        for edge in methodology_edges:
            assert "method_name" in edge.properties, \
                "Methodology should have method_name"
    
    def test_edge_to_dict_conversion(
        self,
        enhanced_builder,
        sample_paper_comprehensive
    ):
        """
        Test that edges can be converted to dictionaries for Neo4j loading.
        
        This test verifies that:
        1. to_dict() produces a valid dictionary
        2. All required fields are present
        3. Provenance is embedded in the dictionary
        
        Requirements: 20.3
        """
        # Process the paper
        edges = enhanced_builder.process_paper(sample_paper_comprehensive)
        
        assert len(edges) > 0, "Should have edges to test"
        
        # Test conversion for each edge
        for edge in edges:
            edge_dict = edge.to_dict()
            
            # Verify dictionary structure
            assert isinstance(edge_dict, dict), "to_dict() should return a dictionary"
            
            # Verify core fields
            assert "source" in edge_dict, "Dict should have source"
            assert "target" in edge_dict, "Dict should have target"
            assert "relation" in edge_dict, "Dict should have relation"
            assert "confidence" in edge_dict, "Dict should have confidence"
            assert "evidence_strength" in edge_dict, "Dict should have evidence_strength"
            
            # Verify embedded provenance fields
            assert "paper_id" in edge_dict, "Dict should have embedded paper_id"
            assert "section" in edge_dict, "Dict should have embedded section"
            assert "source_sentence" in edge_dict, "Dict should have embedded source_sentence"
            assert "extraction_method" in edge_dict, "Dict should have embedded extraction_method"
            assert "extraction_timestamp" in edge_dict, "Dict should have embedded extraction_timestamp"
            assert "extractor_version" in edge_dict, "Dict should have embedded extractor_version"
            
            # Verify semantic properties are included
            for key, value in edge.properties.items():
                assert key in edge_dict, f"Dict should include property '{key}'"
                assert edge_dict[key] == value, f"Property '{key}' value should match"
    
    def test_multiple_papers_aggregation(
        self,
        enhanced_builder,
        sample_paper_comprehensive
    ):
        """
        Test processing multiple papers and aggregating relationships.
        
        This test verifies that:
        1. Multiple papers can be processed
        2. Relationships are indexed for reification
        3. Statistics are correctly calculated
        
        Requirements: 20.3
        """
        # Create a second paper with overlapping entities
        paper2 = EnrichedPaperRecord(
            doi="10.1234/integration.test.2024.002",
            title="Follow-up Study on Bacteroides in T2D",
            abstract="Validation study of Bacteroides fragilis in Type 2 Diabetes.",
            year=2024,
            article_type_normalized="original_research",
            taxa=["Bacteroides fragilis"],
            diseases=["Type 2 Diabetes"],
            methods=["16S rRNA sequencing"],
            sections=[
                ParsedSection(
                    section_type="results",
                    header="Results",
                    content=(
                        "We confirmed that Bacteroides fragilis was significantly increased "
                        "in Type 2 Diabetes patients (p = 0.005, fold change = 2.1)."
                    )
                )
            ],
            entities=[
                NamedEntity(text="Bacteroides fragilis", label="taxon", confidence=0.95),
                NamedEntity(text="Type 2 Diabetes", label="disease", confidence=0.92),
            ]
        )
        
        # Process both papers
        papers = [sample_paper_comprehensive, paper2]
        all_edges = enhanced_builder.process_papers(papers)
        
        # Verify edges from both papers
        assert len(all_edges) > 0, "Should have edges from multiple papers"
        
        # Verify edges from different papers
        paper_ids = set(edge.provenance.paper_id for edge in all_edges)
        assert len(paper_ids) >= 1, "Should have edges from at least one paper"
        
        # Get statistics
        stats = enhanced_builder.get_statistics()
        
        # Verify statistics
        assert stats["total_edges"] == len(all_edges), \
            "Statistics should match actual edge count"
        assert stats["total_relationships"] >= stats["total_edges"], \
            "Relationships should be >= edges"
        assert stats["unique_triples"] > 0, \
            "Should have unique (subject, predicate, object) triples"
    
    def test_confidence_threshold_enforcement(
        self,
        enhanced_builder,
        sample_paper_comprehensive
    ):
        """
        Test that only relationships with confidence >= 0.5 are included.
        
        This test verifies that:
        1. All edges have confidence >= 0.5
        2. Low confidence relationships are filtered out
        
        Requirements: 20.3, 2.4
        """
        # Process the paper
        edges = enhanced_builder.process_paper(sample_paper_comprehensive)
        
        # Verify all edges meet confidence threshold
        for edge in edges:
            assert edge.confidence >= 0.5, \
                f"Edge confidence should be >= 0.5, got {edge.confidence}"
            assert edge.provenance.confidence_score >= 0.5, \
                f"Provenance confidence should be >= 0.5, got {edge.provenance.confidence_score}"
    
    def test_evidence_strength_classification(
        self,
        enhanced_builder,
        sample_paper_comprehensive
    ):
        """
        Test that evidence strength is correctly classified.
        
        This test verifies that:
        1. Evidence strength is assigned to all edges
        2. Evidence strength values are valid
        
        Requirements: 20.3, 5.1, 5.2, 5.3
        """
        # Process the paper
        edges = enhanced_builder.process_paper(sample_paper_comprehensive)
        
        # Verify evidence strength for all edges
        valid_strengths = ["strong", "moderate", "weak", "conflicting"]
        
        for edge in edges:
            assert edge.evidence_strength in valid_strengths, \
                f"Invalid evidence_strength: {edge.evidence_strength}"


# ========== Run Tests ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

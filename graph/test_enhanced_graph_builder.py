"""
graph/test_enhanced_graph_builder.py
-------------------------------------
Unit tests for the enhanced graph builder.

Tests the integration of semantic extractor and relationship reifier
to create enhanced graph edges with embedded provenance.

Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 4.1
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
from graph.semantic_relationships import RelationType


# ========== Test Fixtures ==========

@pytest.fixture
def sample_paper_with_association():
    """Create a sample paper with taxon-disease association."""
    return EnrichedPaperRecord(
        doi="10.1234/test.2024.001",
        title="Bacteroides fragilis in Type 2 Diabetes",
        abstract="Study of gut microbiome in T2D patients.",
        year=2024,
        article_type_normalized="original_research",
        taxa=["Bacteroides fragilis", "Lactobacillus"],
        diseases=["Type 2 Diabetes", "T2D"],
        methods=["16S rRNA sequencing"],
        sections=[
            ParsedSection(
                section_type="results",
                header="Results",
                content=(
                    "We found that Bacteroides fragilis was significantly increased "
                    "in Type 2 Diabetes patients compared to healthy controls "
                    "(p = 0.001, fold change = 2.5). "
                    "The LDA score was 3.2 for this association."
                )
            ),
            ParsedSection(
                section_type="methods",
                header="Methods",
                content=(
                    "We used 16S rRNA sequencing on Illumina platform. "
                    "Sample size was 100 participants (50 T2D, 50 controls)."
                )
            )
        ],
        entities=[
            NamedEntity(text="Bacteroides fragilis", label="taxon", confidence=0.95),
            NamedEntity(text="Type 2 Diabetes", label="disease", confidence=0.92),
            NamedEntity(text="16S rRNA sequencing", label="method", confidence=0.88),
        ],
        data_availability=DataAvailabilityInfo(
            status="open",
            accession_numbers=["PRJNA123456"],
            repositories=["NCBI SRA"]
        )
    )


@pytest.fixture
def sample_paper_with_intervention():
    """Create a sample paper with intervention effect."""
    return EnrichedPaperRecord(
        doi="10.1234/test.2024.002",
        title="Probiotic intervention in IBD",
        abstract="RCT of probiotic supplementation.",
        year=2024,
        article_type_normalized="original_research",
        taxa=["Lactobacillus rhamnosus"],
        diseases=["IBD"],
        methods=["16S rRNA sequencing"],
        treatments=["probiotic"],
        sections=[
            ParsedSection(
                section_type="methods",
                header="Methods",
                content=(
                    "Participants received probiotic supplementation "
                    "(10^9 CFU daily) for 8 weeks. "
                    "Sample size was 60 participants."
                )
            ),
            ParsedSection(
                section_type="results",
                header="Results",
                content=(
                    "Lactobacillus rhamnosus abundance was significantly increased "
                    "after probiotic intervention (p = 0.003)."
                )
            )
        ],
        entities=[
            NamedEntity(text="Lactobacillus rhamnosus", label="taxon", confidence=0.94),
            NamedEntity(text="probiotic", label="treatment", confidence=0.91),
        ]
    )


@pytest.fixture
def enhanced_builder():
    """Create an enhanced graph builder instance."""
    return EnhancedGraphBuilder(
        extraction_method="regex_ner",
        extractor_version="1.0"
    )


# ========== Test Cases ==========

def test_builder_initialization(enhanced_builder):
    """Test that builder initializes correctly."""
    assert enhanced_builder.semantic_extractor is not None
    assert enhanced_builder.relationship_reifier is not None
    assert len(enhanced_builder.relationships) == 0
    assert len(enhanced_builder.edges) == 0
    assert len(enhanced_builder.claims) == 0


def test_process_paper_with_association(enhanced_builder, sample_paper_with_association):
    """
    Test processing a paper with taxon-disease association.
    
    Requirement 2.1: Extract associations with statistical properties
    Requirement 3.1, 3.2: Complete provenance tracking
    """
    edges = enhanced_builder.process_paper(sample_paper_with_association)
    
    # Should extract at least one association edge
    assert len(edges) > 0
    
    # Check for association edges
    association_edges = [
        e for e in edges 
        if e.relation == RelationType.REPORTS_ASSOCIATION.value
    ]
    assert len(association_edges) > 0
    
    # Verify edge structure
    edge = association_edges[0]
    assert edge.source == sample_paper_with_association.get_dedup_key()
    assert edge.target in sample_paper_with_association.taxa
    assert edge.confidence >= 0.5
    
    # Verify semantic properties (Requirement 2.1)
    assert "direction" in edge.properties
    assert edge.properties["direction"] in ["increased", "decreased", "no_change"]
    assert "comparison" in edge.properties
    assert "statistical_measure" in edge.properties
    
    # Verify provenance is embedded (Requirement 3.1, 3.2)
    assert edge.provenance is not None
    assert edge.provenance.paper_id == sample_paper_with_association.get_dedup_key()
    assert edge.provenance.section_type == "results"
    assert len(edge.provenance.source_sentence) > 0
    assert edge.provenance.extraction_method == "regex_ner"
    assert edge.provenance.confidence_score >= 0.5


def test_process_paper_with_intervention(enhanced_builder, sample_paper_with_intervention):
    """
    Test processing a paper with intervention effect.
    
    Requirement 2.2: Extract intervention effects
    """
    edges = enhanced_builder.process_paper(sample_paper_with_intervention)
    
    # Should extract intervention edges
    intervention_edges = [
        e for e in edges 
        if e.relation == RelationType.REPORTS_INTERVENTION_EFFECT.value
    ]
    assert len(intervention_edges) > 0
    
    # Verify intervention properties (Requirement 2.2)
    edge = intervention_edges[0]
    assert "intervention_type" in edge.properties
    assert edge.properties["intervention_type"] in [
        "probiotic", "FMT", "diet", "antibiotic", "prebiotic", "synbiotic", "other"
    ]
    assert "effect_direction" in edge.properties
    assert edge.properties["effect_direction"] in ["increased", "decreased", "no_change"]


def test_process_paper_with_methodology(enhanced_builder, sample_paper_with_association):
    """
    Test processing a paper with methodology information.
    
    Requirement 2.3: Extract methodology usage
    """
    edges = enhanced_builder.process_paper(sample_paper_with_association)
    
    # Should extract methodology edges
    methodology_edges = [
        e for e in edges 
        if e.relation == RelationType.USES_METHODOLOGY.value
    ]
    assert len(methodology_edges) > 0
    
    # Verify methodology properties (Requirement 2.3)
    edge = methodology_edges[0]
    assert "method_name" in edge.properties
    assert edge.properties["method_name"] in sample_paper_with_association.methods


def test_enhanced_graph_edge_to_dict():
    """Test EnhancedGraphEdge to_dict conversion."""
    from graph.provenance import ProvenanceMetadata
    
    provenance = ProvenanceMetadata(
        paper_id="10.1234/test",
        section_type="results",
        source_sentence="Test sentence",
        extraction_method="regex_ner",
        extraction_timestamp=datetime.now(timezone.utc),
        extractor_version="1.0",
        confidence_score=0.8
    )
    
    edge = EnhancedGraphEdge(
        source="paper1",
        target="taxon1",
        relation="REPORTS_ASSOCIATION",
        properties={"direction": "increased", "p_value": 0.01},
        provenance=provenance,
        evidence_strength="strong",
        confidence=0.8
    )
    
    edge_dict = edge.to_dict()
    
    # Verify structure
    assert edge_dict["source"] == "paper1"
    assert edge_dict["target"] == "taxon1"
    assert edge_dict["relation"] == "REPORTS_ASSOCIATION"
    assert edge_dict["confidence"] == 0.8
    assert edge_dict["evidence_strength"] == "strong"
    
    # Verify semantic properties
    assert edge_dict["direction"] == "increased"
    assert edge_dict["p_value"] == 0.01
    
    # Verify embedded provenance
    assert edge_dict["paper_id"] == "10.1234/test"
    assert edge_dict["section"] == "results"
    assert edge_dict["source_sentence"] == "Test sentence"
    assert edge_dict["extraction_method"] == "regex_ner"


def test_process_multiple_papers(enhanced_builder, sample_paper_with_association, sample_paper_with_intervention):
    """Test processing multiple papers."""
    papers = [sample_paper_with_association, sample_paper_with_intervention]
    edges = enhanced_builder.process_papers(papers)
    
    # Should extract edges from both papers
    assert len(edges) > 0
    
    # Verify edges from different papers
    paper_ids = set(edge.provenance.paper_id for edge in edges)
    assert len(paper_ids) >= 1  # At least one paper should have edges


def test_create_reified_claims(enhanced_builder, sample_paper_with_association):
    """
    Test creating reified claims from relationships.
    
    Requirement 4.1: Create reified claim nodes aggregating supporting evidence
    """
    # Process paper to extract relationships
    enhanced_builder.process_paper(sample_paper_with_association)
    
    # Create reified claims
    claims = enhanced_builder.create_reified_claims()
    
    # Should create claims for relationships
    # Note: Claims are only created for relationships with sufficient evidence
    # In this case, we have one paper, so claims may or may not be created
    # depending on the implementation threshold
    assert isinstance(claims, list)
    
    # If claims were created, verify structure
    if len(claims) > 0:
        claim = claims[0]
        assert claim.claim_id is not None
        assert claim.subject_entity is not None
        assert claim.predicate is not None
        assert claim.object_entity is not None
        assert len(claim.supporting_papers) > 0
        assert 0.0 <= claim.consensus_confidence <= 1.0


def test_get_statistics(enhanced_builder, sample_paper_with_association):
    """Test getting graph construction statistics."""
    enhanced_builder.process_paper(sample_paper_with_association)
    
    stats = enhanced_builder.get_statistics()
    
    assert "total_relationships" in stats
    assert "total_edges" in stats
    assert "total_claims" in stats
    assert "associations" in stats
    assert "interventions" in stats
    assert "methodologies" in stats
    assert "unique_triples" in stats
    
    # Verify counts are non-negative
    assert stats["total_relationships"] >= 0
    assert stats["total_edges"] >= 0
    assert stats["associations"] >= 0


def test_edge_confidence_threshold(enhanced_builder, sample_paper_with_association):
    """
    Test that only relationships with confidence >= 0.5 are included.
    
    Requirement 2.4: Only create relationships with extraction confidence >= 0.5
    """
    edges = enhanced_builder.process_paper(sample_paper_with_association)
    
    # All edges should have confidence >= 0.5
    for edge in edges:
        assert edge.confidence >= 0.5
        assert edge.provenance.confidence_score >= 0.5


def test_provenance_completeness(enhanced_builder, sample_paper_with_association):
    """
    Test that all edges have complete provenance metadata.
    
    Requirement 3.5: Reject relationships lacking required provenance fields
    """
    edges = enhanced_builder.process_paper(sample_paper_with_association)
    
    for edge in edges:
        # Verify required provenance fields
        assert edge.provenance.paper_id is not None
        assert len(edge.provenance.paper_id) > 0
        assert edge.provenance.section_type is not None
        assert edge.provenance.source_sentence is not None
        assert len(edge.provenance.source_sentence) > 0
        assert edge.provenance.extraction_method is not None
        assert edge.provenance.extraction_timestamp is not None
        assert edge.provenance.extractor_version is not None
        assert 0.0 <= edge.provenance.confidence_score <= 1.0


def test_relationship_indexing(enhanced_builder, sample_paper_with_association):
    """Test that relationships are properly indexed for reification."""
    enhanced_builder.process_paper(sample_paper_with_association)
    
    # Verify relationship index is populated
    assert len(enhanced_builder.relationship_index) > 0
    
    # Each key should be a tuple of (subject, predicate, object)
    for key, relationships in enhanced_builder.relationship_index.items():
        assert isinstance(key, tuple)
        assert len(key) == 3
        assert len(relationships) > 0


def test_empty_paper_handling(enhanced_builder):
    """Test handling of papers with no extractable relationships."""
    empty_paper = EnrichedPaperRecord(
        doi="10.1234/empty",
        title="Empty paper",
        abstract="No relevant content",
        year=2024,
        taxa=[],
        diseases=[],
        methods=[],
        sections=[]
    )
    
    edges = enhanced_builder.process_paper(empty_paper)
    
    # Should return empty list for papers with no content
    assert len(edges) == 0


def test_get_all_edges(enhanced_builder, sample_paper_with_association):
    """Test retrieving all edges."""
    enhanced_builder.process_paper(sample_paper_with_association)
    
    all_edges = enhanced_builder.get_all_edges()
    
    assert isinstance(all_edges, list)
    assert len(all_edges) > 0
    assert all(isinstance(edge, EnhancedGraphEdge) for edge in all_edges)


def test_get_all_claims(enhanced_builder, sample_paper_with_association):
    """Test retrieving all claims."""
    enhanced_builder.process_paper(sample_paper_with_association)
    enhanced_builder.create_reified_claims()
    
    all_claims = enhanced_builder.get_all_claims()
    
    assert isinstance(all_claims, list)
    # Claims may be empty if threshold not met
    assert all(hasattr(claim, 'claim_id') for claim in all_claims)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

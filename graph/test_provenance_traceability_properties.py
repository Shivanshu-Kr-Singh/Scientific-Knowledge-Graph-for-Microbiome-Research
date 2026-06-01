"""
graph/test_provenance_traceability_properties.py
-------------------------------------------------
Property-based tests for provenance traceability across graph edges.

Tests that every edge in the knowledge graph has complete and valid provenance
metadata that traces back to source text.

**Validates: Requirements 3.1, 3.2, 20.5**
"""

import pytest
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings, assume

from graph.enhanced_graph_builder import EnhancedGraphBuilder, EnhancedGraphEdge
from graph.semantic_extractor import SemanticRelationshipExtractor
from graph.provenance import ProvenanceMetadata, REGISTERED_EXTRACTION_METHODS
from graph.extractor_registry import get_registered_method_ids
from nlp.enriched_record import EnrichedPaperRecord, ParsedSection


# ============================================================================
# Hypothesis Strategies for Generating Test Data
# ============================================================================

# Valid section types as per requirements
VALID_SECTION_TYPES = {"abstract", "methods", "results", "discussion", "introduction", "other"}

# Strategy for valid section types
section_type_strategy = st.sampled_from(list(VALID_SECTION_TYPES))

# Strategy for valid extraction methods (from registry)
extraction_method_strategy = st.sampled_from(list(get_registered_method_ids()))

# Strategy for confidence scores in valid range [0.5, 1.0]
# Note: Only relationships with confidence >= 0.5 are created (Requirement 2.4)
confidence_strategy = st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for non-empty strings
non_empty_string_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

# Strategy for paper IDs
paper_id_strategy = st.one_of(
    st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
    st.from_regex(r"10\.\d{4,}/[a-zA-Z0-9\.\-]+", fullmatch=True),  # DOI format
)

# Strategy for entity names
entity_name_strategy = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())

# Strategy for timestamps
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime.now(),
    timezones=st.just(timezone.utc)
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_paper_with_sections(
    paper_id: str,
    sections: list[ParsedSection]
) -> EnrichedPaperRecord:
    """Helper to create a test paper with specific sections."""
    return EnrichedPaperRecord(
        doi=paper_id if paper_id.startswith("10.") else None,
        pmid=paper_id if not paper_id.startswith("10.") else None,
        title="Test Paper",
        abstract="Test abstract with microbiome content.",
        year=2024,
        authors=[],
        sections=sections,
        entities=[],
        article_type_normalized="original_research",
    )


def create_test_section(section_type: str, content: str) -> ParsedSection:
    """Helper to create a test section."""
    return ParsedSection(
        section_type=section_type,
        header=section_type.capitalize(),
        content=content,
    )


def create_enhanced_graph_edge(
    source: str,
    target: str,
    relation: str,
    section_type: str,
    source_sentence: str,
    extraction_method: str,
    confidence: float,
    paper_id: str,
) -> EnhancedGraphEdge:
    """Helper to create an EnhancedGraphEdge with provenance."""
    provenance = ProvenanceMetadata(
        paper_id=paper_id,
        section_type=section_type,
        source_sentence=source_sentence,
        extraction_method=extraction_method,
        extraction_timestamp=datetime.now(timezone.utc),
        extractor_version="1.0",
        confidence_score=confidence,
    )
    
    return EnhancedGraphEdge(
        source=source,
        target=target,
        relation=relation,
        properties={},
        provenance=provenance,
        evidence_strength="moderate",
        confidence=confidence,
    )


# ============================================================================
# Property 4: Provenance Traceability
# **Validates: Requirements 3.1, 3.2, 20.5**
# ============================================================================

@given(
    source_entity=entity_name_strategy,
    target_entity=entity_name_strategy,
    relation=st.sampled_from(["REPORTS_ASSOCIATION", "REPORTS_INTERVENTION_EFFECT", "USES_METHODOLOGY"]),
    section_type=section_type_strategy,
    source_sentence=non_empty_string_strategy,
    extraction_method=extraction_method_strategy,
    confidence=confidence_strategy,
    paper_id=paper_id_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_provenance_traceability(
    source_entity,
    target_entity,
    relation,
    section_type,
    source_sentence,
    extraction_method,
    confidence,
    paper_id,
):
    """
    **Property 4: Provenance Traceability**
    **Validates: Requirements 3.1, 3.2, 20.5**
    
    Test that every edge traces back to valid provenance metadata:
    - Every edge has a valid section type (abstract, methods, results, discussion, etc.)
    - Every edge has a non-empty source_sentence
    - Every edge's extraction_method exists in registered extractors
    
    Universal Property:
    - For all valid edges, provenance metadata is complete and traceable
    - section_type is always in the valid set
    - source_sentence is always non-empty
    - extraction_method is always registered
    """
    # Create an enhanced graph edge with provenance
    edge = create_enhanced_graph_edge(
        source=source_entity,
        target=target_entity,
        relation=relation,
        section_type=section_type,
        source_sentence=source_sentence,
        extraction_method=extraction_method,
        confidence=confidence,
        paper_id=paper_id,
    )
    
    # Property 4a: Every edge has a valid section type
    # Requirement 3.1: Provenance SHALL capture section type for every relationship
    assert edge.provenance.section_type is not None
    assert edge.provenance.section_type in VALID_SECTION_TYPES, (
        f"section_type '{edge.provenance.section_type}' is not in valid set: {VALID_SECTION_TYPES}"
    )
    
    # Property 4b: Every edge has a non-empty source_sentence
    # Requirement 3.1: Provenance SHALL capture source sentence for every relationship
    assert edge.provenance.source_sentence is not None
    assert edge.provenance.source_sentence.strip() != "", (
        "source_sentence must be non-empty"
    )
    
    # Property 4c: Every edge's extraction_method exists in registered extractors
    # Requirement 10.2: System SHALL validate that extraction_method exists in registered extractors
    registered_methods = get_registered_method_ids()
    assert edge.provenance.extraction_method in registered_methods, (
        f"extraction_method '{edge.provenance.extraction_method}' is not registered. "
        f"Registered methods: {registered_methods}"
    )
    
    # Property 4d: Edge can be converted to dict with all provenance fields
    edge_dict = edge.to_dict()
    assert "section" in edge_dict
    assert "source_sentence" in edge_dict
    assert "extraction_method" in edge_dict
    assert edge_dict["section"] == section_type
    # Note: ProvenanceEncoder strips the sentence, so we compare with stripped version
    assert edge_dict["source_sentence"].strip() == source_sentence.strip()
    assert edge_dict["extraction_method"] == extraction_method


@given(
    invalid_section=st.text(min_size=1, max_size=50).filter(
        lambda s: s not in VALID_SECTION_TYPES and s.strip()
    )
)
@settings(max_examples=100, deadline=None)
def test_property_invalid_section_type_rejected(invalid_section):
    """
    **Property 4: Provenance Traceability (Negative Test)**
    **Validates: Requirements 3.1, 3.2, 20.5**
    
    Test that edges with invalid section types are rejected.
    
    Universal Property:
    - For all section types not in the valid set, ProvenanceMetadata creation fails
    """
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError) as exc_info:
        ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type=invalid_section,
            source_sentence="Test sentence with content.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.85,
        )
    
    # Verify the error is about section_type
    assert "section_type" in str(exc_info.value)


@given(
    empty_sentence=st.one_of(
        st.just(""),
        st.text(max_size=20).filter(lambda s: not s.strip())
    )
)
@settings(max_examples=100, deadline=None)
def test_property_empty_source_sentence_rejected(empty_sentence):
    """
    **Property 4: Provenance Traceability (Negative Test)**
    **Validates: Requirements 3.1, 3.2, 20.5**
    
    Test that edges with empty source_sentence are rejected.
    
    Universal Property:
    - For all empty or whitespace-only sentences, edge creation fails
    """
    with pytest.raises(ValueError, match="sentence must be a non-empty string"):
        from graph.provenance import ProvenanceEncoder
        
        encoder = ProvenanceEncoder()
        paper = create_test_paper_with_sections(
            "10.1234/test",
            [create_test_section("results", "Some content")]
        )
        section = create_test_section("results", "Some content")
        
        encoder.encode(
            paper=paper,
            section=section,
            sentence=empty_sentence,
            extraction_method="regex_ner",
            confidence=0.85,
        )


@given(
    unregistered_method=st.text(min_size=1, max_size=50).filter(
        lambda s: s not in get_registered_method_ids() and s.strip()
    )
)
@settings(max_examples=100, deadline=None)
def test_property_unregistered_extraction_method_rejected(unregistered_method):
    """
    **Property 4: Provenance Traceability (Negative Test)**
    **Validates: Requirements 3.1, 3.2, 10.2, 20.5**
    
    Test that edges with unregistered extraction methods are rejected.
    
    Universal Property:
    - For all extraction methods not in the registry, edge creation fails
    """
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError) as exc_info:
        ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence with content.",
            extraction_method=unregistered_method,
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.85,
        )
    
    # Verify the error is about extraction_method
    error_msg = str(exc_info.value)
    assert "extraction_method" in error_msg
    assert "not registered" in error_msg


@given(
    num_edges=st.integers(min_value=1, max_value=20),
    section_type=section_type_strategy,
    extraction_method=extraction_method_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_all_edges_have_traceable_provenance(
    num_edges,
    section_type,
    extraction_method,
):
    """
    **Property 4: Provenance Traceability (Batch Test)**
    **Validates: Requirements 3.1, 3.2, 20.5**
    
    Test that all edges in a batch have complete and traceable provenance.
    
    Universal Property:
    - For all edges in a batch, every edge has valid provenance
    - All section types are valid
    - All source sentences are non-empty
    - All extraction methods are registered
    """
    edges = []
    
    # Create multiple edges
    for i in range(num_edges):
        edge = create_enhanced_graph_edge(
            source=f"Entity_{i}",
            target=f"Target_{i}",
            relation="REPORTS_ASSOCIATION",
            section_type=section_type,
            source_sentence=f"Test sentence {i} with meaningful content.",
            extraction_method=extraction_method,
            confidence=0.75,
            paper_id=f"10.1234/paper_{i}",
        )
        edges.append(edge)
    
    # Property: All edges have valid provenance
    for edge in edges:
        # Valid section type
        assert edge.provenance.section_type in VALID_SECTION_TYPES
        
        # Non-empty source sentence
        assert edge.provenance.source_sentence is not None
        assert edge.provenance.source_sentence.strip() != ""
        
        # Registered extraction method
        assert edge.provenance.extraction_method in get_registered_method_ids()
        
        # Paper ID is present
        assert edge.provenance.paper_id is not None
        assert edge.provenance.paper_id.strip() != ""


@given(
    paper_id=paper_id_strategy,
    section_type=section_type_strategy,
    extraction_method=extraction_method_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_edge_dict_preserves_provenance(
    paper_id,
    section_type,
    extraction_method,
):
    """
    **Property 4: Provenance Traceability (Serialization Test)**
    **Validates: Requirements 3.1, 3.2, 20.5**
    
    Test that edge serialization to dict preserves all provenance fields.
    
    Universal Property:
    - For all edges, to_dict() preserves provenance metadata
    - All required provenance fields are present in the dict
    - Field values match the original provenance
    """
    edge = create_enhanced_graph_edge(
        source="Bacteroides fragilis",
        target="Type 2 Diabetes",
        relation="REPORTS_ASSOCIATION",
        section_type=section_type,
        source_sentence="Bacteroides fragilis was significantly increased in T2D patients.",
        extraction_method=extraction_method,
        confidence=0.85,
        paper_id=paper_id,
    )
    
    # Convert to dict
    edge_dict = edge.to_dict()
    
    # Property: All provenance fields are preserved
    assert "paper_id" in edge_dict
    assert "section" in edge_dict
    assert "source_sentence" in edge_dict
    assert "extraction_method" in edge_dict
    assert "extraction_timestamp" in edge_dict
    assert "extractor_version" in edge_dict
    
    # Property: Field values match original provenance
    assert edge_dict["paper_id"] == paper_id
    assert edge_dict["section"] == section_type
    assert edge_dict["source_sentence"] == "Bacteroides fragilis was significantly increased in T2D patients."
    assert edge_dict["extraction_method"] == extraction_method
    assert edge_dict["extractor_version"] == "1.0"
    
    # Property: Section type is valid
    assert edge_dict["section"] in VALID_SECTION_TYPES
    
    # Property: Extraction method is registered
    assert edge_dict["extraction_method"] in get_registered_method_ids()


@given(
    section_types=st.lists(
        section_type_strategy,
        min_size=1,
        max_size=5,
        unique=True
    ),
    extraction_methods=st.lists(
        extraction_method_strategy,
        min_size=1,
        max_size=3,
        unique=True
    ),
)
@settings(max_examples=100, deadline=None)
def test_property_multiple_sections_and_methods_all_valid(
    section_types,
    extraction_methods,
):
    """
    **Property 4: Provenance Traceability (Multi-Source Test)**
    **Validates: Requirements 3.1, 3.2, 20.5**
    
    Test that edges from multiple sections and extraction methods all have valid provenance.
    
    Universal Property:
    - For all combinations of section types and extraction methods,
      all created edges have valid and traceable provenance
    """
    edges = []
    
    # Create edges for each combination
    for i, section_type in enumerate(section_types):
        for j, extraction_method in enumerate(extraction_methods):
            edge = create_enhanced_graph_edge(
                source=f"Taxon_{i}_{j}",
                target=f"Disease_{i}_{j}",
                relation="REPORTS_ASSOCIATION",
                section_type=section_type,
                source_sentence=f"Sentence from {section_type} section using {extraction_method}.",
                extraction_method=extraction_method,
                confidence=0.80,
                paper_id=f"10.1234/paper_{i}_{j}",
            )
            edges.append(edge)
    
    # Property: All edges have valid provenance
    for edge in edges:
        # Section type is valid
        assert edge.provenance.section_type in VALID_SECTION_TYPES
        
        # Source sentence is non-empty
        assert edge.provenance.source_sentence.strip() != ""
        
        # Extraction method is registered
        assert edge.provenance.extraction_method in get_registered_method_ids()
        
        # Provenance can be serialized
        edge_dict = edge.to_dict()
        assert "section" in edge_dict
        assert "source_sentence" in edge_dict
        assert "extraction_method" in edge_dict

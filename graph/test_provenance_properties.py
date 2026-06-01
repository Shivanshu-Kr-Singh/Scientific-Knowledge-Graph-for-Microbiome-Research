"""
graph/test_provenance_properties.py
------------------------------------
Property-based tests for provenance tracking module.

Tests universal properties that should hold for all ProvenanceMetadata instances.

**Validates: Requirements 3.5**
"""

import pytest
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings
from pydantic import ValidationError

from graph.provenance import (
    ProvenanceMetadata,
    ProvenanceEncoder,
    REGISTERED_EXTRACTION_METHODS,
)
from nlp.enriched_record import EnrichedPaperRecord, ParsedSection


# ============================================================================
# Hypothesis Strategies for Generating Test Data
# ============================================================================

# Strategy for valid section types
section_type_strategy = st.sampled_from([
    "abstract", "methods", "results", "discussion", "introduction", "other"
])

# Strategy for valid extraction methods
extraction_method_strategy = st.sampled_from(list(REGISTERED_EXTRACTION_METHODS))

# Strategy for valid validation statuses
validation_status_strategy = st.sampled_from([
    "unvalidated", "human_verified", "cross_validated"
])

# Strategy for confidence scores in valid range [0.0, 1.0]
confidence_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for non-empty strings (for paper_id, source_sentence, etc.)
non_empty_string_strategy = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

# Strategy for optional strings
optional_string_strategy = st.one_of(st.none(), st.text(min_size=1, max_size=100))

# Strategy for optional integers (for sentence_offset)
optional_int_strategy = st.one_of(st.none(), st.integers(min_value=0, max_value=100000))

# Strategy for timestamps (past and present, not future)
# Note: datetimes() requires naive datetimes for min/max, then adds timezone
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime.now(),
    timezones=st.just(timezone.utc)
)

# Strategy for extractor versions
extractor_version_strategy = st.text(min_size=1, max_size=20).filter(lambda s: s.strip())


# ============================================================================
# Property 1: Provenance Completeness
# **Validates: Requirements 3.5**
# ============================================================================

@given(
    paper_id=non_empty_string_strategy,
    section_type=section_type_strategy,
    source_sentence=non_empty_string_strategy,
    extraction_method=extraction_method_strategy,
    extraction_timestamp=timestamp_strategy,
    extractor_version=extractor_version_strategy,
    confidence_score=confidence_strategy,
    validation_status=validation_status_strategy,
    sentence_offset=optional_int_strategy,
    llm_prompt_hash=optional_string_strategy,
    validator_id=optional_string_strategy,
    surrounding_context=optional_string_strategy,
    figure_table_ref=optional_string_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_provenance_completeness(
    paper_id,
    section_type,
    source_sentence,
    extraction_method,
    extraction_timestamp,
    extractor_version,
    confidence_score,
    validation_status,
    sentence_offset,
    llm_prompt_hash,
    validator_id,
    surrounding_context,
    figure_table_ref,
):
    """
    **Property 1: Provenance Completeness**
    **Validates: Requirements 3.5**
    
    Test that all created ProvenanceMetadata instances have required fields
    and that confidence scores are always in range [0.0, 1.0].
    
    Universal Property:
    - For all valid inputs, ProvenanceMetadata creation succeeds
    - All required fields are present and non-empty
    - confidence_score is always in range [0.0, 1.0]
    - extraction_method is always a registered method
    - section_type is always a valid section type
    - validation_status is always a valid status
    """
    # Create ProvenanceMetadata with generated values
    provenance = ProvenanceMetadata(
        paper_id=paper_id,
        section_type=section_type,
        source_sentence=source_sentence,
        extraction_method=extraction_method,
        extraction_timestamp=extraction_timestamp,
        extractor_version=extractor_version,
        confidence_score=confidence_score,
        validation_status=validation_status,
        sentence_offset=sentence_offset,
        llm_prompt_hash=llm_prompt_hash,
        validator_id=validator_id,
        surrounding_context=surrounding_context,
        figure_table_ref=figure_table_ref,
    )
    
    # Property 1a: All required fields are present and non-empty
    assert provenance.paper_id is not None
    assert provenance.paper_id.strip() != ""
    assert provenance.section_type is not None
    assert provenance.section_type.strip() != ""
    assert provenance.source_sentence is not None
    assert provenance.source_sentence.strip() != ""
    assert provenance.extraction_method is not None
    assert provenance.extraction_method.strip() != ""
    assert provenance.extraction_timestamp is not None
    assert provenance.extractor_version is not None
    assert provenance.extractor_version.strip() != ""
    
    # Property 1b: Confidence score is always in range [0.0, 1.0]
    assert 0.0 <= provenance.confidence_score <= 1.0
    
    # Property 1c: extraction_method is always a registered method
    assert provenance.extraction_method in REGISTERED_EXTRACTION_METHODS
    
    # Property 1d: section_type is always a valid section type
    valid_sections = {"abstract", "methods", "results", "discussion", "introduction", "other"}
    assert provenance.section_type in valid_sections
    
    # Property 1e: validation_status is always a valid status
    valid_statuses = {"unvalidated", "human_verified", "cross_validated"}
    assert provenance.validation_status in valid_statuses
    
    # Property 1f: timestamp has timezone information (UTC)
    assert provenance.extraction_timestamp.tzinfo is not None


@given(
    confidence_score=st.floats(allow_nan=False, allow_infinity=False).filter(
        lambda x: x < 0.0 or x > 1.0
    )
)
@settings(max_examples=100, deadline=None)
def test_property_confidence_score_validation(confidence_score):
    """
    **Property 1: Provenance Completeness (Negative Test)**
    **Validates: Requirements 3.5**
    
    Test that confidence scores outside [0.0, 1.0] are always rejected.
    
    Universal Property:
    - For all confidence scores < 0.0 or > 1.0, ProvenanceMetadata creation fails
    """
    with pytest.raises(ValidationError) as exc_info:
        ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=confidence_score,
        )
    
    # Verify the error is about confidence_score
    assert "confidence_score" in str(exc_info.value)


@given(
    invalid_method=st.text(min_size=1, max_size=50).filter(
        lambda s: s not in REGISTERED_EXTRACTION_METHODS
    )
)
@settings(max_examples=100, deadline=None)
def test_property_extraction_method_validation(invalid_method):
    """
    **Property 1: Provenance Completeness (Negative Test)**
    **Validates: Requirements 3.5, 10.2**
    
    Test that unregistered extraction methods are always rejected.
    
    Universal Property:
    - For all extraction methods not in REGISTERED_EXTRACTION_METHODS,
      ProvenanceMetadata creation fails
    """
    with pytest.raises(ValidationError) as exc_info:
        ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method=invalid_method,
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.85,
        )
    
    # Verify the error is about extraction_method
    error_msg = str(exc_info.value)
    assert "extraction_method" in error_msg
    assert "not registered" in error_msg


@given(
    invalid_section=st.text(min_size=1, max_size=50).filter(
        lambda s: s not in {"abstract", "methods", "results", "discussion", "introduction", "other"}
    )
)
@settings(max_examples=100, deadline=None)
def test_property_section_type_validation(invalid_section):
    """
    **Property 1: Provenance Completeness (Negative Test)**
    **Validates: Requirements 3.5**
    
    Test that invalid section types are always rejected.
    
    Universal Property:
    - For all section types not in the valid set, ProvenanceMetadata creation fails
    """
    with pytest.raises(ValidationError) as exc_info:
        ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type=invalid_section,
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.85,
        )
    
    # Verify the error is about section_type
    assert "section_type" in str(exc_info.value)


# ============================================================================
# Property Tests for ProvenanceEncoder
# ============================================================================

def create_test_paper_with_id(paper_id: str) -> EnrichedPaperRecord:
    """Helper to create a test paper with specific ID."""
    return EnrichedPaperRecord(
        doi=paper_id if paper_id.startswith("10.") else None,
        pmid=paper_id if not paper_id.startswith("10.") else None,
        title="Test Paper",
        abstract="Test abstract",
        year=2024,
        authors=[],
    )


def create_test_section_with_type(section_type: str) -> ParsedSection:
    """Helper to create a test section with specific type."""
    return ParsedSection(
        section_type=section_type,
        header=section_type.capitalize(),
        content="Test content with a sentence. Another sentence for context.",
    )


@given(
    paper_id=non_empty_string_strategy,
    section_type=section_type_strategy,
    sentence=non_empty_string_strategy,
    extraction_method=extraction_method_strategy,
    confidence=confidence_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_encoder_creates_valid_provenance(
    paper_id, section_type, sentence, extraction_method, confidence
):
    """
    **Property 1: Provenance Completeness (Encoder)**
    **Validates: Requirements 3.5**
    
    Test that ProvenanceEncoder always creates valid ProvenanceMetadata
    for all valid inputs.
    
    Universal Property:
    - For all valid inputs, encoder.encode() succeeds
    - The returned ProvenanceMetadata passes validation
    - All required fields are populated
    - confidence_score matches input confidence
    """
    encoder = ProvenanceEncoder()
    paper = create_test_paper_with_id(paper_id)
    section = create_test_section_with_type(section_type)
    
    # Encode provenance
    provenance = encoder.encode(
        paper=paper,
        section=section,
        sentence=sentence,
        extraction_method=extraction_method,
        confidence=confidence,
    )
    
    # Property: Encoder creates valid provenance
    assert encoder.validate_provenance(provenance) is True
    
    # Property: All required fields are populated
    assert provenance.paper_id is not None
    assert provenance.section_type == section_type
    assert provenance.source_sentence == sentence.strip()
    assert provenance.extraction_method == extraction_method
    assert provenance.confidence_score == confidence
    
    # Property: Confidence score is in valid range
    assert 0.0 <= provenance.confidence_score <= 1.0
    
    # Property: Timestamp is set and not in the future
    assert provenance.extraction_timestamp is not None
    assert provenance.extraction_timestamp <= datetime.now(timezone.utc)


@given(
    confidence=st.floats(allow_nan=False, allow_infinity=False).filter(
        lambda x: x < 0.0 or x > 1.0
    )
)
@settings(max_examples=100, deadline=None)
def test_property_encoder_rejects_invalid_confidence(confidence):
    """
    **Property 1: Provenance Completeness (Encoder Negative Test)**
    **Validates: Requirements 3.5**
    
    Test that ProvenanceEncoder always rejects invalid confidence scores.
    
    Universal Property:
    - For all confidence scores outside [0.0, 1.0], encoder.encode() raises ValueError
    """
    encoder = ProvenanceEncoder()
    paper = create_test_paper_with_id("10.1234/test")
    section = create_test_section_with_type("results")
    
    with pytest.raises(ValueError, match="confidence must be in range"):
        encoder.encode(
            paper=paper,
            section=section,
            sentence="Test sentence.",
            extraction_method="regex_ner",
            confidence=confidence,
        )


@given(
    invalid_method=st.text(min_size=1, max_size=50).filter(
        lambda s: s not in REGISTERED_EXTRACTION_METHODS
    )
)
@settings(max_examples=100, deadline=None)
def test_property_encoder_rejects_unregistered_method(invalid_method):
    """
    **Property 1: Provenance Completeness (Encoder Negative Test)**
    **Validates: Requirements 3.5, 10.2**
    
    Test that ProvenanceEncoder always rejects unregistered extraction methods.
    
    Universal Property:
    - For all extraction methods not in REGISTERED_EXTRACTION_METHODS,
      encoder.encode() raises ValueError
    """
    encoder = ProvenanceEncoder()
    paper = create_test_paper_with_id("10.1234/test")
    section = create_test_section_with_type("results")
    
    with pytest.raises(ValueError, match="not registered"):
        encoder.encode(
            paper=paper,
            section=section,
            sentence="Test sentence.",
            extraction_method=invalid_method,
            confidence=0.85,
        )


@given(
    empty_sentence=st.one_of(
        st.just(""),
        st.text(max_size=20).filter(lambda s: not s.strip())
    )
)
@settings(max_examples=100, deadline=None)
def test_property_encoder_rejects_empty_sentence(empty_sentence):
    """
    **Property 1: Provenance Completeness (Encoder Negative Test)**
    **Validates: Requirements 3.5**
    
    Test that ProvenanceEncoder always rejects empty or whitespace-only sentences.
    
    Universal Property:
    - For all empty or whitespace-only sentences, encoder.encode() raises ValueError
    """
    encoder = ProvenanceEncoder()
    paper = create_test_paper_with_id("10.1234/test")
    section = create_test_section_with_type("results")
    
    with pytest.raises(ValueError, match="sentence must be a non-empty string"):
        encoder.encode(
            paper=paper,
            section=section,
            sentence=empty_sentence,
            extraction_method="regex_ner",
            confidence=0.85,
        )

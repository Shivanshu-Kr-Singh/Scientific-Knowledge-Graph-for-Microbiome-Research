"""
graph/test_provenance.py
------------------------
Unit tests for provenance tracking module.

Tests ProvenanceMetadata model validation and ProvenanceEncoder functionality.
"""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from graph.provenance import (
    ProvenanceMetadata,
    ProvenanceEncoder,
    REGISTERED_EXTRACTION_METHODS,
)
from nlp.enriched_record import EnrichedPaperRecord, ParsedSection


class TestProvenanceMetadata:
    """Test ProvenanceMetadata model validation."""
    
    def test_valid_provenance_metadata(self):
        """Test creating valid provenance metadata."""
        provenance = ProvenanceMetadata(
            paper_id="10.1234/example",
            section_type="results",
            source_sentence="Bacteroides fragilis was significantly increased in T2D patients.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.85,
        )
        
        assert provenance.paper_id == "10.1234/example"
        assert provenance.section_type == "results"
        assert provenance.confidence_score == 0.85
        assert provenance.validation_status == "unvalidated"
    
    def test_confidence_score_validation_lower_bound(self):
        """Test confidence score must be >= 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            ProvenanceMetadata(
                paper_id="10.1234/example",
                section_type="results",
                source_sentence="Test sentence.",
                extraction_method="regex_ner",
                extraction_timestamp=datetime.now(timezone.utc),
                extractor_version="1.0",
                confidence_score=-0.1,  # Invalid: below 0.0
            )
        
        assert "confidence_score" in str(exc_info.value)
    
    def test_confidence_score_validation_upper_bound(self):
        """Test confidence score must be <= 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            ProvenanceMetadata(
                paper_id="10.1234/example",
                section_type="results",
                source_sentence="Test sentence.",
                extraction_method="regex_ner",
                extraction_timestamp=datetime.now(timezone.utc),
                extractor_version="1.0",
                confidence_score=1.5,  # Invalid: above 1.0
            )
        
        assert "confidence_score" in str(exc_info.value)
    
    def test_confidence_score_boundary_values(self):
        """Test confidence score accepts boundary values 0.0 and 1.0."""
        # Test 0.0
        provenance_min = ProvenanceMetadata(
            paper_id="10.1234/example",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.0,
        )
        assert provenance_min.confidence_score == 0.0
        
        # Test 1.0
        provenance_max = ProvenanceMetadata(
            paper_id="10.1234/example",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=1.0,
        )
        assert provenance_max.confidence_score == 1.0
    
    def test_empty_source_sentence_validation(self):
        """Test source_sentence must be non-empty."""
        with pytest.raises(ValidationError) as exc_info:
            ProvenanceMetadata(
                paper_id="10.1234/example",
                section_type="results",
                source_sentence="",  # Invalid: empty string
                extraction_method="regex_ner",
                extraction_timestamp=datetime.now(timezone.utc),
                extractor_version="1.0",
                confidence_score=0.85,
            )
        
        assert "source_sentence" in str(exc_info.value)
    
    def test_invalid_section_type(self):
        """Test section_type must be one of allowed values."""
        with pytest.raises(ValidationError) as exc_info:
            ProvenanceMetadata(
                paper_id="10.1234/example",
                section_type="invalid_section",  # Invalid section type
                source_sentence="Test sentence.",
                extraction_method="regex_ner",
                extraction_timestamp=datetime.now(timezone.utc),
                extractor_version="1.0",
                confidence_score=0.85,
            )
        
        assert "section_type" in str(exc_info.value)
    
    def test_valid_section_types(self):
        """Test all valid section types are accepted."""
        valid_sections = ["abstract", "methods", "results", "discussion", "introduction", "other"]
        
        for section_type in valid_sections:
            provenance = ProvenanceMetadata(
                paper_id="10.1234/example",
                section_type=section_type,
                source_sentence="Test sentence.",
                extraction_method="regex_ner",
                extraction_timestamp=datetime.now(timezone.utc),
                extractor_version="1.0",
                confidence_score=0.85,
            )
            assert provenance.section_type == section_type
    
    def test_unregistered_extraction_method(self):
        """Test extraction_method must be registered."""
        with pytest.raises(ValidationError) as exc_info:
            ProvenanceMetadata(
                paper_id="10.1234/example",
                section_type="results",
                source_sentence="Test sentence.",
                extraction_method="unregistered_method",  # Not in REGISTERED_EXTRACTION_METHODS
                extraction_timestamp=datetime.now(timezone.utc),
                extractor_version="1.0",
                confidence_score=0.85,
            )
        
        error_msg = str(exc_info.value)
        assert "extraction_method" in error_msg
        assert "not registered" in error_msg
    
    def test_registered_extraction_methods(self):
        """Test all registered extraction methods are accepted."""
        for method in REGISTERED_EXTRACTION_METHODS:
            provenance = ProvenanceMetadata(
                paper_id="10.1234/example",
                section_type="results",
                source_sentence="Test sentence.",
                extraction_method=method,
                extraction_timestamp=datetime.now(timezone.utc),
                extractor_version="1.0",
                confidence_score=0.85,
            )
            assert provenance.extraction_method == method
    
    def test_invalid_validation_status(self):
        """Test validation_status must be one of allowed values."""
        with pytest.raises(ValidationError) as exc_info:
            ProvenanceMetadata(
                paper_id="10.1234/example",
                section_type="results",
                source_sentence="Test sentence.",
                extraction_method="regex_ner",
                extraction_timestamp=datetime.now(timezone.utc),
                extractor_version="1.0",
                confidence_score=0.85,
                validation_status="invalid_status",
            )
        
        assert "validation_status" in str(exc_info.value)
    
    def test_optional_fields(self):
        """Test optional fields can be None."""
        provenance = ProvenanceMetadata(
            paper_id="10.1234/example",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.85,
            # Optional fields not provided
        )
        
        assert provenance.sentence_offset is None
        assert provenance.llm_prompt_hash is None
        assert provenance.validator_id is None
        assert provenance.surrounding_context is None
        assert provenance.figure_table_ref is None


class TestProvenanceEncoder:
    """Test ProvenanceEncoder functionality."""
    
    def create_test_paper(self, doi: str = "10.1234/test") -> EnrichedPaperRecord:
        """Helper to create a test paper."""
        return EnrichedPaperRecord(
            doi=doi,
            title="Test Paper",
            abstract="Test abstract",
            year=2024,
            authors=[],
        )
    
    def create_test_section(self, section_type: str = "results") -> ParsedSection:
        """Helper to create a test section."""
        return ParsedSection(
            section_type=section_type,
            header="Results",
            content="Bacteroides fragilis was significantly increased in T2D patients. This is context.",
        )
    
    def test_encode_basic(self):
        """Test basic provenance encoding."""
        encoder = ProvenanceEncoder()
        paper = self.create_test_paper()
        section = self.create_test_section()
        sentence = "Bacteroides fragilis was significantly increased in T2D patients."
        
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence=sentence,
            extraction_method="regex_ner",
            confidence=0.85,
        )
        
        assert provenance.paper_id == "10.1234/test"
        assert provenance.section_type == "results"
        assert provenance.source_sentence == sentence
        assert provenance.extraction_method == "regex_ner"
        assert provenance.confidence_score == 0.85
        assert provenance.extractor_version == "1.0"
        assert provenance.validation_status == "unvalidated"
    
    def test_encode_with_optional_fields(self):
        """Test encoding with optional fields."""
        encoder = ProvenanceEncoder()
        paper = self.create_test_paper()
        section = self.create_test_section()
        sentence = "Test sentence."
        
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence=sentence,
            extraction_method="llm_extractor_v1.2",
            confidence=0.92,
            extractor_version="1.2",
            llm_prompt_hash="abc123def456",
            surrounding_context="Previous sentence. Test sentence. Next sentence.",
            figure_table_ref="Figure 2A",
            sentence_offset=42,
        )
        
        assert provenance.extractor_version == "1.2"
        assert provenance.llm_prompt_hash == "abc123def456"
        assert provenance.surrounding_context == "Previous sentence. Test sentence. Next sentence."
        assert provenance.figure_table_ref == "Figure 2A"
        assert provenance.sentence_offset == 42
    
    def test_encode_timestamp_is_utc(self):
        """Test extraction_timestamp is set to current UTC time."""
        encoder = ProvenanceEncoder()
        paper = self.create_test_paper()
        section = self.create_test_section()
        
        before = datetime.now(timezone.utc)
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence="Test sentence.",
            extraction_method="regex_ner",
            confidence=0.85,
        )
        after = datetime.now(timezone.utc)
        
        assert before <= provenance.extraction_timestamp <= after
        assert provenance.extraction_timestamp.tzinfo == timezone.utc
    
    def test_encode_empty_sentence_raises_error(self):
        """Test encoding with empty sentence raises ValueError."""
        encoder = ProvenanceEncoder()
        paper = self.create_test_paper()
        section = self.create_test_section()
        
        with pytest.raises(ValueError, match="sentence must be a non-empty string"):
            encoder.encode(
                paper=paper,
                section=section,
                sentence="",
                extraction_method="regex_ner",
                confidence=0.85,
            )
    
    def test_encode_whitespace_sentence_raises_error(self):
        """Test encoding with whitespace-only sentence raises ValueError."""
        encoder = ProvenanceEncoder()
        paper = self.create_test_paper()
        section = self.create_test_section()
        
        with pytest.raises(ValueError, match="sentence must be a non-empty string"):
            encoder.encode(
                paper=paper,
                section=section,
                sentence="   \n\t  ",
                extraction_method="regex_ner",
                confidence=0.85,
            )
    
    def test_encode_invalid_confidence_raises_error(self):
        """Test encoding with invalid confidence raises ValueError."""
        encoder = ProvenanceEncoder()
        paper = self.create_test_paper()
        section = self.create_test_section()
        
        # Test below 0.0
        with pytest.raises(ValueError, match="confidence must be in range"):
            encoder.encode(
                paper=paper,
                section=section,
                sentence="Test sentence.",
                extraction_method="regex_ner",
                confidence=-0.1,
            )
        
        # Test above 1.0
        with pytest.raises(ValueError, match="confidence must be in range"):
            encoder.encode(
                paper=paper,
                section=section,
                sentence="Test sentence.",
                extraction_method="regex_ner",
                confidence=1.5,
            )
    
    def test_encode_unregistered_method_raises_error(self):
        """Test encoding with unregistered extraction method raises ValueError."""
        encoder = ProvenanceEncoder()
        paper = self.create_test_paper()
        section = self.create_test_section()
        
        with pytest.raises(ValueError, match="not registered"):
            encoder.encode(
                paper=paper,
                section=section,
                sentence="Test sentence.",
                extraction_method="unregistered_method",
                confidence=0.85,
            )
    
    def test_encode_paper_without_identifier_raises_error(self):
        """Test encoding with paper lacking identifier raises ValueError."""
        encoder = ProvenanceEncoder()
        paper = EnrichedPaperRecord(
            # No doi, pmid, or title
            abstract="Test abstract",
            year=2024,
            authors=[],
        )
        section = self.create_test_section()
        
        with pytest.raises(ValueError, match="paper must have at least one of"):
            encoder.encode(
                paper=paper,
                section=section,
                sentence="Test sentence.",
                extraction_method="regex_ner",
                confidence=0.85,
            )
    
    def test_encode_prefers_doi_over_pmid(self):
        """Test encoder prefers DOI over PMID for paper_id."""
        encoder = ProvenanceEncoder()
        paper = EnrichedPaperRecord(
            doi="10.1234/test",
            pmid="12345678",
            title="Test Paper",
            abstract="Test abstract",
            year=2024,
            authors=[],
        )
        section = self.create_test_section()
        
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence="Test sentence.",
            extraction_method="regex_ner",
            confidence=0.85,
        )
        
        assert provenance.paper_id == "10.1234/test"
    
    def test_encode_uses_pmid_when_no_doi(self):
        """Test encoder uses PMID when DOI is not available."""
        encoder = ProvenanceEncoder()
        paper = EnrichedPaperRecord(
            pmid="12345678",
            title="Test Paper",
            abstract="Test abstract",
            year=2024,
            authors=[],
        )
        section = self.create_test_section()
        
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence="Test sentence.",
            extraction_method="regex_ner",
            confidence=0.85,
        )
        
        assert provenance.paper_id == "12345678"
    
    def test_encode_uses_title_when_no_doi_or_pmid(self):
        """Test encoder uses title when DOI and PMID are not available."""
        encoder = ProvenanceEncoder()
        paper = EnrichedPaperRecord(
            title="Test Paper Title",
            abstract="Test abstract",
            year=2024,
            authors=[],
        )
        section = self.create_test_section()
        
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence="Test sentence.",
            extraction_method="regex_ner",
            confidence=0.85,
        )
        
        assert provenance.paper_id == "Test Paper Title"
    
    def test_validate_provenance_valid(self):
        """Test validate_provenance returns True for valid provenance."""
        encoder = ProvenanceEncoder()
        paper = self.create_test_paper()
        section = self.create_test_section()
        
        provenance = encoder.encode(
            paper=paper,
            section=section,
            sentence="Test sentence.",
            extraction_method="regex_ner",
            confidence=0.85,
        )
        
        assert encoder.validate_provenance(provenance) is True
    
    def test_validate_provenance_missing_required_field(self):
        """Test validate_provenance returns False when required field is missing."""
        encoder = ProvenanceEncoder()
        
        # Create provenance with missing extraction_method
        provenance = ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.85,
        )
        
        # Manually set extraction_method to None to test validation
        provenance.extraction_method = None
        
        assert encoder.validate_provenance(provenance) is False
    
    def test_validate_provenance_invalid_confidence(self):
        """Test validate_provenance returns False for invalid confidence."""
        encoder = ProvenanceEncoder()
        
        provenance = ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version="1.0",
            confidence_score=0.85,
        )
        
        # Manually set invalid confidence to test validation
        provenance.confidence_score = 1.5
        
        assert encoder.validate_provenance(provenance) is False
    
    def test_validate_provenance_future_timestamp(self):
        """Test validate_provenance returns False for future timestamp."""
        encoder = ProvenanceEncoder()
        
        # Create provenance with future timestamp (more than 1 minute ahead)
        future_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        provenance = ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=future_time,
            extractor_version="1.0",
            confidence_score=0.85,
        )
        
        assert encoder.validate_provenance(provenance) is False
    
    def test_validate_provenance_allows_small_clock_skew(self):
        """Test validate_provenance allows small clock skew (< 1 minute)."""
        encoder = ProvenanceEncoder()
        
        # Create provenance with timestamp 30 seconds in the future
        near_future = datetime.now(timezone.utc) + timedelta(seconds=30)
        provenance = ProvenanceMetadata(
            paper_id="10.1234/test",
            section_type="results",
            source_sentence="Test sentence.",
            extraction_method="regex_ner",
            extraction_timestamp=near_future,
            extractor_version="1.0",
            confidence_score=0.85,
        )
        
        assert encoder.validate_provenance(provenance) is True

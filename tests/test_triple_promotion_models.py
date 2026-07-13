"""
Unit tests for triple promotion data model validation.

Tests PromotedTriple, OpenWorldClaim, and EvidenceItem Pydantic models
for field constraints, defaults, and validation rules.

Validates: Requirements 1.1, 3.1
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from graph.triple_promotion_models import (
    PromotedTriple,
    OpenWorldClaim,
    EvidenceItem,
    PaperMetadata,
)
from graph.provenance import ProvenanceMetadata


# --- Helpers ---


def _make_provenance(**overrides) -> ProvenanceMetadata:
    """Create a valid ProvenanceMetadata instance for testing."""
    defaults = {
        "paper_id": "PMID:12345",
        "section_type": "results",
        "source_sentence": "Akkermansia muciniphila produces propionate.",
        "extraction_method": "llm_triple_extractor",
        "extraction_timestamp": datetime.now(timezone.utc),
        "extractor_version": "llama3.1",
        "confidence_score": 0.85,
        "validation_status": "unvalidated",
    }
    defaults.update(overrides)
    return ProvenanceMetadata(**defaults)


def _make_promoted_triple(**overrides) -> dict:
    """Return kwargs dict for a valid PromotedTriple."""
    defaults = {
        "subject_id": "NCBI:txid239935",
        "subject_name": "Akkermansia muciniphila",
        "subject_type": "taxon",
        "subject_grounded": True,
        "subject_ontology": "NCBI Taxonomy",
        "object_id": "CHEBI:17272",
        "object_name": "propionate",
        "object_type": "metabolite",
        "object_grounded": True,
        "object_ontology": "ChEBI",
        "raw_predicate": "produces",
        "canonical_predicate": "PRODUCES",
        "predicate_category": "biosynthetic",
        "is_novel_predicate": False,
        "relationship_type": "PRODUCES",
        "provenance": _make_provenance(),
        "evidence_strength": "strong",
        "confidence": 0.85,
        "paper_id": "PMID:12345",
        "section_type": "results",
        "extracted_at": "2024-06-01T12:00:00Z",
    }
    defaults.update(overrides)
    return defaults


def _make_evidence_item(**overrides) -> dict:
    """Return kwargs dict for a valid EvidenceItem."""
    defaults = {
        "paper_id": "PMID:12345",
        "confidence": 0.85,
        "evidence_strength": "strong",
        "section_type": "results",
        "source_sentence": "Akkermansia muciniphila produces propionate.",
        "extraction_timestamp": "2024-06-01T12:00:00Z",
    }
    defaults.update(overrides)
    return defaults


def _make_open_world_claim(**overrides) -> dict:
    """Return kwargs dict for a valid OpenWorldClaim."""
    defaults = {
        "claim_id": "550e8400-e29b-41d4-a716-446655440000",
        "subject_id": "NCBI:txid239935",
        "subject_name": "Akkermansia muciniphila",
        "canonical_predicate": "PRODUCES",
        "object_id": "CHEBI:17272",
        "object_name": "propionate",
        "supporting_papers": ["PMID:12345", "PMID:67890"],
        "paper_count": 2,
        "consensus_confidence": 0.82,
        "evidence_strength": "moderate",
        "first_reported": "2024-01-15T10:00:00Z",
        "last_updated": "2024-06-01T12:00:00Z",
        "evidence_items": [],
    }
    defaults.update(overrides)
    return defaults


# =============================================================================
# PromotedTriple validation tests
# =============================================================================


class TestPromotedTripleValidation:
    """Tests for PromotedTriple model validation."""

    def test_valid_promoted_triple_creates_successfully(self):
        """A fully valid PromotedTriple should be created without error."""
        triple = PromotedTriple(**_make_promoted_triple())
        assert triple.subject_id == "NCBI:txid239935"
        assert triple.confidence == 0.85
        assert triple.evidence_strength == "strong"

    def test_confidence_at_minimum_boundary(self):
        """Confidence exactly at 0.5 should be accepted."""
        triple = PromotedTriple(**_make_promoted_triple(confidence=0.5))
        assert triple.confidence == 0.5

    def test_confidence_at_maximum_boundary(self):
        """Confidence exactly at 1.0 should be accepted."""
        triple = PromotedTriple(**_make_promoted_triple(confidence=1.0))
        assert triple.confidence == 1.0

    def test_confidence_below_minimum_rejected(self):
        """Confidence below 0.5 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PromotedTriple(**_make_promoted_triple(confidence=0.49))
        assert "confidence" in str(exc_info.value).lower() or "greater than" in str(exc_info.value).lower()

    def test_confidence_above_maximum_rejected(self):
        """Confidence above 1.0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PromotedTriple(**_make_promoted_triple(confidence=1.01))
        assert "confidence" in str(exc_info.value).lower() or "less than" in str(exc_info.value).lower()

    def test_confidence_zero_rejected(self):
        """Confidence of 0.0 (below 0.5) should be rejected."""
        with pytest.raises(ValidationError):
            PromotedTriple(**_make_promoted_triple(confidence=0.0))

    def test_confidence_negative_rejected(self):
        """Negative confidence should be rejected."""
        with pytest.raises(ValidationError):
            PromotedTriple(**_make_promoted_triple(confidence=-0.1))

    def test_evidence_strength_strong_accepted(self):
        """evidence_strength='strong' should be accepted."""
        triple = PromotedTriple(**_make_promoted_triple(evidence_strength="strong"))
        assert triple.evidence_strength == "strong"

    def test_evidence_strength_moderate_accepted(self):
        """evidence_strength='moderate' should be accepted."""
        triple = PromotedTriple(**_make_promoted_triple(evidence_strength="moderate"))
        assert triple.evidence_strength == "moderate"

    def test_evidence_strength_weak_accepted(self):
        """evidence_strength='weak' should be accepted."""
        triple = PromotedTriple(**_make_promoted_triple(evidence_strength="weak"))
        assert triple.evidence_strength == "weak"

    def test_evidence_strength_invalid_value_rejected(self):
        """Invalid evidence_strength values should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PromotedTriple(**_make_promoted_triple(evidence_strength="very_strong"))
        assert "evidence_strength" in str(exc_info.value)

    def test_evidence_strength_empty_string_rejected(self):
        """Empty string evidence_strength should be rejected."""
        with pytest.raises(ValidationError):
            PromotedTriple(**_make_promoted_triple(evidence_strength=""))

    def test_evidence_strength_case_sensitive(self):
        """Evidence strength validation is case-sensitive ('Strong' not valid)."""
        with pytest.raises(ValidationError):
            PromotedTriple(**_make_promoted_triple(evidence_strength="Strong"))

    def test_missing_required_field_rejected(self):
        """Omitting a required field should raise ValidationError."""
        kwargs = _make_promoted_triple()
        del kwargs["subject_id"]
        with pytest.raises(ValidationError):
            PromotedTriple(**kwargs)

    def test_optional_subject_ontology_none(self):
        """subject_ontology can be None when entity is ungrounded."""
        triple = PromotedTriple(**_make_promoted_triple(
            subject_ontology=None,
            subject_grounded=False,
            subject_id="ungrounded:akkermansia muciniphila",
        ))
        assert triple.subject_ontology is None
        assert triple.subject_grounded is False


# =============================================================================
# OpenWorldClaim validation tests
# =============================================================================


class TestOpenWorldClaimValidation:
    """Tests for OpenWorldClaim model defaults and validation."""

    def test_valid_claim_creates_successfully(self):
        """A fully valid OpenWorldClaim should be created without error."""
        claim = OpenWorldClaim(**_make_open_world_claim())
        assert claim.claim_id == "550e8400-e29b-41d4-a716-446655440000"
        assert claim.consensus_confidence == 0.82

    def test_claim_type_defaults_to_open_world(self):
        """claim_type should default to 'open_world' when not provided."""
        kwargs = _make_open_world_claim()
        del kwargs["claim_id"]  # Remove to test separately
        kwargs["claim_id"] = "test-claim-id"
        # Don't pass claim_type — it should default
        if "claim_type" in kwargs:
            del kwargs["claim_type"]
        claim = OpenWorldClaim(**kwargs)
        assert claim.claim_type == "open_world"

    def test_claim_type_can_be_overridden(self):
        """claim_type can be set to a custom value."""
        claim = OpenWorldClaim(**_make_open_world_claim(claim_type="custom_type"))
        assert claim.claim_type == "custom_type"

    def test_paper_count_zero_accepted(self):
        """paper_count of 0 should be accepted (non-negative)."""
        claim = OpenWorldClaim(**_make_open_world_claim(paper_count=0))
        assert claim.paper_count == 0

    def test_paper_count_positive_accepted(self):
        """Positive paper_count should be accepted."""
        claim = OpenWorldClaim(**_make_open_world_claim(paper_count=10))
        assert claim.paper_count == 10

    def test_paper_count_negative_rejected(self):
        """Negative paper_count should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            OpenWorldClaim(**_make_open_world_claim(paper_count=-1))
        assert "paper_count" in str(exc_info.value)

    def test_consensus_confidence_valid_range(self):
        """consensus_confidence within [0.0, 1.0] should be accepted."""
        claim = OpenWorldClaim(**_make_open_world_claim(consensus_confidence=0.0))
        assert claim.consensus_confidence == 0.0

        claim = OpenWorldClaim(**_make_open_world_claim(consensus_confidence=1.0))
        assert claim.consensus_confidence == 1.0

    def test_consensus_confidence_below_zero_rejected(self):
        """consensus_confidence below 0.0 should be rejected."""
        with pytest.raises(ValidationError):
            OpenWorldClaim(**_make_open_world_claim(consensus_confidence=-0.01))

    def test_consensus_confidence_above_one_rejected(self):
        """consensus_confidence above 1.0 should be rejected."""
        with pytest.raises(ValidationError):
            OpenWorldClaim(**_make_open_world_claim(consensus_confidence=1.01))

    def test_evidence_strength_invalid_rejected(self):
        """Invalid evidence_strength should be rejected on OpenWorldClaim."""
        with pytest.raises(ValidationError):
            OpenWorldClaim(**_make_open_world_claim(evidence_strength="invalid"))

    def test_supporting_papers_defaults_to_empty_list(self):
        """supporting_papers should default to an empty list."""
        kwargs = _make_open_world_claim()
        del kwargs["supporting_papers"]
        claim = OpenWorldClaim(**kwargs)
        assert claim.supporting_papers == []

    def test_evidence_items_defaults_to_empty_list(self):
        """evidence_items should default to an empty list."""
        kwargs = _make_open_world_claim()
        del kwargs["evidence_items"]
        claim = OpenWorldClaim(**kwargs)
        assert claim.evidence_items == []

    def test_paper_count_defaults_to_zero(self):
        """paper_count should default to 0."""
        kwargs = _make_open_world_claim()
        del kwargs["paper_count"]
        claim = OpenWorldClaim(**kwargs)
        assert claim.paper_count == 0

    def test_claim_with_evidence_items(self):
        """OpenWorldClaim should accept nested EvidenceItem objects."""
        items = [
            _make_evidence_item(paper_id="PMID:111", confidence=0.9),
            _make_evidence_item(paper_id="PMID:222", confidence=0.75),
        ]
        claim = OpenWorldClaim(**_make_open_world_claim(
            evidence_items=items,
        ))
        assert len(claim.evidence_items) == 2
        assert claim.evidence_items[0].paper_id == "PMID:111"
        assert claim.evidence_items[1].confidence == 0.75


# =============================================================================
# EvidenceItem validation tests
# =============================================================================


class TestEvidenceItemValidation:
    """Tests for EvidenceItem model validation."""

    def test_valid_evidence_item_creates_successfully(self):
        """A fully valid EvidenceItem should be created without error."""
        item = EvidenceItem(**_make_evidence_item())
        assert item.paper_id == "PMID:12345"
        assert item.confidence == 0.85

    def test_empty_source_sentence_rejected(self):
        """An empty source_sentence should be rejected (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            EvidenceItem(**_make_evidence_item(source_sentence=""))
        assert "source_sentence" in str(exc_info.value)

    def test_empty_paper_id_rejected(self):
        """An empty paper_id should be rejected (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            EvidenceItem(**_make_evidence_item(paper_id=""))
        assert "paper_id" in str(exc_info.value)

    def test_evidence_strength_strong_accepted(self):
        """evidence_strength='strong' should be accepted."""
        item = EvidenceItem(**_make_evidence_item(evidence_strength="strong"))
        assert item.evidence_strength == "strong"

    def test_evidence_strength_moderate_accepted(self):
        """evidence_strength='moderate' should be accepted."""
        item = EvidenceItem(**_make_evidence_item(evidence_strength="moderate"))
        assert item.evidence_strength == "moderate"

    def test_evidence_strength_weak_accepted(self):
        """evidence_strength='weak' should be accepted."""
        item = EvidenceItem(**_make_evidence_item(evidence_strength="weak"))
        assert item.evidence_strength == "weak"

    def test_evidence_strength_invalid_rejected(self):
        """Invalid evidence_strength values should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            EvidenceItem(**_make_evidence_item(evidence_strength="very_weak"))
        assert "evidence_strength" in str(exc_info.value)

    def test_evidence_strength_empty_rejected(self):
        """Empty evidence_strength should be rejected."""
        with pytest.raises(ValidationError):
            EvidenceItem(**_make_evidence_item(evidence_strength=""))

    def test_evidence_strength_case_sensitive(self):
        """Evidence strength should be case-sensitive."""
        with pytest.raises(ValidationError):
            EvidenceItem(**_make_evidence_item(evidence_strength="Moderate"))

    def test_confidence_range_valid(self):
        """Confidence within [0.0, 1.0] should be accepted for EvidenceItem."""
        item = EvidenceItem(**_make_evidence_item(confidence=0.0))
        assert item.confidence == 0.0

        item = EvidenceItem(**_make_evidence_item(confidence=1.0))
        assert item.confidence == 1.0

    def test_confidence_below_zero_rejected(self):
        """Confidence below 0.0 should be rejected."""
        with pytest.raises(ValidationError):
            EvidenceItem(**_make_evidence_item(confidence=-0.1))

    def test_confidence_above_one_rejected(self):
        """Confidence above 1.0 should be rejected."""
        with pytest.raises(ValidationError):
            EvidenceItem(**_make_evidence_item(confidence=1.1))

    def test_missing_required_field_rejected(self):
        """Omitting a required field should raise ValidationError."""
        kwargs = _make_evidence_item()
        del kwargs["section_type"]
        with pytest.raises(ValidationError):
            EvidenceItem(**kwargs)


# =============================================================================
# PaperMetadata validation tests
# =============================================================================


class TestPaperMetadataValidation:
    """Tests for PaperMetadata model validation."""

    def test_valid_paper_metadata_creates_successfully(self):
        """A valid PaperMetadata should be created without error."""
        meta = PaperMetadata(
            paper_id="PMID:12345",
            article_type="original_research",
            publication_year=2024,
            sections_available=["abstract", "results", "discussion"],
        )
        assert meta.paper_id == "PMID:12345"
        assert meta.article_type == "original_research"

    def test_empty_paper_id_rejected(self):
        """Empty paper_id should be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            PaperMetadata(
                paper_id="",
                article_type="original_research",
            )

    def test_publication_year_optional(self):
        """publication_year can be None."""
        meta = PaperMetadata(
            paper_id="PMID:12345",
            article_type="review",
        )
        assert meta.publication_year is None

    def test_sections_available_defaults_to_empty(self):
        """sections_available should default to an empty list."""
        meta = PaperMetadata(
            paper_id="PMID:12345",
            article_type="meta_analysis",
        )
        assert meta.sections_available == []

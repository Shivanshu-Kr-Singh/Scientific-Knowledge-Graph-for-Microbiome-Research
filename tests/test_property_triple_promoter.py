"""
Property-based tests for TriplePromoter provenance correctness.

Uses Hypothesis to verify correctness properties across a wide range
of generated inputs.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import re
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from graph.triple_promoter import TriplePromoter
from graph.triple_promotion_models import PaperMetadata


# ---------------------------------------------------------------------------
# Module-level mock helpers (used across Property 1, 4, etc.)
# ---------------------------------------------------------------------------

class MockEntityNormalizer:
    def normalize(self, text, entity_type):
        return {
            "grounded": False,
            "id": f"ungrounded:{text.lower()}",
            "canonical_name": text,
            "ontology": None,
            "confidence": 0.0,
        }


class MockPredicateRegistry:
    def normalize(self, raw_predicate):
        return ("RELATES_TO", False)

    def track_paper_occurrence(self, raw_predicate, paper_id):
        return ("RELATES_TO", False, False)

    def get_category(self, canonical_predicate):
        return "generic"

    def promote_predicate(self, raw_predicate):
        return raw_predicate.upper().replace(" ", "_")


class MockEvidenceClassifier:
    def classify_single(self, confidence, section_type, article_type):
        return "weak"


# ---------------------------------------------------------------------------
# Shared mock-helper builders (kept for Properties 2 and 3)
# ---------------------------------------------------------------------------

def _make_mock_entity_normalizer(grounded: bool = False):
    """Return a mock EntityNormalizer that always returns ungrounded results."""
    mock = MagicMock()
    mock.normalize.return_value = {
        "id": "ungrounded:entity",
        "canonical_name": "entity",
        "ontology": None,
        "grounded": grounded,
        "confidence": 0.0,
    }
    return mock


def _make_mock_predicate_registry():
    """Return a mock PredicateRegistry with sensible defaults."""
    mock = MagicMock()
    mock.track_paper_occurrence.return_value = ("RELATES_TO", False, False)
    mock.normalize.return_value = ("RELATES_TO", False)
    mock.get_category.return_value = "unknown"
    return mock


def _make_mock_evidence_classifier():
    """Return a mock EvidenceStrengthClassifier that always returns 'moderate'."""
    mock = MagicMock()
    mock.classify_single.return_value = "moderate"
    return mock


def _build_promoter() -> TriplePromoter:
    """Construct a TriplePromoter with all-mock collaborators."""
    return TriplePromoter(
        entity_normalizer=_make_mock_entity_normalizer(),
        predicate_registry=_make_mock_predicate_registry(),
        evidence_classifier=_make_mock_evidence_classifier(),
        promotion_threshold=5,
    )


def _build_promoter_with_class_mocks() -> TriplePromoter:
    """Construct a TriplePromoter using the module-level class-based mocks."""
    return TriplePromoter(
        entity_normalizer=MockEntityNormalizer(),
        predicate_registry=MockPredicateRegistry(),
        evidence_classifier=MockEvidenceClassifier(),
        promotion_threshold=5,
    )


def _base_raw_triple(evidence: str, section_content: str | None = None) -> dict:
    """Return a minimal valid raw triple dict."""
    return {
        "subject": "SubjectEntity",
        "subject_type": "gene",
        "predicate": "affects",
        "object": "ObjectEntity",
        "object_type": "disease",
        "confidence": 0.75,
        "evidence": evidence,
        "paper_id": "paper_001",
        "section_type": "results",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "section_content": section_content,
    }


_PAPER_META = PaperMetadata(
    paper_id="paper_001",
    article_type="original_research",
    publication_year=2024,
    sections_available=["results"],
)


# ---------------------------------------------------------------------------
# Property 2: Sentence offset correctness
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 2: Sentence offset correctness
class TestProperty2SentenceOffsetCorrectness:
    """
    Property 2: Sentence offset correctness

    For any section_content and evidence that is a substring of
    section_content, sentence_offset == section_content.find(evidence).

    Validates: Requirements 1.2
    """

    @given(
        prefix=st.text(
            min_size=0,
            max_size=30,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        ),
        evidence=st.text(
            min_size=5,
            max_size=50,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        ).filter(lambda s: s.strip() and len(s.strip()) >= 3),
        suffix=st.text(
            min_size=0,
            max_size=30,
            alphabet=st.characters(blacklist_categories=("Cs",)),
        ),
    )
    @settings(max_examples=100)
    def test_sentence_offset_equals_find(self, prefix: str, evidence: str, suffix: str):
        """
        sentence_offset in provenance SHALL equal section_content.find(evidence)
        when evidence is a substring of section_content.

        Validates: Requirements 1.2
        """
        # Use stripped evidence (as the promoter strips internally)
        stripped = evidence.strip()
        assume(stripped)  # must be non-empty after stripping

        section_content = prefix + stripped + suffix

        # Ensure stripped evidence does not appear in prefix (no ambiguous offset)
        assume(prefix.find(stripped) == -1)

        # The expected offset is simply the length of the prefix
        expected_offset_stripped = len(prefix)
        # Sanity-check oracle against actual string method
        assume(section_content.find(stripped) == expected_offset_stripped)

        promoter = _build_promoter()
        raw = _base_raw_triple(evidence=stripped, section_content=section_content)
        result = promoter.promote_triple(raw, _PAPER_META)

        # promote_triple may return None only if quality gate rejects — but
        # evidence is non-empty (filtered above) and confidence = 0.75, so
        # it must not be None.
        assert result is not None, (
            f"promote_triple returned None unexpectedly. "
            f"evidence={evidence!r}, section_content={section_content!r}"
        )

        assert result.provenance.sentence_offset == expected_offset_stripped, (
            f"sentence_offset mismatch: got {result.provenance.sentence_offset}, "
            f"expected {expected_offset_stripped}. "
            f"section_content={section_content!r}, evidence={evidence.strip()!r}"
        )


# ---------------------------------------------------------------------------
# Property 3: Surrounding context extraction
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 3: Surrounding context extraction
class TestProperty3SurroundingContextExtraction:
    """
    Property 3: Surrounding context extraction

    For a sentence at index i in a list, surrounding_context contains
    sentences from max(0, i-2) to min(len-1, i+2) inclusive, preserving
    order.

    Validates: Requirements 1.3
    """

    # The TriplePromoter splits on this pattern
    _SPLIT_RE = re.compile(r"(?<=[.?!])\s+")

    @given(
        sentences=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Ll", "Lu", "Nd"),
                    min_codepoint=32,
                    max_codepoint=122,
                ),
                min_size=5,
                max_size=20,
            ),
            min_size=3,
            max_size=8,
        ),
        evidence_idx_raw=st.integers(min_value=0),
    )
    @settings(max_examples=50)
    def test_surrounding_context_contains_correct_neighbors(
        self, sentences: list, evidence_idx_raw: int
    ):
        """
        surrounding_context SHALL contain exactly the sentences from
        max(0, i-2) to min(len-1, i+2) inclusive.

        Validates: Requirements 1.3
        """
        # Derive a valid index (modulo keeps it in range)
        evidence_idx = evidence_idx_raw % len(sentences)

        # Build section_content by joining sentences with ". " separator and
        # appending "." to each so that the sentence-split regex fires on them.
        section_content = " ".join(s + "." for s in sentences)

        # The evidence is the raw sentence text (no trailing period).
        evidence = sentences[evidence_idx]

        # Skip if the evidence is empty or whitespace after stripping
        assume(evidence.strip() and len(evidence.strip()) >= 3)

        # Skip degenerate cases where evidence appears earlier in the text
        # (would confuse the index lookup in TriplePromoter._extract_surrounding_context)
        # Check that split finds evidence at the right position
        split_sentences = self._SPLIT_RE.split(section_content)
        # Find the first sentence that contains our evidence text
        found_idx = None
        for i, sent in enumerate(split_sentences):
            if evidence in sent or sent in evidence:
                found_idx = i
                break
        assume(found_idx is not None)

        # Compute expected context boundaries
        start = max(0, found_idx - 2)
        end = min(len(split_sentences), found_idx + 3)  # exclusive upper bound
        expected_sentences = split_sentences[start:end]
        expected_context = " ".join(expected_sentences)

        # Build and call the promoter
        promoter = _build_promoter()
        raw = _base_raw_triple(evidence=evidence, section_content=section_content)

        result = promoter.promote_triple(raw, _PAPER_META)

        assert result is not None, (
            f"promote_triple returned None unexpectedly. "
            f"evidence={evidence!r}, section_content={section_content!r}"
        )

        actual_context = result.provenance.surrounding_context

        assert actual_context == expected_context, (
            f"surrounding_context mismatch.\n"
            f"  evidence={evidence!r}\n"
            f"  evidence_idx (in split)={found_idx}\n"
            f"  expected={expected_context!r}\n"
            f"  got={actual_context!r}\n"
            f"  split_sentences={split_sentences}"
        )


# ---------------------------------------------------------------------------
# Property 1: Provenance completeness
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 1: Provenance completeness
class TestProperty1ProvenanceCompleteness:
    """
    Property 1: Provenance completeness

    For any raw triple with non-empty evidence and confidence >= 0.5,
    the resulting PromotedTriple.provenance SHALL have:
    - paper_id == input paper_id
    - source_sentence == evidence.strip()
    - extraction_method == "llm_triple_extractor"
    - confidence_score == input confidence
    - validation_status == "unvalidated"
    - extraction_timestamp is a valid UTC datetime (isinstance check)

    Validates: Requirements 1.1, 1.4
    """

    @given(
        paper_id=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=3,
            max_size=20,
        ),
        evidence=st.text(min_size=3, max_size=100).filter(lambda s: bool(s.strip())),
        confidence=st.floats(
            min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        section_type=st.sampled_from(
            ["results", "discussion", "abstract", "introduction"]
        ),
    )
    @settings(max_examples=100)
    def test_provenance_completeness(
        self,
        paper_id: str,
        evidence: str,
        confidence: float,
        section_type: str,
    ):
        """
        promote_triple SHALL produce a PromotedTriple whose provenance fields
        exactly match the input values, with extraction_method fixed to
        "llm_triple_extractor" and validation_status fixed to "unvalidated".

        Validates: Requirements 1.1, 1.4
        """
        raw_triple = {
            "subject": "A",
            "subject_type": "taxon",
            "predicate": "relates",
            "object": "B",
            "object_type": "metabolite",
            "confidence": confidence,
            "evidence": evidence,
            "paper_id": paper_id,
            "section_type": section_type,
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        paper_metadata = PaperMetadata(
            paper_id=paper_id,
            article_type="original_research",
        )

        promoter = _build_promoter_with_class_mocks()
        result = promoter.promote_triple(raw_triple, paper_metadata)

        assert result is not None, (
            f"promote_triple returned None unexpectedly for valid triple "
            f"(evidence={evidence!r}, confidence={confidence})"
        )

        prov = result.provenance

        assert prov.paper_id == paper_id, (
            f"paper_id mismatch: expected {paper_id!r}, got {prov.paper_id!r}"
        )
        assert prov.source_sentence == evidence.strip(), (
            f"source_sentence mismatch: expected {evidence.strip()!r}, "
            f"got {prov.source_sentence!r}"
        )
        assert prov.extraction_method == "llm_triple_extractor", (
            f"extraction_method mismatch: expected 'llm_triple_extractor', "
            f"got {prov.extraction_method!r}"
        )
        assert prov.confidence_score == confidence, (
            f"confidence_score mismatch: expected {confidence}, "
            f"got {prov.confidence_score}"
        )
        assert prov.validation_status == "unvalidated", (
            f"validation_status mismatch: expected 'unvalidated', "
            f"got {prov.validation_status!r}"
        )
        assert isinstance(prov.extraction_timestamp, datetime), (
            f"extraction_timestamp is not a datetime instance: "
            f"{type(prov.extraction_timestamp)}"
        )
        # Must be UTC-aware
        assert prov.extraction_timestamp.tzinfo is not None, (
            "extraction_timestamp has no timezone info (expected UTC-aware datetime)"
        )


# ---------------------------------------------------------------------------
# Property 4: Quality gate rejection
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 4: Quality gate rejection
class TestProperty4QualityGateRejection:
    """
    Property 4: Quality gate rejection

    - Low confidence (< 0.5) → promote_triple returns None
    - Whitespace-only evidence → promote_triple returns None
    - Valid triple (non-empty evidence + confidence >= 0.5) → result is not None

    Validates: Requirements 1.5
    """

    @given(
        confidence=st.floats(
            min_value=0.0,
            max_value=0.4999,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100)
    def test_rejects_low_confidence(self, confidence: float):
        """
        promote_triple SHALL return None for confidence < 0.5,
        regardless of other fields.

        Validates: Requirements 1.5
        """
        raw_triple = {
            "subject": "A",
            "subject_type": "taxon",
            "predicate": "relates",
            "object": "B",
            "object_type": "metabolite",
            "confidence": confidence,
            "evidence": "This is a valid non-empty evidence sentence.",
            "paper_id": "paper_001",
            "section_type": "results",
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        paper_metadata = PaperMetadata(
            paper_id="paper_001",
            article_type="original_research",
        )
        promoter = _build_promoter_with_class_mocks()
        result = promoter.promote_triple(raw_triple, paper_metadata)
        assert result is None, (
            f"Expected None for confidence={confidence}, got a PromotedTriple"
        )

    @given(
        evidence=st.sampled_from(["", " ", "  ", "\t", "\n", "   \t  "]),
    )
    @settings(max_examples=100)
    def test_rejects_empty_evidence(self, evidence: str):
        """
        promote_triple SHALL return None when evidence is empty or whitespace-only.

        Validates: Requirements 1.5
        """
        raw_triple = {
            "subject": "A",
            "subject_type": "taxon",
            "predicate": "relates",
            "object": "B",
            "object_type": "metabolite",
            "confidence": 0.8,
            "evidence": evidence,
            "paper_id": "paper_001",
            "section_type": "results",
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        paper_metadata = PaperMetadata(
            paper_id="paper_001",
            article_type="original_research",
        )
        promoter = _build_promoter_with_class_mocks()
        result = promoter.promote_triple(raw_triple, paper_metadata)
        assert result is None, (
            f"Expected None for whitespace-only evidence={evidence!r}, "
            f"got a PromotedTriple"
        )

    @given(
        paper_id=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=3,
            max_size=20,
        ),
        evidence=st.text(min_size=3, max_size=100).filter(lambda s: bool(s.strip())),
        confidence=st.floats(
            min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        section_type=st.sampled_from(
            ["results", "discussion", "abstract", "introduction"]
        ),
    )
    @settings(max_examples=100)
    def test_accepts_valid_triple(
        self,
        paper_id: str,
        evidence: str,
        confidence: float,
        section_type: str,
    ):
        """
        promote_triple SHALL NOT return None for valid triples
        (non-empty evidence AND confidence >= 0.5).

        Validates: Requirements 1.5
        """
        raw_triple = {
            "subject": "A",
            "subject_type": "taxon",
            "predicate": "relates",
            "object": "B",
            "object_type": "metabolite",
            "confidence": confidence,
            "evidence": evidence,
            "paper_id": paper_id,
            "section_type": section_type,
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        paper_metadata = PaperMetadata(
            paper_id=paper_id,
            article_type="original_research",
        )
        promoter = _build_promoter_with_class_mocks()
        result = promoter.promote_triple(raw_triple, paper_metadata)
        assert result is not None, (
            f"Expected a PromotedTriple but got None for valid triple "
            f"(evidence={evidence!r}, confidence={confidence})"
        )


# ---------------------------------------------------------------------------
# Property 5: Entity ID format based on grounding status
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 5: Entity ID format based on grounding status
class TestProperty5EntityIDFormat:
    """
    Property 5: Entity ID format based on grounding status

    - If EntityNormalizer returns grounded=True with an ontology_id,
      then PromotedTriple SHALL use that ontology_id as subject_id (or object_id),
      subject_grounded=True, and subject_ontology matching the returned ontology.
    - If EntityNormalizer returns grounded=False, the entity ID SHALL equal
      "ungrounded:{text.lower()}", subject_grounded=False, subject_ontology=None.

    Validates: Requirements 2.3, 2.4
    """

    @given(
        ontology_id=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=48, max_codepoint=122),
            min_size=3,
            max_size=30,
        ),
        canonical_name=st.text(min_size=2, max_size=30).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_grounded_entity_uses_ontology_id(self, ontology_id: str, canonical_name: str):
        """
        When EntityNormalizer returns grounded=True, the PromotedTriple SHALL
        use the ontology_id as subject_id, set subject_grounded=True, and
        store the ontology name.

        Validates: Requirements 2.3
        """

        class GroundedEntityNormalizer:
            def normalize(self, text, entity_type):
                return {
                    "grounded": True,
                    "id": ontology_id,
                    "canonical_name": canonical_name,
                    "ontology": "TestOntology",
                    "confidence": 0.9,
                }

        promoter = TriplePromoter(
            entity_normalizer=GroundedEntityNormalizer(),
            predicate_registry=MockPredicateRegistry(),
            evidence_classifier=MockEvidenceClassifier(),
        )
        raw_triple = {
            "subject": "Akkermansia muciniphila",
            "subject_type": "taxon",
            "predicate": "produces",
            "object": "propionate",
            "object_type": "metabolite",
            "confidence": 0.85,
            "evidence": "Akkermansia muciniphila produces propionate in the gut.",
            "paper_id": "PMID:12345",
            "section_type": "results",
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        paper_metadata = PaperMetadata(paper_id="PMID:12345", article_type="original_research")

        result = promoter.promote_triple(raw_triple, paper_metadata)
        assert result is not None

        assert result.subject_id == ontology_id, (
            f"Expected subject_id={ontology_id!r}, got {result.subject_id!r}"
        )
        assert result.subject_grounded is True, (
            f"Expected subject_grounded=True, got {result.subject_grounded}"
        )
        assert result.subject_ontology == "TestOntology", (
            f"Expected subject_ontology='TestOntology', got {result.subject_ontology!r}"
        )

    @given(
        entity_text=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs"), min_codepoint=32, max_codepoint=122),
            min_size=2,
            max_size=30,
        ).filter(lambda s: s.strip()),
    )
    @settings(max_examples=100)
    def test_ungrounded_entity_uses_fallback_id(self, entity_text: str):
        """
        When EntityNormalizer returns grounded=False, the PromotedTriple SHALL
        use "ungrounded:{text.lower()}" as subject_id, set subject_grounded=False,
        and subject_ontology=None.

        Validates: Requirements 2.4
        """

        class UngroundedEntityNormalizer:
            def normalize(self, text, entity_type):
                return {
                    "grounded": False,
                    "id": f"ungrounded:{text.lower()}",
                    "canonical_name": text,
                    "ontology": None,
                    "confidence": 0.0,
                }

        promoter = TriplePromoter(
            entity_normalizer=UngroundedEntityNormalizer(),
            predicate_registry=MockPredicateRegistry(),
            evidence_classifier=MockEvidenceClassifier(),
        )
        raw_triple = {
            "subject": entity_text,
            "subject_type": "unknown",
            "predicate": "relates",
            "object": "B",
            "object_type": "unknown",
            "confidence": 0.6,
            "evidence": "Some valid evidence sentence here.",
            "paper_id": "PMID:99999",
            "section_type": "results",
            "extracted_at": "2024-01-01T00:00:00Z",
        }
        paper_metadata = PaperMetadata(paper_id="PMID:99999", article_type="review")

        result = promoter.promote_triple(raw_triple, paper_metadata)
        assert result is not None

        expected_id = f"ungrounded:{entity_text.lower()}"
        assert result.subject_id == expected_id, (
            f"Expected subject_id={expected_id!r}, got {result.subject_id!r}"
        )
        assert result.subject_grounded is False, (
            f"Expected subject_grounded=False, got {result.subject_grounded}"
        )
        assert result.subject_ontology is None, (
            f"Expected subject_ontology=None, got {result.subject_ontology!r}"
        )

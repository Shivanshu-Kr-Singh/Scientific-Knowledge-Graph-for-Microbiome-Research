"""
Integration tests for the end-to-end triple promotion flow.

Tests the full pipeline: extract raw triples, promote them, aggregate into
claims — using mocked collaborators so no real Neo4j or Ollama connection
is required.

Requirements: 1.1, 2.1, 3.1, 4.4
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from graph.evidence_strength_classifier import EvidenceStrengthClassifier
from graph.predicate_registry import PredicateRegistry
from graph.provenance import ProvenanceMetadata
from graph.triple_promoter import TriplePromoter
from graph.triple_promotion_models import (
    PaperMetadata,
    PromotedTriple,
)


# ---------------------------------------------------------------------------
# Shared mock helpers (kept local — mirrors patterns from existing PBT tests)
# ---------------------------------------------------------------------------

class MockEntityNormalizerUngrounded:
    """Always returns ungrounded results."""

    def normalize(self, text, entity_type):
        return {
            "grounded": False,
            "id": f"ungrounded:{text.lower()}",
            "canonical_name": text,
            "ontology": None,
            "confidence": 0.0,
        }


class MockPredicateRegistryRelatesToFalse:
    """Returns RELATES_TO / not-known / not-newly-promoted for every predicate."""

    def normalize(self, raw_predicate):
        return ("RELATES_TO", False)

    def track_paper_occurrence(self, raw_predicate, paper_id):
        return ("RELATES_TO", False, False)

    def get_category(self, canonical_predicate):
        return "generic"

    def promote_predicate(self, raw_predicate):
        return raw_predicate.upper().replace(" ", "_")

    def get_promotion_threshold(self):
        return 5

    def get_novel_predicates(self, min_frequency=2):
        return []


class MockEvidenceClassifierWeak:
    """Always classifies individual triples as 'weak'."""

    def classify_single(self, confidence, section_type, article_type):
        return "weak"

    def classify_claim(self, individual_strengths, paper_count):
        # Delegate to real logic
        return EvidenceStrengthClassifier().classify_claim(individual_strengths, paper_count)


def _fixed_provenance(paper_id: str, sentence: str = "Evidence sentence.") -> ProvenanceMetadata:
    """Build a minimal valid ProvenanceMetadata."""
    return ProvenanceMetadata(
        paper_id=paper_id,
        section_type="results",
        source_sentence=sentence,
        sentence_offset=None,
        extraction_method="llm_triple_extractor",
        extraction_timestamp=datetime.now(timezone.utc),
        extractor_version="llama3.1",
        confidence_score=0.75,
        validation_status="unvalidated",
        surrounding_context=None,
    )


def _build_promoted_triple(
    subject_id: str,
    object_id: str,
    predicate: str,
    paper_id: str,
    confidence: float,
    extracted_at: str,
    subject_name: str = "Subject",
    object_name: str = "Object",
    source_sentence: str = "Evidence sentence.",
) -> PromotedTriple:
    """
    Helper to construct a PromotedTriple for use in aggregation tests.
    Mirrors the pattern from test_property_claim_aggregation.py.
    """
    return PromotedTriple(
        subject_id=subject_id,
        subject_name=subject_name,
        subject_type="taxon",
        subject_grounded=True,
        subject_ontology="TestOntology",
        object_id=object_id,
        object_name=object_name,
        object_type="metabolite",
        object_grounded=True,
        object_ontology="TestOntology",
        raw_predicate=predicate.lower(),
        canonical_predicate=predicate,
        predicate_category="generic",
        is_novel_predicate=False,
        relationship_type=predicate,
        provenance=_fixed_provenance(paper_id, source_sentence),
        evidence_strength="weak",
        confidence=confidence,
        paper_id=paper_id,
        section_type="results",
        extracted_at=extracted_at,
    )


# ---------------------------------------------------------------------------
# Test 1: Extract → Promote → Aggregate → Load round trip
# ---------------------------------------------------------------------------

class TestPromoteBatchAndAggregateRoundTrip:
    """
    Integration test: promote_batch followed by aggregate_claims forms a
    complete extraction-to-claim pipeline without real Neo4j or Ollama.

    Requirements: 1.1, 2.1, 3.1
    """

    def test_promote_batch_and_aggregate_round_trip(self):
        """
        Extract raw triples, promote them, aggregate into claims.

        Steps:
        1. Build mocked collaborators: ungrounded EntityNormalizer,
           RELATES_TO PredicateRegistry, 'weak' EvidenceClassifier.
        2. Create TriplePromoter with these mocks.
        3. Build 3 raw_triple dicts sharing the same
           (subject="Akkermansia", predicate="produces", object="propionate")
           but with distinct paper_ids and confidence >= 0.5.
        4. Create a PaperMetadata.
        5. Call promote_batch(raw_triples, paper_metadata).
        6. Assert all 3 are returned (none rejected).
        7. Assert each promoted triple has
           provenance.extraction_method == "llm_triple_extractor".
        8. Assert each promoted triple has
           subject_id == "ungrounded:akkermansia".
        9. Call aggregate_claims(promoted_triples).
        10. Assert exactly 1 claim with paper_count == 3.
        11. Assert claim.consensus_confidence == mean of the 3
            confidence scores (within 1e-9).

        Requirements: 1.1, 2.1, 3.1
        """
        promoter = TriplePromoter(
            entity_normalizer=MockEntityNormalizerUngrounded(),
            predicate_registry=MockPredicateRegistryRelatesToFalse(),
            evidence_classifier=MockEvidenceClassifierWeak(),
            promotion_threshold=5,
        )

        # Three raw triples: same SPO surface text, distinct paper_ids, all passing quality gate
        confidences = [0.6, 0.75, 0.9]
        raw_triples = [
            {
                "subject": "Akkermansia",
                "subject_type": "taxon",
                "predicate": "produces",
                "object": "propionate",
                "object_type": "metabolite",
                "confidence": confidences[i],
                "evidence": f"Akkermansia produces propionate (paper {i+1}).",
                "paper_id": f"paper_00{i+1}",
                "section_type": "results",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }
            for i in range(3)
        ]

        paper_metadata = PaperMetadata(
            paper_id="paper_batch",
            article_type="original_research",
            publication_year=2024,
            sections_available=["results"],
        )

        # ── Step 5: promote_batch ──────────────────────────────────────────
        promoted_triples = promoter.promote_batch(raw_triples, paper_metadata)

        # ── Step 6: all 3 returned, none rejected ─────────────────────────
        assert len(promoted_triples) == 3, (
            f"Expected 3 promoted triples, got {len(promoted_triples)}"
        )

        # ── Step 7: provenance.extraction_method == "llm_triple_extractor" ─
        for pt in promoted_triples:
            assert pt.provenance.extraction_method == "llm_triple_extractor", (
                f"Expected extraction_method='llm_triple_extractor', "
                f"got {pt.provenance.extraction_method!r} for paper_id={pt.paper_id!r}"
            )

        # ── Step 8: subject_id == "ungrounded:akkermansia" ────────────────
        for pt in promoted_triples:
            assert pt.subject_id == "ungrounded:akkermansia", (
                f"Expected subject_id='ungrounded:akkermansia', "
                f"got {pt.subject_id!r} for paper_id={pt.paper_id!r}"
            )

        # ── Step 9: aggregate_claims ───────────────────────────────────────
        claims = promoter.aggregate_claims(promoted_triples)

        # ── Step 10: exactly 1 claim with paper_count == 3 ────────────────
        assert len(claims) == 1, (
            f"Expected exactly 1 claim after aggregation, got {len(claims)}"
        )
        assert claims[0].paper_count == 3, (
            f"Expected paper_count == 3, got {claims[0].paper_count}"
        )

        # ── Step 11: consensus_confidence == arithmetic mean ───────────────
        expected_mean = sum(confidences) / len(confidences)
        actual_confidence = claims[0].consensus_confidence
        assert abs(actual_confidence - expected_mean) < 1e-9, (
            f"consensus_confidence mismatch: expected {expected_mean}, "
            f"got {actual_confidence}. confidences={confidences}"
        )


# ---------------------------------------------------------------------------
# Test 2: Retroactive edge promotion after threshold reached
# ---------------------------------------------------------------------------

class TestCheckPredicatePromotionAtThreshold:
    """
    Integration test: check_predicate_promotion promotes predicates that
    have hit the paper-frequency threshold.

    Requirements: 4.4
    """

    def test_check_predicate_promotion_promotes_at_threshold(self, tmp_path):
        """
        When a novel predicate reaches the promotion threshold via
        track_paper_occurrence, check_predicate_promotion() should return
        its canonical form.

        Steps:
        1. Create a real PredicateRegistry backed by a temp SQLite DB.
        2. Set PREDICATE_PROMOTION_THRESHOLD=3 via os.environ patch.
        3. Call track_paper_occurrence("xylophages interaction", paper_id)
           for 3 distinct papers.
        4. Create a TriplePromoter with this registry (mocked entity_normalizer
           and evidence_classifier).
        5. Call promoter.check_predicate_promotion().
        6. Assert the returned list contains the canonical form
           ("XYLOPHAGES_INTERACTION").

        Requirements: 4.4
        """
        db_path = tmp_path / "test_pred_registry.db"
        novel_predicate = "xylophages interaction"

        with patch("graph.predicate_registry.REGISTRY_DB_PATH", db_path), \
             patch.dict(os.environ, {"PREDICATE_PROMOTION_THRESHOLD": "3"}):

            # Step 1 & 2: fresh PredicateRegistry with temp DB and threshold=3
            registry = PredicateRegistry()

            # Step 3: track 3 distinct papers for the novel predicate
            paper_ids = ["paper_1", "paper_2", "paper_3"]
            for pid in paper_ids:
                registry.track_paper_occurrence(novel_predicate, pid)

            # Step 4: build TriplePromoter with real registry
            promoter = TriplePromoter(
                entity_normalizer=MockEntityNormalizerUngrounded(),
                predicate_registry=registry,
                evidence_classifier=MockEvidenceClassifierWeak(),
                promotion_threshold=3,
            )

            # Step 5: call check_predicate_promotion
            promoted_list = promoter.check_predicate_promotion()

            # Step 6: verify the canonical form is returned
            expected_canonical = "XYLOPHAGES_INTERACTION"
            assert expected_canonical in promoted_list, (
                f"Expected {expected_canonical!r} in promoted list, "
                f"got {promoted_list!r}"
            )


# ---------------------------------------------------------------------------
# Test 3: SUBJECT_OF/OBJECT_OF structure in claims (evidence_items)
# ---------------------------------------------------------------------------

class TestAggregateClaimsEvidenceItemsStructure:
    """
    Integration test: aggregate_claims creates claims with correctly
    structured evidence_items linking to both subject and object.

    Requirements: 3.1
    """

    def test_aggregate_claims_evidence_items_structure(self):
        """
        Verify that each OpenWorldClaim has evidence_items populated with
        correct paper_id, confidence, and source_sentence values.

        Steps:
        1. Build 2 PromotedTriples from distinct papers with same canonical triple.
        2. Call aggregate_claims.
        3. Assert 1 claim with 2 evidence_items.
        4. Assert evidence_items[0].paper_id and evidence_items[1].paper_id
           match the input paper_ids.
        5. Assert evidence_items[0].source_sentence == the provenance
           source_sentence from that triple.

        Requirements: 3.1
        """
        ts = datetime.now(timezone.utc).isoformat()

        paper_id_a = "paper_alpha"
        paper_id_b = "paper_beta"
        sentence_a = "Akkermansia produces propionate in the gut."
        sentence_b = "Propionate is produced by Akkermansia."

        triple_a = _build_promoted_triple(
            subject_id="ungrounded:akkermansia",
            object_id="ungrounded:propionate",
            predicate="PRODUCES",
            paper_id=paper_id_a,
            confidence=0.75,
            extracted_at=ts,
            subject_name="Akkermansia",
            object_name="propionate",
            source_sentence=sentence_a,
        )
        triple_b = _build_promoted_triple(
            subject_id="ungrounded:akkermansia",
            object_id="ungrounded:propionate",
            predicate="PRODUCES",
            paper_id=paper_id_b,
            confidence=0.80,
            extracted_at=ts,
            subject_name="Akkermansia",
            object_name="propionate",
            source_sentence=sentence_b,
        )

        promoter = TriplePromoter(
            entity_normalizer=MockEntityNormalizerUngrounded(),
            predicate_registry=MockPredicateRegistryRelatesToFalse(),
            evidence_classifier=MockEvidenceClassifierWeak(),
            promotion_threshold=5,
        )

        claims = promoter.aggregate_claims([triple_a, triple_b])

        # ── Step 3: 1 claim with 2 evidence_items ─────────────────────────
        assert len(claims) == 1, (
            f"Expected exactly 1 claim, got {len(claims)}"
        )
        claim = claims[0]
        assert len(claim.evidence_items) == 2, (
            f"Expected 2 evidence_items, got {len(claim.evidence_items)}"
        )

        # ── Step 4: paper_ids match input ─────────────────────────────────
        evidence_paper_ids = [ei.paper_id for ei in claim.evidence_items]
        assert paper_id_a in evidence_paper_ids, (
            f"Expected {paper_id_a!r} in evidence_items paper_ids, "
            f"got {evidence_paper_ids!r}"
        )
        assert paper_id_b in evidence_paper_ids, (
            f"Expected {paper_id_b!r} in evidence_items paper_ids, "
            f"got {evidence_paper_ids!r}"
        )

        # ── Step 5: source_sentence matches provenance ─────────────────────
        # Build lookup: paper_id → evidence_item
        ei_by_paper = {ei.paper_id: ei for ei in claim.evidence_items}

        assert ei_by_paper[paper_id_a].source_sentence == sentence_a, (
            f"evidence_items[paper_id_a].source_sentence mismatch: "
            f"expected {sentence_a!r}, got {ei_by_paper[paper_id_a].source_sentence!r}"
        )
        assert ei_by_paper[paper_id_b].source_sentence == sentence_b, (
            f"evidence_items[paper_id_b].source_sentence mismatch: "
            f"expected {sentence_b!r}, got {ei_by_paper[paper_id_b].source_sentence!r}"
        )

        # Bonus: confidence values preserved
        assert ei_by_paper[paper_id_a].confidence == 0.75
        assert ei_by_paper[paper_id_b].confidence == 0.80

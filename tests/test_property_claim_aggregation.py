"""
Property-based tests for TriplePromoter claim aggregation.

Tests Properties 6–10 from the open-world triple promotion design.

Requirements: 2.5, 3.1, 3.2, 3.3, 3.4, 3.5
"""

import uuid
from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graph.evidence_strength_classifier import EvidenceStrengthClassifier
from graph.provenance import ProvenanceMetadata
from graph.triple_promoter import TriplePromoter
from graph.triple_promotion_models import (
    EvidenceItem,
    OpenWorldClaim,
    PaperMetadata,
    PromotedTriple,
)


# ---------------------------------------------------------------------------
# Shared mock helpers
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
    """Mock that returns 'weak' for single classification; uses real logic for claim."""

    def classify_single(self, confidence, section_type, article_type):
        return "weak"

    def classify_claim(self, individual_strengths, paper_count):
        # Use real EvidenceStrengthClassifier logic
        real = EvidenceStrengthClassifier()
        return real.classify_claim(individual_strengths, paper_count)


def _make_promoter() -> TriplePromoter:
    return TriplePromoter(
        entity_normalizer=MockEntityNormalizer(),
        predicate_registry=MockPredicateRegistry(),
        evidence_classifier=MockEvidenceClassifier(),
        promotion_threshold=5,
    )


def _fixed_provenance(paper_id: str, sentence: str = "Evidence sentence.") -> ProvenanceMetadata:
    """Build a minimal valid ProvenanceMetadata for tests."""
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
) -> PromotedTriple:
    """Helper to construct a PromotedTriple for use in aggregation tests."""
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
        provenance=_fixed_provenance(paper_id),
        evidence_strength="weak",
        confidence=confidence,
        paper_id=paper_id,
        section_type="results",
        extracted_at=extracted_at,
    )


def _build_existing_claim(
    supporting_papers: list,
    confidences: list,
    timestamps: list,
    subject_id: str = "NCBI:1",
    predicate: str = "PRODUCES",
    object_id: str = "CHEBI:1",
) -> OpenWorldClaim:
    """Build an OpenWorldClaim directly (bypasses aggregate_claims) for update_claim tests."""
    assert len(supporting_papers) == len(confidences) == len(timestamps)
    evidence_items = [
        EvidenceItem(
            paper_id=pid,
            confidence=conf,
            evidence_strength="weak",
            section_type="results",
            source_sentence="Evidence sentence.",
            extraction_timestamp=ts,
        )
        for pid, conf, ts in zip(supporting_papers, confidences, timestamps)
    ]
    consensus = sum(confidences) / len(confidences) if confidences else 0.0
    return OpenWorldClaim(
        claim_id=str(uuid.uuid4()),
        subject_id=subject_id,
        subject_name="Subject",
        canonical_predicate=predicate,
        object_id=object_id,
        object_name="Object",
        supporting_papers=list(supporting_papers),
        paper_count=len(supporting_papers),
        consensus_confidence=consensus,
        evidence_strength="weak",
        first_reported=min(timestamps) if timestamps else "2024-01-01T00:00:00+00:00",
        last_updated=max(timestamps) if timestamps else "2024-01-01T00:00:00+00:00",
        evidence_items=evidence_items,
    )


# ---------------------------------------------------------------------------
# Property 6: Canonical ID deduplication for claim grouping
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 6: Canonical ID deduplication for claim grouping
class TestProperty6CanonicalIDDeduplication:
    """
    Property 6: Canonical ID deduplication for claim grouping

    Two triples with different surface text but same canonical IDs and same
    canonical_predicate → aggregate_claims returns exactly ONE claim.

    Validates: Requirements 2.5
    """

    @given(
        confidence_a=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
        confidence_b=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
        ts_a=st.datetimes(timezones=st.just(timezone.utc)).map(lambda dt: dt.isoformat()),
        ts_b=st.datetimes(timezones=st.just(timezone.utc)).map(lambda dt: dt.isoformat()),
    )
    @settings(max_examples=50)
    def test_same_canonical_ids_different_surface_text(
        self,
        confidence_a: float,
        confidence_b: float,
        ts_a: str,
        ts_b: str,
    ):
        """
        Two PromotedTriples with same (subject_id, canonical_predicate, object_id)
        but different surface text, from distinct papers → exactly ONE claim with paper_count == 2.

        Validates: Requirements 2.5
        """
        # Two triples: different surface names, same canonical IDs, distinct papers
        triple_a = _build_promoted_triple(
            subject_id="NCBI:1",
            object_id="CHEBI:1",
            predicate="PRODUCES",
            paper_id="paper_001",
            confidence=confidence_a,
            extracted_at=ts_a,
            subject_name="Akkermansia muciniphila",  # different surface text
            object_name="propionate",
        )
        triple_b = _build_promoted_triple(
            subject_id="NCBI:1",
            object_id="CHEBI:1",
            predicate="PRODUCES",
            paper_id="paper_002",
            confidence=confidence_b,
            extracted_at=ts_b,
            subject_name="A. muciniphila",           # different surface text, same canonical
            object_name="propionic acid",             # different surface text, same canonical
        )

        promoter = _make_promoter()
        claims = promoter.aggregate_claims([triple_a, triple_b])

        assert len(claims) == 1, (
            f"Expected 1 claim (canonical ID deduplication), got {len(claims)}"
        )
        assert claims[0].paper_count == 2, (
            f"Expected paper_count == 2, got {claims[0].paper_count}"
        )


# ---------------------------------------------------------------------------
# Property 7: Claim creation from cross-paper triples
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 7: Claim creation from cross-paper triples
class TestProperty7ClaimCreationFromCrossPaperTriples:
    """
    Property 7: Claim creation from cross-paper triples

    N >= 2 triples from distinct papers with same (subject_id, canonical_predicate,
    object_id) → exactly one claim with paper_count == N.

    Validates: Requirements 3.1, 3.3
    """

    @given(
        n=st.integers(min_value=2, max_value=8),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_n_distinct_papers_produce_one_claim(self, n: int, data):
        """
        N triples from N distinct papers sharing same canonical triple key
        → len(claims) == 1 and claims[0].paper_count == N.

        Validates: Requirements 3.1, 3.3
        """
        paper_ids = [f"paper_{i:04d}" for i in range(n)]
        ts_base = "2024-01-01T00:00:00+00:00"

        triples = [
            _build_promoted_triple(
                subject_id="NCBI:1",
                object_id="CHEBI:1",
                predicate="PRODUCES",
                paper_id=paper_ids[i],
                confidence=data.draw(
                    st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
                ),
                extracted_at=ts_base,
            )
            for i in range(n)
        ]

        promoter = _make_promoter()
        claims = promoter.aggregate_claims(triples)

        assert len(claims) == 1, (
            f"Expected exactly 1 claim for {n} distinct papers, got {len(claims)}"
        )
        assert claims[0].paper_count == n, (
            f"Expected paper_count == {n}, got {claims[0].paper_count}"
        )


# ---------------------------------------------------------------------------
# Property 8: Consensus confidence equals arithmetic mean
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 8: Consensus confidence equals arithmetic mean
class TestProperty8ConsensusConfidenceArithmeticMean:
    """
    Property 8: Consensus confidence equals arithmetic mean

    For N evidence items with confidence scores [c1, ..., cN],
    consensus_confidence == sum(c_i) / N within floating-point tolerance 1e-9.

    Validates: Requirements 3.2
    """

    @given(
        confidences=st.lists(
            st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_consensus_confidence_equals_mean(self, confidences: list):
        """
        aggregate_claims consensus_confidence SHALL equal sum(confidences) / N.

        Validates: Requirements 3.2
        """
        n = len(confidences)
        ts_base = "2024-01-01T00:00:00+00:00"

        triples = [
            _build_promoted_triple(
                subject_id="NCBI:1",
                object_id="CHEBI:1",
                predicate="PRODUCES",
                paper_id=f"paper_{i:04d}",
                confidence=confidences[i],
                extracted_at=ts_base,
            )
            for i in range(n)
        ]

        promoter = _make_promoter()
        claims = promoter.aggregate_claims(triples)

        assert len(claims) == 1, f"Expected 1 claim, got {len(claims)}"

        expected_mean = sum(confidences) / n
        actual = claims[0].consensus_confidence

        assert abs(actual - expected_mean) < 1e-9, (
            f"consensus_confidence mismatch: expected {expected_mean}, got {actual}. "
            f"confidences={confidences}"
        )


# ---------------------------------------------------------------------------
# Property 9: Temporal bounds correctness
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 9: Temporal bounds correctness
class TestProperty9TemporalBoundsCorrectness:
    """
    Property 9: Temporal bounds correctness

    For N evidence items with timestamps [t1, ..., tN]:
    first_reported == min(timestamps) and last_updated == max(timestamps).

    Validates: Requirements 3.4
    """

    @given(
        datetimes=st.lists(
            st.datetimes(timezones=st.just(timezone.utc)),
            min_size=2,
            max_size=10,
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_temporal_bounds(self, datetimes: list, data):
        """
        first_reported == min(timestamps) and last_updated == max(timestamps).

        Validates: Requirements 3.4
        """
        timestamps = [dt.isoformat() for dt in datetimes]
        n = len(timestamps)

        triples = [
            _build_promoted_triple(
                subject_id="NCBI:1",
                object_id="CHEBI:1",
                predicate="PRODUCES",
                paper_id=f"paper_{i:04d}",
                confidence=data.draw(
                    st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
                ),
                extracted_at=timestamps[i],
            )
            for i in range(n)
        ]

        promoter = _make_promoter()
        claims = promoter.aggregate_claims(triples)

        assert len(claims) == 1, f"Expected 1 claim, got {len(claims)}"

        expected_min = min(timestamps)
        expected_max = max(timestamps)

        assert claims[0].first_reported == expected_min, (
            f"first_reported mismatch: expected {expected_min!r}, "
            f"got {claims[0].first_reported!r}"
        )
        assert claims[0].last_updated == expected_max, (
            f"last_updated mismatch: expected {expected_max!r}, "
            f"got {claims[0].last_updated!r}"
        )


# ---------------------------------------------------------------------------
# Property 10: Incremental claim update preserves invariants
# ---------------------------------------------------------------------------

# Feature: open-world-triple-promotion, Property 10: Incremental claim update preserves invariants
class TestProperty10IncrementalClaimUpdateInvariants:
    """
    Property 10: Incremental claim update preserves invariants

    Starting from an existing claim (K papers), adding a new triple from a
    new (distinct) paper must:
    - Add new paper_id to supporting_papers
    - Increment paper_count by 1
    - Update consensus_confidence to arithmetic mean of all K+1 confidence scores
    - Set last_updated >= new triple's extracted_at

    Also tests idempotency: adding the SAME paper_id a second time does NOT
    change paper_count or supporting_papers.

    Validates: Requirements 3.5
    """

    @given(
        k=st.integers(min_value=2, max_value=6),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_update_adds_new_paper_and_updates_metrics(self, k: int, data):
        """
        After update_claim with a new paper:
        - new paper_id in supporting_papers
        - paper_count == K + 1
        - consensus_confidence == mean of all K+1 confidence scores
        - last_updated >= new triple's extracted_at

        Validates: Requirements 3.5
        """
        # Build K existing papers with random confidences and timestamps
        existing_paper_ids = [f"paper_{i:04d}" for i in range(k)]
        existing_confidences = [
            data.draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False))
            for _ in range(k)
        ]
        existing_timestamps = [
            data.draw(st.datetimes(timezones=st.just(timezone.utc))).isoformat()
            for _ in range(k)
        ]

        claim = _build_existing_claim(
            supporting_papers=existing_paper_ids,
            confidences=existing_confidences,
            timestamps=existing_timestamps,
        )

        # New triple from a fresh paper
        new_paper_id = f"paper_{k:04d}"  # not in existing_paper_ids
        new_confidence = data.draw(
            st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
        )
        new_ts = data.draw(st.datetimes(timezones=st.just(timezone.utc))).isoformat()

        new_triple = _build_promoted_triple(
            subject_id="NCBI:1",
            object_id="CHEBI:1",
            predicate="PRODUCES",
            paper_id=new_paper_id,
            confidence=new_confidence,
            extracted_at=new_ts,
        )

        promoter = _make_promoter()
        updated = promoter.update_claim(claim, new_triple)

        # (a) New paper_id in supporting_papers
        assert new_paper_id in updated.supporting_papers, (
            f"Expected {new_paper_id!r} in supporting_papers, "
            f"got {updated.supporting_papers}"
        )

        # (b) paper_count == K + 1
        assert updated.paper_count == k + 1, (
            f"Expected paper_count == {k + 1}, got {updated.paper_count}"
        )

        # (c) consensus_confidence == mean of all K+1 confidence scores
        all_confidences = existing_confidences + [new_confidence]
        expected_mean = sum(all_confidences) / len(all_confidences)
        assert abs(updated.consensus_confidence - expected_mean) < 1e-9, (
            f"consensus_confidence mismatch: expected {expected_mean}, "
            f"got {updated.consensus_confidence}. confidences={all_confidences}"
        )

        # (d) last_updated >= new triple's extracted_at
        assert updated.last_updated >= new_ts, (
            f"last_updated {updated.last_updated!r} should be >= {new_ts!r}"
        )

    @given(
        k=st.integers(min_value=2, max_value=6),
        data=st.data(),
    )
    @settings(max_examples=50)
    def test_update_is_idempotent_for_duplicate_paper(self, k: int, data):
        """
        Adding the same paper_id a second time does NOT change paper_count
        or supporting_papers length.

        Validates: Requirements 3.5
        """
        existing_paper_ids = [f"paper_{i:04d}" for i in range(k)]
        existing_confidences = [
            data.draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False))
            for _ in range(k)
        ]
        existing_timestamps = [
            data.draw(st.datetimes(timezones=st.just(timezone.utc))).isoformat()
            for _ in range(k)
        ]

        claim = _build_existing_claim(
            supporting_papers=existing_paper_ids,
            confidences=existing_confidences,
            timestamps=existing_timestamps,
        )

        # Triple using an EXISTING paper_id (duplicate)
        duplicate_paper_id = existing_paper_ids[0]
        duplicate_triple = _build_promoted_triple(
            subject_id="NCBI:1",
            object_id="CHEBI:1",
            predicate="PRODUCES",
            paper_id=duplicate_paper_id,
            confidence=data.draw(
                st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
            ),
            extracted_at=data.draw(
                st.datetimes(timezones=st.just(timezone.utc))
            ).isoformat(),
        )

        promoter = _make_promoter()
        updated = promoter.update_claim(claim, duplicate_triple)

        # paper_count must NOT increase
        assert updated.paper_count == k, (
            f"Expected paper_count to remain {k} after duplicate paper, "
            f"got {updated.paper_count}"
        )

        # supporting_papers length must NOT increase
        assert len(updated.supporting_papers) == k, (
            f"Expected supporting_papers length to remain {k} after duplicate, "
            f"got {len(updated.supporting_papers)}"
        )

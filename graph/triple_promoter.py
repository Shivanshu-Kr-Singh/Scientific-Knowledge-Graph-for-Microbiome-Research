"""
graph/triple_promoter.py
-------------------------
Orchestrates the promotion of LLM-extracted open-world triples to first-class
relationships with full provenance, entity normalization, evidence strength
classification, and claim reification.

The TriplePromoter sits between the LLMTripleExtractor output and the Neo4j
loading step. It enriches each raw triple with:
1. Full provenance metadata (source sentence, section, extraction context)
2. Entity normalization (via EntityNormalizer)
3. Evidence strength classification (strong/moderate/weak)
4. Predicate normalization and frequency tracking
5. Claim aggregation into OpenWorldClaim nodes

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5,
             3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.4, 6.1, 6.4
"""

import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from graph.provenance import ProvenanceMetadata
from graph.triple_promotion_models import (
    EvidenceItem,
    OpenWorldClaim,
    PaperMetadata,
    PromotedTriple,
)


# Allowed section types recognized by ProvenanceMetadata
_ALLOWED_SECTION_TYPES = {
    "abstract", "methods", "results", "discussion", "introduction",
    "background", "conclusion", "study_population", "bioinformatics",
    "statistical_analysis", "limitations", "strengths", "future_directions",
    "data_availability", "supplementary", "ethics", "trial_registration",
    "conflict_of_interest", "funding", "acknowledgements", "references",
    "glossary", "other",
}

# Sentence boundary pattern for splitting text into sentences
_SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.?!])\s+')


class TriplePromoter:
    """
    Orchestrates the promotion of LLM-extracted triples to first-class
    relationships with full provenance, normalization, and reification.

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """

    def __init__(
        self,
        entity_normalizer,
        predicate_registry,
        evidence_classifier,
        promotion_threshold: int = 5,
    ) -> None:
        """
        Initialize the TriplePromoter.

        Args:
            entity_normalizer: EntityNormalizer instance for grounding entities
            predicate_registry: PredicateRegistry instance for predicate normalization
            evidence_classifier: EvidenceStrengthClassifier instance
            promotion_threshold: Minimum distinct papers for automatic predicate
                promotion (default: 5)
        """
        self.entity_normalizer = entity_normalizer
        self.predicate_registry = predicate_registry
        self.evidence_classifier = evidence_classifier
        self.promotion_threshold = promotion_threshold

    def promote_triple(
        self,
        raw_triple: Dict[str, Any],
        paper_metadata: PaperMetadata,
    ) -> Optional[PromotedTriple]:
        """
        Promote a single raw LLM triple to a fully enriched relationship.

        Returns None if the triple fails quality gates (empty evidence,
        confidence < 0.5).

        This method implements:
        - Quality gate (Requirement 1.5)
        - Provenance attachment (Requirements 1.1, 1.4)
        - Sentence offset calculation (Requirement 1.2)
        - Surrounding context extraction (Requirement 1.3)

        Args:
            raw_triple: Dict from LLMTripleExtractor with keys:
                subject, subject_type, predicate, object, object_type,
                confidence, evidence, paper_id, section_type, extracted_at,
                section_content (optional)
            paper_metadata: Paper-level metadata (article_type, publication_year)

        Returns:
            PromotedTriple with provenance, normalized entities, and
            evidence strength, or None if rejected.
        """
        # ─── Quality Gate (Requirement 1.5) ───────────────────────────────
        evidence = raw_triple.get("evidence", "")
        if not evidence or not evidence.strip():
            logger.debug(
                "Quality gate: rejected triple with empty evidence "
                "(paper_id={})",
                raw_triple.get("paper_id", "unknown"),
            )
            return None

        confidence = raw_triple.get("confidence", 0)
        if confidence < 0.5:
            logger.debug(
                "Quality gate: rejected triple with confidence {:.3f} < 0.5 "
                "(paper_id={})",
                confidence,
                raw_triple.get("paper_id", "unknown"),
            )
            return None

        # ─── Provenance Attachment (Requirements 1.1, 1.4) ───────────────
        paper_id = raw_triple["paper_id"]
        section_type = self._normalize_section_type(raw_triple.get("section_type", "other"))
        source_sentence = evidence.strip()

        # Sentence offset (Requirement 1.2)
        sentence_offset = self._calculate_sentence_offset(
            raw_triple.get("section_content"), source_sentence
        )

        # Surrounding context (Requirement 1.3)
        surrounding_context = self._extract_surrounding_context(
            raw_triple.get("section_content"), source_sentence
        )

        # Extractor version: Ollama model name from environment
        extractor_version = os.getenv("OLLAMA_MODEL", "llama3.1")

        provenance = ProvenanceMetadata(
            paper_id=paper_id,
            section_type=section_type,
            source_sentence=source_sentence,
            sentence_offset=sentence_offset,
            extraction_method="llm_triple_extractor",
            extraction_timestamp=datetime.now(timezone.utc),
            extractor_version=extractor_version,
            confidence_score=confidence,
            validation_status="unvalidated",
            surrounding_context=surrounding_context,
        )

        # ─── Entity Normalization (Requirements 2.1, 2.2, 2.3, 2.4) ────
        subject_text = raw_triple.get("subject", "")
        subject_type = raw_triple.get("subject_type", "unknown")
        object_text = raw_triple.get("object", "")
        object_type = raw_triple.get("object_type", "unknown")

        # Normalize subject via EntityNormalizer with graceful degradation
        subject_grounded = False
        subject_ontology = None
        subject_name = subject_text
        subject_id = f"ungrounded:{subject_text.lower()}"

        try:
            subject_result = self.entity_normalizer.normalize(subject_text, subject_type)
            if subject_result.get("grounded", False):
                subject_id = subject_result["id"]
                subject_name = subject_result.get("canonical_name", subject_text)
                subject_ontology = subject_result.get("ontology")
                subject_grounded = True
            else:
                subject_id = f"ungrounded:{subject_text.lower()}"
                subject_name = subject_text
        except Exception as exc:
            logger.warning(
                "EntityNormalizer failed for subject {!r}: {}. Using ungrounded fallback.",
                subject_text, exc,
            )

        # Normalize object via EntityNormalizer with graceful degradation
        object_grounded = False
        object_ontology = None
        object_name = object_text
        object_id = f"ungrounded:{object_text.lower()}"

        try:
            object_result = self.entity_normalizer.normalize(object_text, object_type)
            if object_result.get("grounded", False):
                object_id = object_result["id"]
                object_name = object_result.get("canonical_name", object_text)
                object_ontology = object_result.get("ontology")
                object_grounded = True
            else:
                object_id = f"ungrounded:{object_text.lower()}"
                object_name = object_text
        except Exception as exc:
            logger.warning(
                "EntityNormalizer failed for object {!r}: {}. Using ungrounded fallback.",
                object_text, exc,
            )

        # ─── Predicate Normalization (Requirements 2.1, 2.2, 4.1) ──────────
        raw_predicate = raw_triple.get("predicate", "")
        # Track paper occurrence and check for promotion
        canonical_predicate, is_known, is_newly_promoted = self.predicate_registry.track_paper_occurrence(
            raw_predicate, paper_id
        )
        # If newly promoted, get the actual canonical form from promote_predicate
        if is_newly_promoted:
            canonical_predicate = self.predicate_registry.promote_predicate(raw_predicate)
        is_novel = not is_known
        predicate_category = self.predicate_registry.get_category(canonical_predicate)
        relationship_type = canonical_predicate if is_known or is_newly_promoted else "RELATES_TO"

        # ─── Evidence Strength (Requirements 6.1, 6.4) ───────────────────
        evidence_strength = self.evidence_classifier.classify_single(
            confidence, section_type, paper_metadata.article_type
        )

        # ─── Build PromotedTriple ─────────────────────────────────────────
        promoted = PromotedTriple(
            subject_id=subject_id,
            subject_name=subject_name,
            subject_type=subject_type,
            subject_grounded=subject_grounded,
            subject_ontology=subject_ontology,
            object_id=object_id,
            object_name=object_name,
            object_type=object_type,
            object_grounded=object_grounded,
            object_ontology=object_ontology,
            raw_predicate=raw_predicate,
            canonical_predicate=canonical_predicate,
            predicate_category=predicate_category,
            is_novel_predicate=is_novel,
            relationship_type=relationship_type,
            provenance=provenance,
            evidence_strength=evidence_strength,
            confidence=confidence,
            paper_id=paper_id,
            section_type=section_type,
            extracted_at=raw_triple.get("extracted_at", datetime.now(timezone.utc).isoformat()),
        )

        return promoted

    def promote_batch(
        self,
        raw_triples: List[Dict[str, Any]],
        paper_metadata: PaperMetadata,
    ) -> List[PromotedTriple]:
        """
        Promote a batch of triples from a single paper.

        Iterates over raw_triples, calls promote_triple for each, and
        collects non-None results. Triples that fail quality gates are
        silently skipped (promote_triple returns None for them).

        Args:
            raw_triples: List of raw triple dicts from LLMTripleExtractor
            paper_metadata: Paper-level metadata shared by all triples in the batch

        Returns:
            List of successfully promoted PromotedTriple objects
        """
        results: List[PromotedTriple] = []
        for triple in raw_triples:
            promoted = self.promote_triple(triple, paper_metadata)
            if promoted is not None:
                results.append(promoted)
        return results

    def aggregate_claims(
        self,
        promoted_triples: List[PromotedTriple],
    ) -> List[OpenWorldClaim]:
        """
        Group promoted triples by (subject_id, canonical_predicate, object_id).
        For groups with >= 2 distinct papers: create OpenWorldClaim.

        Returns list of OpenWorldClaim objects.

        Requirements: 3.1, 3.2, 3.3, 3.4, 6.5, 6.6
        """
        # Step 1: Group triples by (subject_id, canonical_predicate, object_id)
        groups: Dict[tuple, List[PromotedTriple]] = defaultdict(list)
        for triple in promoted_triples:
            key = (triple.subject_id, triple.canonical_predicate, triple.object_id)
            groups[key].append(triple)

        claims: List[OpenWorldClaim] = []

        for (subject_id, canonical_predicate, object_id), group_triples in groups.items():
            # Step 2: Collect unique paper_ids (preserving first-seen order)
            seen_papers: set = set()
            supporting_papers: List[str] = []
            for t in group_triples:
                if t.paper_id not in seen_papers:
                    seen_papers.add(t.paper_id)
                    supporting_papers.append(t.paper_id)

            # Only create a claim if >= 2 distinct papers
            if len(supporting_papers) < 2:
                continue

            paper_count = len(supporting_papers)

            # Step 3: consensus_confidence = arithmetic mean of all confidence scores
            confidence_scores = [t.confidence for t in group_triples]
            consensus_confidence = sum(confidence_scores) / len(confidence_scores)

            # Step 4: evidence_strength via classify_claim
            individual_strengths = [t.evidence_strength for t in group_triples]
            evidence_strength = self.evidence_classifier.classify_claim(
                individual_strengths, paper_count
            )

            # Step 5: temporal bounds
            first_reported = min(t.extracted_at for t in group_triples)
            last_updated = max(t.extracted_at for t in group_triples)

            # Step 6: build EvidenceItem list from each triple's provenance
            evidence_items = [
                EvidenceItem(
                    paper_id=t.paper_id,
                    confidence=t.confidence,
                    evidence_strength=t.evidence_strength,
                    section_type=t.section_type,
                    source_sentence=t.provenance.source_sentence,
                    extraction_timestamp=t.extracted_at,
                )
                for t in group_triples
            ]

            # Step 7: use first triple's names for the claim
            first = group_triples[0]

            claim = OpenWorldClaim(
                claim_id=str(uuid.uuid4()),
                subject_id=subject_id,
                subject_name=first.subject_name,
                canonical_predicate=canonical_predicate,
                object_id=object_id,
                object_name=first.object_name,
                supporting_papers=supporting_papers,
                paper_count=paper_count,
                consensus_confidence=consensus_confidence,
                evidence_strength=evidence_strength,
                first_reported=first_reported,
                last_updated=last_updated,
                evidence_items=evidence_items,
            )
            claims.append(claim)

        return claims

    def update_claim(
        self,
        claim: OpenWorldClaim,
        new_triple: PromotedTriple,
    ) -> OpenWorldClaim:
        """
        Update an existing OpenWorldClaim with a new matching triple.

        Idempotent: if new_triple.paper_id is already in claim.supporting_papers,
        the supporting_papers list and paper_count are NOT changed, and no new
        EvidenceItem is appended.

        Logic:
        1. If new_triple.paper_id NOT in claim.supporting_papers:
           - Append paper_id to supporting_papers
           - Increment paper_count
           - Append a new EvidenceItem built from new_triple
        2. Recalculate consensus_confidence = mean of all evidence_items.confidence
        3. Update last_updated = max(claim.last_updated, new_triple.extracted_at)
        4. Update evidence_strength via classify_claim
        5. Return the (modified-in-place) claim

        Requirements: 3.5
        """
        if new_triple.paper_id not in claim.supporting_papers:
            # Add paper to supporting list
            claim.supporting_papers.append(new_triple.paper_id)
            claim.paper_count += 1

            # Add new evidence item
            evidence_item = EvidenceItem(
                paper_id=new_triple.paper_id,
                confidence=new_triple.confidence,
                evidence_strength=new_triple.evidence_strength,
                section_type=new_triple.section_type,
                source_sentence=new_triple.provenance.source_sentence,
                extraction_timestamp=new_triple.extracted_at,
            )
            claim.evidence_items.append(evidence_item)

        # Recalculate consensus_confidence from all current evidence items
        if claim.evidence_items:
            claim.consensus_confidence = (
                sum(item.confidence for item in claim.evidence_items)
                / len(claim.evidence_items)
            )

        # Update last_updated = max of current last_updated and new triple's extracted_at
        if new_triple.extracted_at > claim.last_updated:
            claim.last_updated = new_triple.extracted_at

        # Update evidence_strength via classify_claim
        individual_strengths = [item.evidence_strength for item in claim.evidence_items]
        claim.evidence_strength = self.evidence_classifier.classify_claim(
            individual_strengths, claim.paper_count
        )

        return claim

    def check_predicate_promotion(self) -> List[str]:
        """
        Check if any novel predicates have reached the promotion threshold.

        Queries PredicateRegistry for novel predicates with paper count >= threshold
        that haven't been promoted yet. Promotes each one.

        Returns list of newly promoted canonical predicate names.

        Requirements: 4.4, 4.6
        """
        threshold = self.predicate_registry.get_promotion_threshold()
        candidates = self.predicate_registry.get_novel_predicates(min_frequency=threshold)

        newly_promoted: List[str] = []
        for predicate in candidates:
            canonical = self.predicate_registry.promote_predicate(predicate["raw_predicate"])
            newly_promoted.append(canonical)

        return newly_promoted

    def retroactive_promote(self, canonical_predicate: str, neo4j_driver=None) -> int:
        """
        Retroactively update existing RELATES_TO edges in Neo4j that match
        the promoted predicate to use the new relationship type.

        If Neo4j is unavailable (neo4j_driver is None), logs a warning and
        returns 0. Otherwise counts and returns the number of matching edges.

        Note: Full relationship-type migration (delete RELATES_TO + create new
        typed edge) is deferred to pipeline integration. This implementation
        counts affected edges to confirm scope.

        Args:
            canonical_predicate: The promoted canonical predicate name to search for.
            neo4j_driver: Optional Neo4j driver instance. If None, skips Neo4j.

        Returns:
            Count of edges that match the promoted predicate (0 if driver absent).

        Requirements: 4.4, 4.6
        """
        if neo4j_driver is None:
            logger.warning("retroactive_promote: no Neo4j driver provided, skipping")
            return 0

        try:
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (s)-[r:RELATES_TO]->(o)
                    WHERE r.raw_predicate = $pred
                    RETURN count(r) as count
                    """,
                    pred=canonical_predicate.lower(),
                )
                return result.single()["count"]
        except Exception as exc:
            logger.warning("retroactive_promote failed: {}", exc)
            return 0

    # ─── Private Helpers ──────────────────────────────────────────────────

    def _normalize_section_type(self, section_type: str) -> str:
        """
        Normalize section_type to an allowed value for ProvenanceMetadata.

        If the raw section_type is not in the allowed set, returns "other".

        Args:
            section_type: Raw section type string from the LLM triple

        Returns:
            Normalized section type string
        """
        normalized = section_type.lower().strip()
        if normalized in _ALLOWED_SECTION_TYPES:
            return normalized
        return "other"

    def _calculate_sentence_offset(
        self, section_content: Optional[str], evidence: str
    ) -> Optional[int]:
        """
        Calculate the character position of the evidence sentence within section content.

        Requirement 1.2: Populate sentence_offset with the character position of
        the evidence sentence within the section content.

        Args:
            section_content: Full text of the section (may be None)
            evidence: The evidence sentence to locate

        Returns:
            Character offset (>= 0) if found, None otherwise
        """
        if not section_content or not evidence:
            return None

        offset = section_content.find(evidence)
        if offset >= 0:
            return offset
        return None

    def _extract_surrounding_context(
        self, section_content: Optional[str], evidence: str
    ) -> Optional[str]:
        """
        Extract ±2 sentences around the evidence sentence from section content.

        Requirement 1.3: Populate surrounding_context with ±2 sentences around
        the evidence sentence.

        Algorithm:
        1. Split section_content into sentences using sentence boundary pattern
        2. Find the index of the sentence containing the evidence
        3. Extract sentences from max(0, idx-2) to min(len, idx+3) (exclusive end)
        4. Join back as surrounding_context

        Args:
            section_content: Full text of the section (may be None)
            evidence: The evidence sentence to find context around

        Returns:
            Surrounding context string, or None if section_content is unavailable
            or evidence not found
        """
        if not section_content or not evidence:
            return None

        # Split into sentences
        sentences = _SENTENCE_SPLIT_PATTERN.split(section_content)

        # Find the sentence that contains (or matches) the evidence
        evidence_idx = None
        for i, sent in enumerate(sentences):
            if evidence in sent or sent in evidence:
                evidence_idx = i
                break

        if evidence_idx is None:
            return None

        # Extract ±2 sentences
        start = max(0, evidence_idx - 2)
        end = min(len(sentences), evidence_idx + 3)
        context_sentences = sentences[start:end]

        return " ".join(context_sentences)

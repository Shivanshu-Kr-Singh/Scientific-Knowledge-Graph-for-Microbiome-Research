"""
Unit tests for _extract_direction() returning "associated" for
non-directional association language.

Validates: Requirements 2.1, 7.2
"""

import pytest
from graph.semantic_extractor import SemanticRelationshipExtractor


@pytest.fixture
def extractor():
    return SemanticRelationshipExtractor()


ASSOCIATED_SENTENCES = [
    ("Bacteroides fragilis was implicated in the pathogenesis of IBD",),
    ("Faecalibacterium prausnitzii plays a protective role in gut inflammation",),
    ("Akkermansia muciniphila is a biomarker for metabolic health",),
    ("Fusobacterium nucleatum is a pathobiont associated with colorectal cancer",),
    ("Gut dysbiosis was observed in patients with depression",),
    ("This species is linked to cardiovascular disease",),
    ("Prevotella copri is a contributor to rheumatoid arthritis",),
    ("Ruminococcus gnavus is a marker of Crohn's disease activity",),
    ("A distinct microbial signature of inflammatory bowel disease was identified",),
    ("Clostridium difficile is involved in post-antibiotic complications",),
    ("The abundance of Lachnospiraceae correlated with disease severity",),
    ("These organisms are related to allergic responses",),
    ("Enterococcus faecalis is pathogenic in immunocompromised patients",),
    ("Bifidobacterium is a commensal organism in healthy infants",),
    ("This mutualistic relationship supports host immunity",),
    ("The symbiont Wolbachia modulates host fitness",),
    ("Bacteroides was the dominant genus in the disease group",),
    ("Firmicutes were identified as a key species in the microbiome",),
    ("Community composition differed significantly between groups",),
]


@pytest.mark.parametrize("sentence", ASSOCIATED_SENTENCES, ids=[
    "implicated_in_pathogenesis",
    "protective_role",
    "biomarker_for",
    "pathobiont_associated",
    "dysbiosis_observed",
    "linked_to",
    "contributor_to",
    "marker_of",
    "signature_of",
    "involved_in",
    "correlated_with",
    "related_to",
    "pathogenic",
    "commensal",
    "mutualistic",
    "symbiont",
    "dominant_genus",
    "key_species",
    "differed_significantly",
])
def test_extract_direction_associated_examples(extractor, sentence):
    """
    Verify _extract_direction() returns 'associated' for each example sentence
    containing non-directional association language from the feature request.

    These sentences contain association signal patterns but no directional
    (increased/decreased/no_change) keywords, so the expected result is 'associated'.
    """
    result = extractor._extract_direction(sentence[0])
    assert result == "associated", (
        f"Expected 'associated' for sentence: {sentence[0]!r}, got {result!r}"
    )


# --- Property-Based Tests (Hypothesis) ---

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


DIRECTIONAL_KEYWORDS = [
    "increased", "higher", "elevated", "enriched",
    "decreased", "lower", "reduced", "depleted",
    "no significant change", "unchanged",
]

ASSOCIATION_KEYWORDS = [
    "implicated", "role in", "biomarker", "pathobiont",
    "dysbiosis", "protective", "linked to", "commensal",
    "mutualistic", "symbiont",
]


@given(
    directional=st.sampled_from(DIRECTIONAL_KEYWORDS),
    association=st.sampled_from(ASSOCIATION_KEYWORDS),
    filler=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'Zs')),
        min_size=1,
        max_size=30,
    ),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_direction_priority_over_association(extractor, directional, association, filler):
    """
    Property 1: Directional patterns always take priority over association patterns.

    Validates: Requirements 2.2, 2.4

    For any sentence containing both a directional keyword and an association
    signal keyword, _extract_direction() must return the directional value
    (increased, decreased, or no_change), never "associated".
    """
    sentence = f"The taxon was {directional} and {association} in {filler} disease"
    result = extractor._extract_direction(sentence)
    assert result in {"increased", "decreased", "no_change"}, (
        f"Expected directional result for sentence with both signals, got {result!r}"
    )


# --- Property 2: Association Signal Detection ---

SAFE_FILLERS = [
    "the taxon",
    "this species",
    "a gut organism",
    "the bacterium",
    "these microbes",
    "the patients",
    "this disease",
    "gut flora",
    "the community",
    "these organisms",
]


@given(
    association=st.sampled_from(ASSOCIATION_KEYWORDS),
    filler=st.sampled_from(SAFE_FILLERS),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_association_signal_returns_associated(extractor, association, filler):
    """
    Property 2: Sentences with only association signal keywords return 'associated'.

    Validates: Requirements 2.1, 7.2

    For any sentence containing an association signal keyword and NO directional
    keywords, _extract_direction() must return "associated".
    """
    sentence = f"{filler} was {association} the condition"
    result = extractor._extract_direction(sentence)
    assert result == "associated", (
        f"Expected 'associated' for sentence with only association signal, got {result!r}. "
        f"Sentence: {sentence!r}"
    )


# --- Property 3: Confidence Equivalence ---


@given(
    p_value=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    effect_size=st.one_of(st.none(), st.floats(min_value=0.0, max_value=10.0)),
    statistical_measure=st.one_of(st.none(), st.sampled_from(["LDA score", "fold change", "relative abundance"])),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_confidence_equivalence_associated_vs_increased(extractor, p_value, effect_size, statistical_measure):
    """
    Property 3: Confidence is the same for 'associated' and 'increased' given same inputs.

    Validates: Requirements 3.2

    The confidence calculation depends only on whether direction is truthy, not on
    the specific direction value. Both "associated" and "increased" are truthy strings,
    so they must produce identical confidence scores.
    """
    conf_associated = extractor._calculate_association_confidence("associated", p_value, effect_size, statistical_measure)
    conf_increased = extractor._calculate_association_confidence("increased", p_value, effect_size, statistical_measure)
    assert conf_associated == conf_increased, (
        f"Confidence differs: associated={conf_associated}, increased={conf_increased}"
    )


# --- Property 4: Non-Conflict Invariant ---

from graph.relationship_reifier import RelationshipReifier
from graph.reified_claims import ScientificClaim, EvidenceStrength


@pytest.fixture
def reifier():
    return RelationshipReifier()


PREDICATES_WITH_ASSOCIATED = [
    "associated_with_associated",
    "associated_with_increased",
    "associated_with_decreased",
    "associated_with_no_change",
]

OTHER_PREDICATES = [
    "associated_with_increased",
    "associated_with_decreased",
    "associated_with_no_change",
    "associated_with_associated",
]


def _make_claim(predicate, claim_id="test-claim-1", subject="TaxonA", obj="DiseaseA"):
    return ScientificClaim(
        claim_id=claim_id,
        claim_type="association",
        subject_entity=subject,
        predicate=predicate,
        object_entity=obj,
        supporting_papers=["PMID:12345"],
        evidence_strength=EvidenceStrength.MODERATE,
        consensus_confidence=0.8,
        effect_direction_consistency=0.9,
        first_reported="2024-01-01",
        last_updated="2024-06-01",
    )


@given(
    pred1=st.sampled_from(PREDICATES_WITH_ASSOCIATED),
    pred2=st.sampled_from(OTHER_PREDICATES),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_associated_predicates_never_conflict(reifier, pred1, pred2):
    """
    Property 4: Claims with 'associated' in predicate never conflict.

    Validates: Requirements 4.2

    For any pair of claims where at least one has "associated" in its predicate,
    detect_conflicting_claims() must never flag them as conflicting.
    """
    claim1 = _make_claim(pred1, claim_id="test-claim-1", subject="TaxonA", obj="DiseaseA")
    claim2 = _make_claim(pred2, claim_id="test-claim-2", subject="TaxonA", obj="DiseaseA")
    conflicts = reifier.detect_conflicting_claims([claim1, claim2])
    assert len(conflicts) == 0, (
        f"Expected no conflicts when 'associated' is in predicate, got {conflicts}. "
        f"pred1={pred1!r}, pred2={pred2!r}"
    )


# --- Property 5: Valid Direction Set ---

from graph.semantic_relationships import create_association_relationship, SemanticRelationship, RelationType
from graph.provenance import ProvenanceMetadata
from pydantic import ValidationError
from datetime import datetime, timezone

VALID_DIRECTIONS = ["increased", "decreased", "no_change", "associated"]


def _make_provenance():
    """Create a valid ProvenanceMetadata instance for testing."""
    return ProvenanceMetadata(
        paper_id="PMID:12345",
        section_type="results",
        source_sentence="Test sentence for association extraction.",
        extraction_method="regex_ner",
        extraction_timestamp=datetime.now(timezone.utc),
        extractor_version="1.0",
        confidence_score=0.85,
    )


@pytest.mark.parametrize("direction", VALID_DIRECTIONS)
def test_valid_directions_pass_validation(direction):
    """
    Property 5a: All four valid direction values pass validation.

    Validates: Requirements 1.3

    For any direction in {"increased", "decreased", "no_change", "associated"},
    create_association_relationship() succeeds and stores the direction correctly.
    """
    rel = create_association_relationship(
        source_entity="paper1",
        target_entity="taxon1",
        direction=direction,
        comparison="disease vs healthy",
        statistical_measure="relative abundance",
        provenance=_make_provenance(),
        evidence_strength="moderate",
        extraction_confidence=0.8,
    )
    assert rel.properties["direction"] == direction
    assert rel.relation_type == RelationType.REPORTS_ASSOCIATION


@given(
    invalid_direction=st.text(min_size=1, max_size=20).filter(
        lambda x: x not in {"increased", "decreased", "no_change", "associated"}
    )
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_invalid_directions_raise_validation_error(invalid_direction):
    """
    Property 5b: Invalid direction values raise ValueError.

    Validates: Requirements 1.3

    For any direction NOT in {"increased", "decreased", "no_change", "associated"},
    create_association_relationship() raises ValueError during validation.
    """
    with pytest.raises((ValueError, ValidationError)):
        create_association_relationship(
            source_entity="paper1",
            target_entity="taxon1",
            direction=invalid_direction,
            comparison="disease vs healthy",
            statistical_measure="relative abundance",
            provenance=_make_provenance(),
            evidence_strength="moderate",
            extraction_confidence=0.8,
        )


# --- Integration Test: EnrichedPaperRecord → EnhancedGraphEdge with direction="associated" ---

from nlp.enriched_record import EnrichedPaperRecord, ParsedSection
from graph.enhanced_graph_builder import EnhancedGraphBuilder, EnhancedGraphEdge


def test_integration_association_only_language_produces_associated_edges():
    """
    Integration test: An EnrichedPaperRecord with association-only language
    produces EnhancedGraphEdge objects with direction="associated".

    Validates: Requirements 2.1, 7.2
    """
    paper = EnrichedPaperRecord(
        doi="10.1234/test.2024",
        title="Association of Bacteroides fragilis with IBD",
        abstract="Bacteroides fragilis was implicated in the pathogenesis of IBD.",
        taxa=["Bacteroides fragilis"],
        diseases=["IBD"],
        sections=[
            ParsedSection(
                section_type="results",
                content="Bacteroides fragilis was implicated in the pathogenesis of IBD. This pathobiont plays a role in intestinal inflammation."
            )
        ],
        article_type_normalized="original_research",
    )

    builder = EnhancedGraphBuilder()
    edges = builder.process_paper(paper)

    # At least one edge should have direction="associated"
    associated_edges = [e for e in edges if e.properties.get("direction") == "associated"]
    assert len(associated_edges) > 0, (
        f"Expected at least one edge with direction='associated', got directions: "
        f"{[e.properties.get('direction') for e in edges]}"
    )

    # Verify edge structure
    edge = associated_edges[0]
    assert edge.relation == "REPORTS_ASSOCIATION"
    assert edge.confidence >= 0.5

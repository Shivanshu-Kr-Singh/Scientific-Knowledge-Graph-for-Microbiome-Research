"""
graph/test_reified_claims_properties.py
----------------------------------------
Property-based tests for reified claim consistency.

Tests universal properties that should hold for all ScientificClaim and
ReifiedClaimNode instances.

**Validates: Requirements 4.2, 4.3**
"""

import pytest
from datetime import datetime, timezone, timedelta
from hypothesis import given, strategies as st, settings, assume
from pydantic import ValidationError

from graph.reified_claims import (
    EvidenceStrength,
    ScientificClaim,
    ReifiedClaimNode,
)


# ============================================================================
# Hypothesis Strategies for Generating Test Data
# ============================================================================

# Strategy for valid claim types
claim_type_strategy = st.sampled_from([
    "association", "intervention_effect", "methodology_comparison"
])

# Strategy for valid evidence strengths
evidence_strength_strategy = st.sampled_from([
    EvidenceStrength.STRONG,
    EvidenceStrength.MODERATE,
    EvidenceStrength.WEAK,
    EvidenceStrength.CONFLICTING,
])

# Strategy for evidence strength strings (for ReifiedClaimNode)
evidence_strength_string_strategy = st.sampled_from([
    "strong", "moderate", "weak", "conflicting"
])

# Strategy for consensus metrics in valid range [0.0, 1.0]
consensus_metric_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False
)

# Strategy for non-empty strings (for entity names, predicates, etc.)
non_empty_string_strategy = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())

# Strategy for paper ID lists (unique, non-overlapping)
def paper_id_list_strategy(min_size=0, max_size=10):
    """Generate a list of unique paper IDs."""
    return st.lists(
        st.text(min_size=5, max_size=20, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='-_'
        )).filter(lambda s: s.strip()),
        min_size=min_size,
        max_size=max_size,
        unique=True
    )

# Strategy for non-negative integers (sample sizes)
non_negative_int_strategy = st.integers(min_value=0, max_value=10000)

# Strategy for ISO date strings
def iso_date_strategy():
    """Generate ISO date strings."""
    return st.dates(
        min_value=datetime(2020, 1, 1).date(),
        max_value=datetime.now().date()
    ).map(lambda d: d.isoformat())

# Strategy for datetime objects
def datetime_strategy():
    """Generate datetime objects with timezone."""
    return st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime.now(),
        timezones=st.just(timezone.utc)
    )

# Strategy for optional floats (effect sizes, variances)
optional_float_strategy = st.one_of(
    st.none(),
    st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False
    )
)


# ============================================================================
# Property 2: Reified Claim Consistency
# **Validates: Requirements 4.2, 4.3**
# ============================================================================

@given(
    claim_id=non_empty_string_strategy,
    claim_type=claim_type_strategy,
    subject_entity=non_empty_string_strategy,
    predicate=non_empty_string_strategy,
    object_entity=non_empty_string_strategy,
    supporting_papers=paper_id_list_strategy(min_size=0, max_size=10),
    contradicting_papers=paper_id_list_strategy(min_size=0, max_size=10),
    total_sample_size=non_negative_int_strategy,
    evidence_strength=evidence_strength_strategy,
    consensus_confidence=consensus_metric_strategy,
    effect_direction_consistency=consensus_metric_strategy,
    first_reported=iso_date_strategy(),
    last_updated=iso_date_strategy(),
    pooled_effect_size=optional_float_strategy,
    effect_size_variance=st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
)
@settings(max_examples=100, deadline=None)
def test_property_reified_claim_consistency_scientific_claim(
    claim_id,
    claim_type,
    subject_entity,
    predicate,
    object_entity,
    supporting_papers,
    contradicting_papers,
    total_sample_size,
    evidence_strength,
    consensus_confidence,
    effect_direction_consistency,
    first_reported,
    last_updated,
    pooled_effect_size,
    effect_size_variance,
):
    """
    **Property 2: Reified Claim Consistency**
    **Validates: Requirements 4.2, 4.3**
    
    Test that ScientificClaim instances maintain consistency invariants:
    1. supporting_papers and contradicting_papers have no overlap
    2. consensus_confidence is always in range [0.0, 1.0]
    3. first_reported <= last_updated
    
    Universal Property:
    - For all valid inputs with non-overlapping paper lists and valid temporal ordering,
      ScientificClaim creation succeeds
    - All consistency invariants are maintained
    """
    # Ensure no overlap between supporting and contradicting papers
    supporting_set = set(supporting_papers)
    contradicting_set = set(contradicting_papers)
    assume(len(supporting_set & contradicting_set) == 0)
    
    # Ensure temporal ordering is valid
    first_date = datetime.fromisoformat(first_reported)
    last_date = datetime.fromisoformat(last_updated)
    assume(first_date <= last_date)
    
    # Create ScientificClaim
    claim = ScientificClaim(
        claim_id=claim_id,
        claim_type=claim_type,
        subject_entity=subject_entity,
        predicate=predicate,
        object_entity=object_entity,
        supporting_papers=supporting_papers,
        contradicting_papers=contradicting_papers,
        total_sample_size=total_sample_size,
        evidence_strength=evidence_strength,
        consensus_confidence=consensus_confidence,
        effect_direction_consistency=effect_direction_consistency,
        first_reported=first_reported,
        last_updated=last_updated,
        pooled_effect_size=pooled_effect_size,
        effect_size_variance=effect_size_variance,
    )
    
    # Property 2a: No overlap between supporting and contradicting papers (Requirement 4.2)
    supporting_set_result = set(claim.supporting_papers)
    contradicting_set_result = set(claim.contradicting_papers)
    assert len(supporting_set_result & contradicting_set_result) == 0, \
        "supporting_papers and contradicting_papers must not overlap"
    
    # Property 2b: consensus_confidence is in range [0.0, 1.0] (Requirement 4.3)
    assert 0.0 <= claim.consensus_confidence <= 1.0, \
        "consensus_confidence must be in range [0.0, 1.0]"
    
    # Property 2c: effect_direction_consistency is in range [0.0, 1.0] (Requirement 4.3)
    assert 0.0 <= claim.effect_direction_consistency <= 1.0, \
        "effect_direction_consistency must be in range [0.0, 1.0]"
    
    # Property 2d: first_reported <= last_updated (Requirement 4.5)
    first_parsed = datetime.fromisoformat(claim.first_reported)
    last_parsed = datetime.fromisoformat(claim.last_updated)
    assert first_parsed <= last_parsed, \
        "first_reported must be <= last_updated"
    
    # Property 2e: Paper lists contain no duplicates (Requirement 4.2)
    assert len(claim.supporting_papers) == len(set(claim.supporting_papers)), \
        "supporting_papers must not contain duplicates"
    assert len(claim.contradicting_papers) == len(set(claim.contradicting_papers)), \
        "contradicting_papers must not contain duplicates"


@given(
    node_id=non_empty_string_strategy,
    claim_type=claim_type_strategy,
    subject_entity=non_empty_string_strategy,
    predicate=non_empty_string_strategy,
    object_entity=non_empty_string_strategy,
    supporting_paper_ids=paper_id_list_strategy(min_size=0, max_size=10),
    contradicting_paper_ids=paper_id_list_strategy(min_size=0, max_size=10),
    total_sample_size=non_negative_int_strategy,
    evidence_strength=evidence_strength_string_strategy,
    consensus_confidence=consensus_metric_strategy,
    effect_direction_consistency=consensus_metric_strategy,
    first_reported=datetime_strategy(),
    last_updated=datetime_strategy(),
    pooled_effect_size=optional_float_strategy,
    effect_size_variance=st.one_of(st.none(), st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
)
@settings(max_examples=100, deadline=None)
def test_property_reified_claim_consistency_reified_claim_node(
    node_id,
    claim_type,
    subject_entity,
    predicate,
    object_entity,
    supporting_paper_ids,
    contradicting_paper_ids,
    total_sample_size,
    evidence_strength,
    consensus_confidence,
    effect_direction_consistency,
    first_reported,
    last_updated,
    pooled_effect_size,
    effect_size_variance,
):
    """
    **Property 2: Reified Claim Consistency**
    **Validates: Requirements 4.2, 4.3**
    
    Test that ReifiedClaimNode instances maintain consistency invariants:
    1. supporting_paper_ids and contradicting_paper_ids have no overlap
    2. consensus_confidence is always in range [0.0, 1.0]
    3. first_reported <= last_updated
    
    Universal Property:
    - For all valid inputs with non-overlapping paper lists and valid temporal ordering,
      ReifiedClaimNode creation succeeds
    - All consistency invariants are maintained
    """
    # Ensure no overlap between supporting and contradicting papers
    supporting_set = set(supporting_paper_ids)
    contradicting_set = set(contradicting_paper_ids)
    assume(len(supporting_set & contradicting_set) == 0)
    
    # Ensure temporal ordering is valid
    assume(first_reported <= last_updated)
    
    # Create ReifiedClaimNode
    node = ReifiedClaimNode(
        node_id=node_id,
        claim_type=claim_type,
        subject_entity=subject_entity,
        predicate=predicate,
        object_entity=object_entity,
        supporting_paper_ids=supporting_paper_ids,
        contradicting_paper_ids=contradicting_paper_ids,
        total_sample_size=total_sample_size,
        evidence_strength=evidence_strength,
        consensus_confidence=consensus_confidence,
        effect_direction_consistency=effect_direction_consistency,
        first_reported=first_reported,
        last_updated=last_updated,
        pooled_effect_size=pooled_effect_size,
        effect_size_variance=effect_size_variance,
    )
    
    # Property 2a: No overlap between supporting and contradicting papers (Requirement 4.2)
    supporting_set_result = set(node.supporting_paper_ids)
    contradicting_set_result = set(node.contradicting_paper_ids)
    assert len(supporting_set_result & contradicting_set_result) == 0, \
        "supporting_paper_ids and contradicting_paper_ids must not overlap"
    
    # Property 2b: consensus_confidence is in range [0.0, 1.0] (Requirement 4.3)
    assert 0.0 <= node.consensus_confidence <= 1.0, \
        "consensus_confidence must be in range [0.0, 1.0]"
    
    # Property 2c: effect_direction_consistency is in range [0.0, 1.0] (Requirement 4.3)
    assert 0.0 <= node.effect_direction_consistency <= 1.0, \
        "effect_direction_consistency must be in range [0.0, 1.0]"
    
    # Property 2d: first_reported <= last_updated (Requirement 4.5)
    assert node.first_reported <= node.last_updated, \
        "first_reported must be <= last_updated"
    
    # Property 2e: Paper lists contain no duplicates (Requirement 4.2)
    assert len(node.supporting_paper_ids) == len(set(node.supporting_paper_ids)), \
        "supporting_paper_ids must not contain duplicates"
    assert len(node.contradicting_paper_ids) == len(set(node.contradicting_paper_ids)), \
        "contradicting_paper_ids must not contain duplicates"


# ============================================================================
# Negative Property Tests: Overlapping Paper Lists
# ============================================================================

@given(
    claim_id=non_empty_string_strategy,
    claim_type=claim_type_strategy,
    subject_entity=non_empty_string_strategy,
    predicate=non_empty_string_strategy,
    object_entity=non_empty_string_strategy,
    common_papers=paper_id_list_strategy(min_size=1, max_size=5),
    additional_supporting=paper_id_list_strategy(min_size=0, max_size=5),
    additional_contradicting=paper_id_list_strategy(min_size=0, max_size=5),
    evidence_strength=evidence_strength_strategy,
    consensus_confidence=consensus_metric_strategy,
    effect_direction_consistency=consensus_metric_strategy,
    first_reported=iso_date_strategy(),
    last_updated=iso_date_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_property_overlapping_papers_rejected_scientific_claim(
    claim_id,
    claim_type,
    subject_entity,
    predicate,
    object_entity,
    common_papers,
    additional_supporting,
    additional_contradicting,
    evidence_strength,
    consensus_confidence,
    effect_direction_consistency,
    first_reported,
    last_updated,
):
    """
    **Property 2: Reified Claim Consistency (Negative Test)**
    **Validates: Requirements 4.2**
    
    Test that ScientificClaim always rejects overlapping paper lists.
    
    Universal Property:
    - For all inputs where supporting_papers and contradicting_papers overlap,
      ScientificClaim creation fails with ValidationError
    """
    # Ensure temporal ordering is valid
    first_date = datetime.fromisoformat(first_reported)
    last_date = datetime.fromisoformat(last_updated)
    assume(first_date <= last_date)
    
    # Ensure no overlap between additional lists and common papers
    assume(len(set(additional_supporting) & set(common_papers)) == 0)
    assume(len(set(additional_contradicting) & set(common_papers)) == 0)
    assume(len(set(additional_supporting) & set(additional_contradicting)) == 0)
    
    # Create overlapping paper lists
    supporting_papers = list(set(common_papers + additional_supporting))
    contradicting_papers = list(set(common_papers + additional_contradicting))
    
    # Verify there is overlap
    assert len(set(supporting_papers) & set(contradicting_papers)) > 0
    
    # Attempt to create ScientificClaim with overlapping papers
    with pytest.raises(ValidationError) as exc_info:
        ScientificClaim(
            claim_id=claim_id,
            claim_type=claim_type,
            subject_entity=subject_entity,
            predicate=predicate,
            object_entity=object_entity,
            supporting_papers=supporting_papers,
            contradicting_papers=contradicting_papers,
            total_sample_size=100,
            evidence_strength=evidence_strength,
            consensus_confidence=consensus_confidence,
            effect_direction_consistency=effect_direction_consistency,
            first_reported=first_reported,
            last_updated=last_updated,
        )
    
    # Verify the error is about overlapping papers
    error_msg = str(exc_info.value).lower()
    assert "cannot appear in both" in error_msg or "overlap" in error_msg


@given(
    node_id=non_empty_string_strategy,
    claim_type=claim_type_strategy,
    subject_entity=non_empty_string_strategy,
    predicate=non_empty_string_strategy,
    object_entity=non_empty_string_strategy,
    common_papers=paper_id_list_strategy(min_size=1, max_size=5),
    additional_supporting=paper_id_list_strategy(min_size=0, max_size=5),
    additional_contradicting=paper_id_list_strategy(min_size=0, max_size=5),
    evidence_strength=evidence_strength_string_strategy,
    consensus_confidence=consensus_metric_strategy,
    effect_direction_consistency=consensus_metric_strategy,
    first_reported=datetime_strategy(),
    last_updated=datetime_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_property_overlapping_papers_rejected_reified_claim_node(
    node_id,
    claim_type,
    subject_entity,
    predicate,
    object_entity,
    common_papers,
    additional_supporting,
    additional_contradicting,
    evidence_strength,
    consensus_confidence,
    effect_direction_consistency,
    first_reported,
    last_updated,
):
    """
    **Property 2: Reified Claim Consistency (Negative Test)**
    **Validates: Requirements 4.2**
    
    Test that ReifiedClaimNode always rejects overlapping paper lists.
    
    Universal Property:
    - For all inputs where supporting_paper_ids and contradicting_paper_ids overlap,
      ReifiedClaimNode creation fails with ValidationError
    """
    # Ensure temporal ordering is valid
    assume(first_reported <= last_updated)
    
    # Ensure no overlap between additional lists and common papers
    assume(len(set(additional_supporting) & set(common_papers)) == 0)
    assume(len(set(additional_contradicting) & set(common_papers)) == 0)
    assume(len(set(additional_supporting) & set(additional_contradicting)) == 0)
    
    # Create overlapping paper lists
    supporting_paper_ids = list(set(common_papers + additional_supporting))
    contradicting_paper_ids = list(set(common_papers + additional_contradicting))
    
    # Verify there is overlap
    assert len(set(supporting_paper_ids) & set(contradicting_paper_ids)) > 0
    
    # Attempt to create ReifiedClaimNode with overlapping papers
    with pytest.raises(ValidationError) as exc_info:
        ReifiedClaimNode(
            node_id=node_id,
            claim_type=claim_type,
            subject_entity=subject_entity,
            predicate=predicate,
            object_entity=object_entity,
            supporting_paper_ids=supporting_paper_ids,
            contradicting_paper_ids=contradicting_paper_ids,
            total_sample_size=100,
            evidence_strength=evidence_strength,
            consensus_confidence=consensus_confidence,
            effect_direction_consistency=effect_direction_consistency,
            first_reported=first_reported,
            last_updated=last_updated,
        )
    
    # Verify the error is about overlapping papers
    error_msg = str(exc_info.value).lower()
    assert "cannot appear in both" in error_msg or "overlap" in error_msg


# ============================================================================
# Negative Property Tests: Invalid Temporal Ordering
# ============================================================================

@given(
    claim_id=non_empty_string_strategy,
    claim_type=claim_type_strategy,
    subject_entity=non_empty_string_strategy,
    predicate=non_empty_string_strategy,
    object_entity=non_empty_string_strategy,
    supporting_papers=paper_id_list_strategy(min_size=0, max_size=5),
    contradicting_papers=paper_id_list_strategy(min_size=0, max_size=5),
    evidence_strength=evidence_strength_strategy,
    consensus_confidence=consensus_metric_strategy,
    effect_direction_consistency=consensus_metric_strategy,
    first_reported=iso_date_strategy(),
    last_updated=iso_date_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_property_invalid_temporal_ordering_rejected_scientific_claim(
    claim_id,
    claim_type,
    subject_entity,
    predicate,
    object_entity,
    supporting_papers,
    contradicting_papers,
    evidence_strength,
    consensus_confidence,
    effect_direction_consistency,
    first_reported,
    last_updated,
):
    """
    **Property 2: Reified Claim Consistency (Negative Test)**
    **Validates: Requirements 4.5**
    
    Test that ScientificClaim always rejects invalid temporal ordering.
    
    Universal Property:
    - For all inputs where first_reported > last_updated,
      ScientificClaim creation fails with ValidationError
    """
    # Ensure no overlap between paper lists
    supporting_set = set(supporting_papers)
    contradicting_set = set(contradicting_papers)
    assume(len(supporting_set & contradicting_set) == 0)
    
    # Ensure temporal ordering is INVALID
    first_date = datetime.fromisoformat(first_reported)
    last_date = datetime.fromisoformat(last_updated)
    assume(first_date > last_date)
    
    # Attempt to create ScientificClaim with invalid temporal ordering
    with pytest.raises(ValidationError) as exc_info:
        ScientificClaim(
            claim_id=claim_id,
            claim_type=claim_type,
            subject_entity=subject_entity,
            predicate=predicate,
            object_entity=object_entity,
            supporting_papers=supporting_papers,
            contradicting_papers=contradicting_papers,
            total_sample_size=100,
            evidence_strength=evidence_strength,
            consensus_confidence=consensus_confidence,
            effect_direction_consistency=effect_direction_consistency,
            first_reported=first_reported,
            last_updated=last_updated,
        )
    
    # Verify the error is about temporal ordering
    error_msg = str(exc_info.value).lower()
    assert "first_reported" in error_msg


@given(
    node_id=non_empty_string_strategy,
    claim_type=claim_type_strategy,
    subject_entity=non_empty_string_strategy,
    predicate=non_empty_string_strategy,
    object_entity=non_empty_string_strategy,
    supporting_paper_ids=paper_id_list_strategy(min_size=0, max_size=5),
    contradicting_paper_ids=paper_id_list_strategy(min_size=0, max_size=5),
    evidence_strength=evidence_strength_string_strategy,
    consensus_confidence=consensus_metric_strategy,
    effect_direction_consistency=consensus_metric_strategy,
    base_datetime=datetime_strategy(),
    days_offset=st.integers(min_value=1, max_value=365),
)
@settings(max_examples=100, deadline=None)
def test_property_invalid_temporal_ordering_rejected_reified_claim_node(
    node_id,
    claim_type,
    subject_entity,
    predicate,
    object_entity,
    supporting_paper_ids,
    contradicting_paper_ids,
    evidence_strength,
    consensus_confidence,
    effect_direction_consistency,
    base_datetime,
    days_offset,
):
    """
    **Property 2: Reified Claim Consistency (Negative Test)**
    **Validates: Requirements 4.5**
    
    Test that ReifiedClaimNode always rejects invalid temporal ordering.
    
    Universal Property:
    - For all inputs where first_reported > last_updated,
      ReifiedClaimNode creation fails with ValidationError
    """
    # Ensure no overlap between paper lists
    supporting_set = set(supporting_paper_ids)
    contradicting_set = set(contradicting_paper_ids)
    assume(len(supporting_set & contradicting_set) == 0)
    
    # Create invalid temporal ordering: first_reported > last_updated
    last_updated = base_datetime
    first_reported = base_datetime + timedelta(days=days_offset)
    
    # Verify temporal ordering is invalid
    assert first_reported > last_updated
    
    # Attempt to create ReifiedClaimNode with invalid temporal ordering
    with pytest.raises(ValidationError) as exc_info:
        ReifiedClaimNode(
            node_id=node_id,
            claim_type=claim_type,
            subject_entity=subject_entity,
            predicate=predicate,
            object_entity=object_entity,
            supporting_paper_ids=supporting_paper_ids,
            contradicting_paper_ids=contradicting_paper_ids,
            total_sample_size=100,
            evidence_strength=evidence_strength,
            consensus_confidence=consensus_confidence,
            effect_direction_consistency=effect_direction_consistency,
            first_reported=first_reported,
            last_updated=last_updated,
        )
    
    # Verify the error is about temporal ordering
    error_msg = str(exc_info.value).lower()
    assert "first_reported" in error_msg


# ============================================================================
# Negative Property Tests: Invalid Consensus Metrics
# ============================================================================

@given(
    claim_id=non_empty_string_strategy,
    claim_type=claim_type_strategy,
    subject_entity=non_empty_string_strategy,
    predicate=non_empty_string_strategy,
    object_entity=non_empty_string_strategy,
    supporting_papers=paper_id_list_strategy(min_size=0, max_size=5),
    contradicting_papers=paper_id_list_strategy(min_size=0, max_size=5),
    evidence_strength=evidence_strength_strategy,
    invalid_consensus=st.floats(allow_nan=False, allow_infinity=False).filter(
        lambda x: x < 0.0 or x > 1.0
    ),
    effect_direction_consistency=consensus_metric_strategy,
    first_reported=iso_date_strategy(),
    last_updated=iso_date_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_property_invalid_consensus_confidence_rejected(
    claim_id,
    claim_type,
    subject_entity,
    predicate,
    object_entity,
    supporting_papers,
    contradicting_papers,
    evidence_strength,
    invalid_consensus,
    effect_direction_consistency,
    first_reported,
    last_updated,
):
    """
    **Property 2: Reified Claim Consistency (Negative Test)**
    **Validates: Requirements 4.3**
    
    Test that ScientificClaim always rejects invalid consensus_confidence values.
    
    Universal Property:
    - For all consensus_confidence values outside [0.0, 1.0],
      ScientificClaim creation fails with ValidationError
    """
    # Ensure no overlap between paper lists
    supporting_set = set(supporting_papers)
    contradicting_set = set(contradicting_papers)
    assume(len(supporting_set & contradicting_set) == 0)
    
    # Ensure temporal ordering is valid
    first_date = datetime.fromisoformat(first_reported)
    last_date = datetime.fromisoformat(last_updated)
    assume(first_date <= last_date)
    
    # Attempt to create ScientificClaim with invalid consensus_confidence
    with pytest.raises(ValidationError) as exc_info:
        ScientificClaim(
            claim_id=claim_id,
            claim_type=claim_type,
            subject_entity=subject_entity,
            predicate=predicate,
            object_entity=object_entity,
            supporting_papers=supporting_papers,
            contradicting_papers=contradicting_papers,
            total_sample_size=100,
            evidence_strength=evidence_strength,
            consensus_confidence=invalid_consensus,
            effect_direction_consistency=effect_direction_consistency,
            first_reported=first_reported,
            last_updated=last_updated,
        )
    
    # Verify the error is about consensus_confidence
    assert "consensus_confidence" in str(exc_info.value)

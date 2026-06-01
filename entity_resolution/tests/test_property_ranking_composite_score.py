"""
Property 15: Ranking Composite Score Correctness

**Validates: Requirements 4.1, 4.2, 4.3, 2.3**

For any non-empty conflict set:
  - The winner returned by ``rank()`` has the highest composite score among all
    scored candidates.
  - When two candidates have equal composite scores, the one from the
    higher-priority strategy wins.
  - When composite score AND strategy priority are equal, the candidate with
    the lexicographically smallest ``canonical_id`` wins.
  - Calling ``rank()`` twice on the same input returns the same result
    (determinism).

Requirements: 4.1, 4.2, 4.3, 2.3
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.models import CandidateScore
from entity_resolution.ranking_function import RankingFunction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STRATEGIES = list(RankingFunction.PRIORITY_WEIGHTS.keys())
# ["manual_override", "exact", "normalized", "abbreviation", "synonym", "fuzzy", "ontology"]

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for generating a valid canonical_id: non-empty printable ASCII text
_canonical_id_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=30,
)

# Strategy for generating a single CandidateScore with valid fields
_candidate_score_st = st.builds(
    CandidateScore,
    canonical_id=_canonical_id_st,
    strategy=st.sampled_from(VALID_STRATEGIES),
    grounding_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    composite_score=st.just(0.0),  # will be recomputed by score_all()
)

# Strategy for generating a non-empty list of CandidateScore objects
_candidates_st = st.lists(_candidate_score_st, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _recompute_composite(candidate: CandidateScore) -> float:
    """Recompute composite_score = PRIORITY_WEIGHTS[strategy] * grounding_confidence."""
    weight = RankingFunction.PRIORITY_WEIGHTS.get(candidate.strategy, 0.0)
    raw = weight * candidate.grounding_confidence
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Property 15: Ranking Composite Score Correctness
# **Validates: Requirements 4.1, 4.2, 4.3, 2.3**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(candidates=_candidates_st)
def test_property_winner_has_highest_composite_score(
    candidates: list[CandidateScore],
) -> None:
    """
    **Property 15: Ranking Composite Score Correctness**

    **Validates: Requirements 4.1, 4.2**

    For any non-empty conflict set, the winner returned by ``rank()`` must have
    the highest composite score among all scored candidates (as computed by
    ``score_all()``).

    Note: when there is only one candidate, ``rank()`` returns it directly
    without scoring (Requirement 4.6). In that case we verify via ``score_all()``
    that the single candidate's computed composite score is the maximum (trivially
    true), and that the winner's ``canonical_id`` matches.
    """
    rf = RankingFunction()
    winner = rf.rank(candidates)
    scored = rf.score_all(candidates)

    # The winner's canonical_id must match the first element of score_all()
    # (which is the candidate with the highest composite score after tie-breaking)
    assert winner.canonical_id == scored[0].canonical_id, (
        f"Winner canonical_id='{winner.canonical_id}' does not match the top "
        f"scored candidate '{scored[0].canonical_id}' from score_all(): "
        f"{[(c.canonical_id, c.strategy, c.composite_score) for c in scored]}"
    )
    assert winner.strategy == scored[0].strategy, (
        f"Winner strategy='{winner.strategy}' does not match the top "
        f"scored candidate strategy '{scored[0].strategy}' from score_all()"
    )

    # The top scored candidate must have the maximum composite score
    max_composite = max(c.composite_score for c in scored)
    assert scored[0].composite_score == pytest.approx(max_composite), (
        f"Top scored candidate composite_score={scored[0].composite_score} is not "
        f"the maximum ({max_composite}) among all scored candidates: "
        f"{[(c.canonical_id, c.strategy, c.composite_score) for c in scored]}"
    )


@settings(max_examples=100)
@given(candidates=_candidates_st)
def test_property_tie_breaking_by_strategy_priority(
    candidates: list[CandidateScore],
) -> None:
    """
    **Property 15: Ranking Composite Score Correctness — tie-breaking**

    **Validates: Requirements 4.3, 2.3**

    When two or more candidates share the highest composite score, the winner
    must come from the highest-priority strategy (lowest priority weight index
    = highest weight value).
    """
    rf = RankingFunction()
    winner = rf.rank(candidates)
    scored = rf.score_all(candidates)

    max_composite = max(c.composite_score for c in scored)
    top_candidates = [c for c in scored if c.composite_score == pytest.approx(max_composite)]

    # Among top candidates, the winner must have the highest strategy priority weight
    max_priority = max(
        RankingFunction.PRIORITY_WEIGHTS.get(c.strategy, 0.0) for c in top_candidates
    )
    winner_priority = RankingFunction.PRIORITY_WEIGHTS.get(winner.strategy, 0.0)

    assert winner_priority == pytest.approx(max_priority), (
        f"Winner strategy '{winner.strategy}' (priority={winner_priority}) is not "
        f"the highest-priority strategy among top candidates "
        f"(max_priority={max_priority}): "
        f"{[(c.canonical_id, c.strategy, c.composite_score) for c in top_candidates]}"
    )


@settings(max_examples=100)
@given(candidates=_candidates_st)
def test_property_tie_breaking_by_canonical_id(
    candidates: list[CandidateScore],
) -> None:
    """
    **Property 15: Ranking Composite Score Correctness — lexicographic tie-breaking**

    **Validates: Requirements 4.3, 2.3**

    When two or more candidates share both the highest composite score AND the
    highest strategy priority, the winner must be the one with the
    lexicographically smallest ``canonical_id``.
    """
    rf = RankingFunction()
    winner = rf.rank(candidates)
    scored = rf.score_all(candidates)

    max_composite = max(c.composite_score for c in scored)
    top_candidates = [c for c in scored if c.composite_score == pytest.approx(max_composite)]

    max_priority = max(
        RankingFunction.PRIORITY_WEIGHTS.get(c.strategy, 0.0) for c in top_candidates
    )
    top_priority_candidates = [
        c for c in top_candidates
        if RankingFunction.PRIORITY_WEIGHTS.get(c.strategy, 0.0) == pytest.approx(max_priority)
    ]

    # Among candidates with equal composite score and equal strategy priority,
    # the winner must have the lexicographically smallest canonical_id
    min_canonical_id = min(c.canonical_id for c in top_priority_candidates)

    assert winner.canonical_id == min_canonical_id, (
        f"Winner canonical_id='{winner.canonical_id}' is not the lexicographically "
        f"smallest among tied candidates (expected '{min_canonical_id}'): "
        f"{[(c.canonical_id, c.strategy, c.composite_score) for c in top_priority_candidates]}"
    )


@settings(max_examples=100)
@given(candidates=_candidates_st)
def test_property_rank_is_deterministic(
    candidates: list[CandidateScore],
) -> None:
    """
    **Property 15: Ranking Composite Score Correctness — determinism**

    **Validates: Requirements 2.3**

    Calling ``rank()`` twice on the same input must return the same result.
    The winner's ``canonical_id``, ``strategy``, and ``composite_score`` must
    be identical across both calls.
    """
    rf = RankingFunction()
    winner1 = rf.rank(candidates)
    winner2 = rf.rank(candidates)

    assert winner1.canonical_id == winner2.canonical_id, (
        f"rank() is not deterministic: first call returned canonical_id="
        f"'{winner1.canonical_id}', second call returned '{winner2.canonical_id}'"
    )
    assert winner1.strategy == winner2.strategy, (
        f"rank() is not deterministic: first call returned strategy="
        f"'{winner1.strategy}', second call returned '{winner2.strategy}'"
    )
    assert winner1.composite_score == pytest.approx(winner2.composite_score), (
        f"rank() is not deterministic: first call returned composite_score="
        f"{winner1.composite_score}, second call returned {winner2.composite_score}"
    )


@settings(max_examples=100)
@given(candidates=_candidates_st)
def test_property_composite_score_formula(
    candidates: list[CandidateScore],
) -> None:
    """
    **Property 15: Ranking Composite Score Correctness — formula**

    **Validates: Requirements 4.1**

    For every candidate returned by ``score_all()``, the composite_score must
    equal ``PRIORITY_WEIGHTS[strategy] * grounding_confidence``, clamped to
    [0.0, 1.0].
    """
    rf = RankingFunction()
    scored = rf.score_all(candidates)

    for original, scored_candidate in zip(
        sorted(candidates, key=lambda c: (c.canonical_id, c.strategy)),
        sorted(scored, key=lambda c: (c.canonical_id, c.strategy)),
    ):
        # Find the matching scored candidate by canonical_id + strategy
        pass  # We'll check all scored candidates directly below

    for scored_candidate in scored:
        expected = _recompute_composite(scored_candidate)
        assert scored_candidate.composite_score == pytest.approx(expected, abs=1e-9), (
            f"composite_score mismatch for candidate "
            f"(canonical_id='{scored_candidate.canonical_id}', "
            f"strategy='{scored_candidate.strategy}', "
            f"grounding_confidence={scored_candidate.grounding_confidence}): "
            f"expected {expected}, got {scored_candidate.composite_score}"
        )


@settings(max_examples=100)
@given(candidates=_candidates_st)
def test_property_score_all_sorted_correctly(
    candidates: list[CandidateScore],
) -> None:
    """
    **Property 15: Ranking Composite Score Correctness — sort order**

    **Validates: Requirements 4.2, 4.3**

    ``score_all()`` must return candidates sorted by:
      1. composite_score DESC
      2. PRIORITY_WEIGHTS[strategy] DESC
      3. canonical_id ASC

    We verify this by checking that the sort key tuple for each consecutive
    pair is in non-decreasing order (using the negated sort key that Python's
    sort uses internally).
    """
    rf = RankingFunction()
    scored = rf.score_all(candidates)

    def sort_key(c: CandidateScore) -> tuple:
        """Matches the key used in RankingFunction.score_all()."""
        return (
            -c.composite_score,
            -RankingFunction.PRIORITY_WEIGHTS.get(c.strategy, 0.0),
            c.canonical_id,
        )

    for i in range(len(scored) - 1):
        key_a = sort_key(scored[i])
        key_b = sort_key(scored[i + 1])
        assert key_a <= key_b, (
            f"score_all() sort order violated at index {i}: "
            f"sort_key(candidate[{i}])={key_a} > sort_key(candidate[{i+1}])={key_b}\n"
            f"  candidate[{i}]: canonical_id='{scored[i].canonical_id}', "
            f"strategy='{scored[i].strategy}', composite_score={scored[i].composite_score}\n"
            f"  candidate[{i+1}]: canonical_id='{scored[i+1].canonical_id}', "
            f"strategy='{scored[i+1].strategy}', composite_score={scored[i+1].composite_score}"
        )


# ---------------------------------------------------------------------------
# Edge-case unit tests
# ---------------------------------------------------------------------------


def test_single_candidate_returned_directly() -> None:
    """
    Requirement 4.6: when only one candidate is provided, rank() returns it
    directly without scoring.
    """
    rf = RankingFunction()
    candidate = CandidateScore(
        canonical_id="562",
        strategy="exact",
        grounding_confidence=0.9,
        composite_score=0.0,  # not yet scored
    )
    winner = rf.rank([candidate])
    # The single candidate is returned as-is (composite_score not recomputed)
    assert winner.canonical_id == "562"
    assert winner.strategy == "exact"


def test_rank_empty_raises() -> None:
    """rank() must raise ValueError for an empty candidate list."""
    rf = RankingFunction()
    with pytest.raises(ValueError):
        rf.rank([])


def test_rank_highest_composite_wins() -> None:
    """Explicit example: candidate with highest composite score wins."""
    rf = RankingFunction()
    candidates = [
        CandidateScore(canonical_id="A", strategy="fuzzy",    grounding_confidence=1.0, composite_score=0.0),
        CandidateScore(canonical_id="B", strategy="exact",    grounding_confidence=0.5, composite_score=0.0),
        CandidateScore(canonical_id="C", strategy="ontology", grounding_confidence=1.0, composite_score=0.0),
    ]
    # composite scores: fuzzy=0.60, exact=0.475, ontology=0.50
    winner = rf.rank(candidates)
    assert winner.canonical_id == "A"
    assert winner.strategy == "fuzzy"


def test_rank_tie_broken_by_strategy_priority() -> None:
    """
    When composite scores are equal, the higher-priority strategy wins.
    manual_override (1.0) * 0.5 = 0.5 == ontology (0.5) * 1.0 = 0.5
    manual_override has higher priority weight (1.0 > 0.5), so it wins.
    """
    rf = RankingFunction()
    candidates = [
        CandidateScore(canonical_id="Z", strategy="manual_override", grounding_confidence=0.5, composite_score=0.0),
        CandidateScore(canonical_id="A", strategy="ontology",        grounding_confidence=1.0, composite_score=0.0),
    ]
    winner = rf.rank(candidates)
    assert winner.strategy == "manual_override"
    assert winner.canonical_id == "Z"


def test_rank_tie_broken_by_canonical_id() -> None:
    """
    When composite score AND strategy priority are equal, the lexicographically
    smallest canonical_id wins.
    """
    rf = RankingFunction()
    candidates = [
        CandidateScore(canonical_id="beta",  strategy="exact", grounding_confidence=1.0, composite_score=0.0),
        CandidateScore(canonical_id="alpha", strategy="exact", grounding_confidence=1.0, composite_score=0.0),
        CandidateScore(canonical_id="gamma", strategy="exact", grounding_confidence=1.0, composite_score=0.0),
    ]
    winner = rf.rank(candidates)
    assert winner.canonical_id == "alpha"

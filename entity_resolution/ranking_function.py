"""
RankingFunction — deterministic composite scoring for conflict resolution.

Selects the winning canonical entity from a conflict set using:
  composite_score = PRIORITY_WEIGHTS[strategy] × grounding_confidence

Tie-breaking rules (applied in order):
  1. Higher composite_score wins
  2. Equal composite_score → higher-priority strategy wins
  3. Equal strategy priority → lexicographically smallest canonical_id wins

These three rules guarantee a unique winner for any non-empty conflict set.

Requirements: 4.1, 4.2, 4.3, 4.6, 2.3
"""

from __future__ import annotations

from typing import List

from entity_resolution.models import CandidateScore


class RankingFunction:
    """
    Deterministic composite scoring: priority_weight × grounding_confidence.

    Tie-breaking rules (applied in order):
    1. Higher composite score wins
    2. If equal composite score: higher-priority strategy wins
    3. If equal strategy priority: lexicographically smallest canonical_id wins

    These rules guarantee a unique winner for any non-empty conflict set.

    Preconditions for rank():
    - candidates is non-empty
    - Each candidate has a valid strategy name and grounding_confidence in [0.0, 1.0]

    Postconditions for rank():
    - Returns exactly one winner
    - Winner has the highest composite score
    - Tie-breaking is deterministic and consistent across calls
    - If only one candidate: returned directly without scoring
    """

    PRIORITY_WEIGHTS: dict[str, float] = {
        "manual_override": 1.00,
        "exact":           0.95,
        "normalized":      0.85,
        "abbreviation":    0.80,
        "synonym":         0.75,
        "fuzzy":           0.60,
        "ontology":        0.50,
    }

    def score_all(self, candidates: List[CandidateScore]) -> List[CandidateScore]:
        """
        Compute composite scores for all candidates and return them sorted.

        For each candidate:
          composite_score = PRIORITY_WEIGHTS[strategy] × grounding_confidence

        Sort order (descending priority):
          1. composite_score DESC
          2. PRIORITY_WEIGHTS[strategy] DESC  (higher-priority strategy first)
          3. canonical_id ASC                 (lexicographically smallest first)

        Returns a new list of CandidateScore objects with composite_score populated.
        Unknown strategies receive a priority weight of 0.0.

        Requirements: 4.1, 4.2, 4.3
        """
        scored: List[CandidateScore] = []
        for candidate in candidates:
            weight = self.PRIORITY_WEIGHTS.get(candidate.strategy, 0.0)
            composite = weight * candidate.grounding_confidence
            # Clamp to [0.0, 1.0] to satisfy the Field constraint
            composite = max(0.0, min(1.0, composite))
            scored.append(
                CandidateScore(
                    canonical_id=candidate.canonical_id,
                    strategy=candidate.strategy,
                    grounding_confidence=candidate.grounding_confidence,
                    composite_score=composite,
                )
            )

        # Sort: composite_score DESC, strategy priority DESC, canonical_id ASC
        scored.sort(
            key=lambda c: (
                -c.composite_score,
                -self.PRIORITY_WEIGHTS.get(c.strategy, 0.0),
                c.canonical_id,
            )
        )
        return scored

    def rank(self, candidates: List[CandidateScore]) -> CandidateScore:
        """
        Select the winning candidate from a conflict set.

        If only one candidate is provided, it is returned directly without
        scoring (Requirement 4.6).

        Otherwise, composite scores are computed and the first element of
        score_all() is returned as the winner.

        Postconditions:
        - composite_score = PRIORITY_WEIGHTS[strategy] × grounding_confidence
        - Winner is unique (tie-breaking guarantees this)
        - If only one candidate: returned directly without scoring

        Requirements: 4.1, 4.2, 4.3, 4.6, 2.3
        """
        if not candidates:
            raise ValueError("rank() requires at least one candidate")

        # Requirement 4.6: single candidate returned directly without scoring
        if len(candidates) == 1:
            return candidates[0]

        scored = self.score_all(candidates)
        return scored[0]

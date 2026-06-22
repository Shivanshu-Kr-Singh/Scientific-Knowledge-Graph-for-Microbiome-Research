"""
Property Tests for Disagreement Router Decision Logic

**Validates: Requirements 6.1, 6.2, 6.3**

Property 10: Disagreement Router Decision Logic
  For any combination of (stage2_score, stage3_5_decision, blended_confidence),
  the routing decision matches the logic rules exactly:
  - If Stage 2 keeps and Stage 3.5 rejects (or vice versa) → route to LLM
  - If blended_confidence is in [0.40, 0.70] → route to LLM
  - Otherwise → accept Stage 3.5 verdict without LLM
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from collectors.relevance_filter import RelevanceFilter, FilterVerdict
from collectors.embedding_filter import EmbeddingVerdict
from config import BLENDED_CONFIDENCE_LOW, BLENDED_CONFIDENCE_HIGH


# ---------------------------------------------------------------------------
# Fake paper object for testing
# ---------------------------------------------------------------------------

@dataclass
class FakePaper:
    """Minimal paper object with title and abstract attributes."""
    title: str = "Test Paper on Gut Microbiome"
    abstract: str = "This paper studies the human gut microbiome in IBD patients."


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Stage 2 score: float in [0.0, 1.0]
_stage2_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Stage 3.5 decision: one of the three possible outcomes
_stage3_5_decision_st = st.sampled_from(["KEEP", "REJECT", "BORDERLINE"])

# Blended confidence: float in [0.0, 1.0]
_blended_confidence_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Keep threshold for Stage 2 (standard value)
KEEP_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Property 10: Disagreement Router Decision Logic
# **Validates: Requirements 6.1, 6.2, 6.3**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    stage2_score=_stage2_score_st,
    stage3_5_decision=_stage3_5_decision_st,
    blended_confidence=_blended_confidence_st,
)
def test_property_disagreement_router_decision_logic(
    stage2_score: float,
    stage3_5_decision: str,
    blended_confidence: float,
) -> None:
    """
    **Property 10: Disagreement Router Decision Logic**

    **Validates: Requirements 6.1, 6.2, 6.3**

    For any (stage2_score, stage3_5_decision, blended_confidence), the routing
    decision matches the logic rules exactly:
      1. If stage2 keeps and stage3.5 rejects (or vice versa) → route to LLM
      2. If blended_confidence in [BLENDED_CONFIDENCE_LOW, BLENDED_CONFIDENCE_HIGH] → route to LLM
      3. Otherwise → accept Stage 3.5 verdict (no LLM)
    """
    # Create a mock RelevanceFilter with only the attributes needed by
    # _disagreement_router — avoids loading real models/config from disk
    mock_rf = MagicMock(spec=RelevanceFilter)
    mock_rf.thresholds = {"keep": KEEP_THRESHOLD, "review": 0.40}

    # Build the Stage 2 verdict with the generated score
    stage2_verdict = FilterVerdict(
        keep=(stage2_score >= KEEP_THRESHOLD),
        score=stage2_score,
        stage="stage2_rules",
        reason="test",
        review=False,
    )

    # Build the Stage 3.5 verdict with the generated decision
    stage3_5_verdict = EmbeddingVerdict(
        decision=stage3_5_decision,
        pos_similarity=0.5,
        neg_similarity=0.5,
        reason="test",
    )

    # Create a fake paper
    paper = FakePaper()

    # Call the actual _disagreement_router method using the unbound method
    # on our mock instance (provides self.thresholds)
    route_to_llm, reason = RelevanceFilter._disagreement_router(
        mock_rf, paper, stage2_verdict, stage3_5_verdict, blended_confidence
    )

    # Determine expected routing decision by re-implementing the logic
    stage2_keeps = stage2_score >= KEEP_THRESHOLD
    stage3_5_keeps = stage3_5_decision == "KEEP"
    stage3_5_rejects = stage3_5_decision == "REJECT"

    # Condition 1: Verdict disagreement
    verdicts_disagree = (
        (stage2_keeps and stage3_5_rejects)
        or (not stage2_keeps and stage3_5_keeps)
    )

    # Condition 2: Blended confidence in uncertain zone
    confidence_borderline = (
        BLENDED_CONFIDENCE_LOW <= blended_confidence <= BLENDED_CONFIDENCE_HIGH
    )

    # Expected: route to LLM if either condition is met
    # Note: condition 1 is checked first in the implementation
    if verdicts_disagree:
        expected_route = True
    elif confidence_borderline:
        expected_route = True
    else:
        expected_route = False

    assert route_to_llm == expected_route, (
        f"Routing mismatch for stage2_score={stage2_score:.4f} "
        f"(keeps={stage2_keeps}), stage3_5={stage3_5_decision}, "
        f"blended_confidence={blended_confidence:.4f}: "
        f"expected route_to_llm={expected_route}, got {route_to_llm}"
    )

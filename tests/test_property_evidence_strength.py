"""
Property-based tests for EvidenceStrengthClassifier.

Uses Hypothesis to verify correctness properties across a wide range
of generated inputs.

Requirements: 6.5, 6.6
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graph.evidence_strength_classifier import EvidenceStrengthClassifier


# Feature: open-world-triple-promotion, Property 14: Claim-level evidence strength computation
class TestProperty14ClaimLevelEvidenceStrength:
    """
    Property 14: Claim-level evidence strength computation

    For any OpenWorldClaim with N supporting evidence items each having
    individual evidence_strength:
    - If at least 3 items have strength "moderate" or "strong", the claim
      strength SHALL be "strong"
    - Otherwise, the claim strength SHALL be the maximum among all individual
      strengths (where strong > moderate > weak)

    Validates: Requirements 6.5, 6.6
    """

    STRENGTH_ORDER = {"weak": 0, "moderate": 1, "strong": 2}
    STRENGTH_FROM_VALUE = {0: "weak", 1: "moderate", 2: "strong"}

    @pytest.fixture(autouse=True)
    def setup_classifier(self):
        self.classifier = EvidenceStrengthClassifier()

    @given(
        individual_strengths=st.lists(
            st.sampled_from(["strong", "moderate", "weak"]),
            min_size=1,
            max_size=20,
        ),
        paper_count=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_claim_strength_promotion_or_max(self, individual_strengths, paper_count):
        """
        Verify that classify_claim promotes to "strong" when >= 3 papers have
        moderate+ evidence, and returns max-strength otherwise.

        Validates: Requirements 6.5, 6.6
        """
        # Compute expected result using oracle logic
        moderate_or_strong_count = sum(
            1 for s in individual_strengths if s in ("moderate", "strong")
        )

        if moderate_or_strong_count >= 3:
            expected = "strong"
        else:
            # Max strength among all individual items
            max_value = max(
                self.STRENGTH_ORDER[s] for s in individual_strengths
            )
            expected = self.STRENGTH_FROM_VALUE[max_value]

        # Act
        result = self.classifier.classify_claim(individual_strengths, paper_count)

        # Assert
        assert result == expected, (
            f"classify_claim({individual_strengths}, {paper_count}) = {result!r}, "
            f"expected {expected!r}. "
            f"moderate_or_strong_count={moderate_or_strong_count}"
        )

    @given(
        individual_strengths=st.lists(
            st.sampled_from(["strong", "moderate", "weak"]),
            min_size=3,
            max_size=20,
        ).filter(
            lambda xs: sum(1 for s in xs if s in ("moderate", "strong")) >= 3
        ),
        paper_count=st.integers(min_value=3, max_value=20),
    )
    @settings(max_examples=100)
    def test_claim_promoted_to_strong_when_three_or_more_moderate_plus(
        self, individual_strengths, paper_count
    ):
        """
        When at least 3 items have strength "moderate" or "strong", the
        claim-level evidence strength SHALL be "strong".

        Validates: Requirements 6.6
        """
        result = self.classifier.classify_claim(individual_strengths, paper_count)
        assert result == "strong", (
            f"Expected 'strong' when >= 3 moderate/strong items, "
            f"got {result!r}. strengths={individual_strengths}"
        )

    @given(
        individual_strengths=st.lists(
            st.sampled_from(["strong", "moderate", "weak"]),
            min_size=1,
            max_size=20,
        ).filter(
            lambda xs: sum(1 for s in xs if s in ("moderate", "strong")) < 3
        ),
        paper_count=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_claim_returns_max_strength_when_fewer_than_three_moderate_plus(
        self, individual_strengths, paper_count
    ):
        """
        When fewer than 3 items have moderate+ strength, the claim-level
        evidence strength SHALL be the maximum of individual strengths
        (strong > moderate > weak).

        Validates: Requirements 6.5
        """
        max_value = max(
            self.STRENGTH_ORDER[s] for s in individual_strengths
        )
        expected = self.STRENGTH_FROM_VALUE[max_value]

        result = self.classifier.classify_claim(individual_strengths, paper_count)
        assert result == expected, (
            f"Expected max-strength {expected!r}, got {result!r}. "
            f"strengths={individual_strengths}"
        )

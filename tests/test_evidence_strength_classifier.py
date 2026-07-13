"""
Unit tests for EvidenceStrengthClassifier.

Validates classification logic for both single-triple and claim-level
evidence strength computation.

Requirements: 6.1, 6.2, 6.3, 6.5, 6.6
"""

import pytest

from graph.evidence_strength_classifier import EvidenceStrengthClassifier


@pytest.fixture
def classifier():
    return EvidenceStrengthClassifier()


class TestClassifySingle:
    """Tests for classify_single method."""

    # --- Strong classification tests ---

    def test_strong_results_original_research_high_confidence(self, classifier):
        """Results section + original_research + confidence >= 0.85 → strong."""
        result = classifier.classify_single(0.90, "results", "original_research")
        assert result == "strong"

    def test_strong_discussion_meta_analysis_high_confidence(self, classifier):
        """Discussion section + meta_analysis + confidence >= 0.85 → strong."""
        result = classifier.classify_single(0.85, "discussion", "meta_analysis")
        assert result == "strong"

    def test_strong_boundary_confidence_085(self, classifier):
        """Exactly 0.85 confidence with correct section and article type → strong."""
        result = classifier.classify_single(0.85, "results", "original_research")
        assert result == "strong"

    def test_strong_confidence_1_0(self, classifier):
        """Maximum confidence → strong."""
        result = classifier.classify_single(1.0, "results", "meta_analysis")
        assert result == "strong"

    # --- Moderate classification tests ---

    def test_moderate_results_confidence_between_07_085(self, classifier):
        """Results section + confidence in [0.7, 0.85) → moderate."""
        result = classifier.classify_single(0.75, "results", "original_research")
        assert result == "moderate"

    def test_moderate_discussion_confidence_07(self, classifier):
        """Discussion + exactly 0.7 confidence → moderate."""
        result = classifier.classify_single(0.7, "discussion", "review")
        assert result == "moderate"

    def test_moderate_results_high_confidence_review_article(self, classifier):
        """Results + confidence >= 0.85 but article_type=review → moderate.
        
        This tests the edge case where confidence is high but article type
        doesn't qualify for "strong", so it falls to "moderate" since
        section is results/discussion and confidence >= 0.7.
        """
        result = classifier.classify_single(0.90, "results", "review")
        assert result == "moderate"

    def test_moderate_discussion_high_confidence_non_strong_article(self, classifier):
        """Discussion + confidence >= 0.85 but not original_research/meta_analysis → moderate."""
        result = classifier.classify_single(0.95, "discussion", "case_report")
        assert result == "moderate"

    # --- Weak classification tests ---

    def test_weak_abstract_section(self, classifier):
        """Abstract section → weak regardless of confidence."""
        result = classifier.classify_single(0.95, "abstract", "original_research")
        assert result == "weak"

    def test_weak_introduction_section(self, classifier):
        """Introduction section → weak regardless of confidence."""
        result = classifier.classify_single(0.90, "introduction", "meta_analysis")
        assert result == "weak"

    def test_weak_low_confidence(self, classifier):
        """Confidence < 0.7 → weak regardless of section/article type."""
        result = classifier.classify_single(0.65, "results", "original_research")
        assert result == "weak"

    def test_weak_confidence_just_below_07(self, classifier):
        """Confidence just below 0.7 → weak."""
        result = classifier.classify_single(0.699, "results", "original_research")
        assert result == "weak"

    def test_weak_abstract_high_confidence(self, classifier):
        """Abstract with high confidence is still weak."""
        result = classifier.classify_single(1.0, "abstract", "original_research")
        assert result == "weak"

    def test_weak_unknown_section_type(self, classifier):
        """Unknown section type (e.g., "methods") with confidence >= 0.7 → weak."""
        result = classifier.classify_single(0.80, "methods", "original_research")
        assert result == "weak"

    # --- Edge cases ---

    def test_case_insensitive_section(self, classifier):
        """Section type matching is case-insensitive."""
        result = classifier.classify_single(0.90, "Results", "original_research")
        assert result == "strong"

    def test_case_insensitive_article_type(self, classifier):
        """Article type matching is case-insensitive."""
        result = classifier.classify_single(0.90, "results", "Original_Research")
        assert result == "strong"

    def test_whitespace_trimmed(self, classifier):
        """Leading/trailing whitespace in inputs is trimmed."""
        result = classifier.classify_single(0.90, " results ", " original_research ")
        assert result == "strong"


class TestClassifyClaim:
    """Tests for classify_claim method."""

    # --- Promotion rule: >= 3 moderate/strong → "strong" ---

    def test_promotion_three_strong(self, classifier):
        """3 strong items → claim is strong."""
        result = classifier.classify_claim(["strong", "strong", "strong"], 3)
        assert result == "strong"

    def test_promotion_three_moderate(self, classifier):
        """3 moderate items → claim is strong (promotion rule)."""
        result = classifier.classify_claim(["moderate", "moderate", "moderate"], 3)
        assert result == "strong"

    def test_promotion_mixed_moderate_strong(self, classifier):
        """Mix of 2 moderate + 1 strong (3 total moderate+) → strong."""
        result = classifier.classify_claim(["moderate", "strong", "moderate"], 3)
        assert result == "strong"

    def test_promotion_three_moderate_plus_weak(self, classifier):
        """3 moderate + 1 weak → still promoted to strong."""
        result = classifier.classify_claim(
            ["moderate", "weak", "moderate", "moderate"], 4
        )
        assert result == "strong"

    # --- Max-strength fallback (< 3 moderate/strong) ---

    def test_max_strength_two_strong(self, classifier):
        """2 strong items (< 3 moderate+) → max is strong."""
        result = classifier.classify_claim(["strong", "strong"], 2)
        assert result == "strong"

    def test_max_strength_two_moderate(self, classifier):
        """2 moderate items → max is moderate."""
        result = classifier.classify_claim(["moderate", "moderate"], 2)
        assert result == "moderate"

    def test_max_strength_one_moderate_one_weak(self, classifier):
        """1 moderate + 1 weak → max is moderate."""
        result = classifier.classify_claim(["moderate", "weak"], 2)
        assert result == "moderate"

    def test_max_strength_all_weak(self, classifier):
        """All weak → max is weak."""
        result = classifier.classify_claim(["weak", "weak", "weak"], 3)
        assert result == "weak"

    def test_max_strength_single_strong(self, classifier):
        """Single strong item → max is strong."""
        result = classifier.classify_claim(["strong"], 1)
        assert result == "strong"

    # --- Edge cases ---

    def test_empty_list_returns_weak(self, classifier):
        """Empty individual_strengths list → weak."""
        result = classifier.classify_claim([], 0)
        assert result == "weak"

    def test_single_weak(self, classifier):
        """Single weak item → weak."""
        result = classifier.classify_claim(["weak"], 1)
        assert result == "weak"

    def test_exactly_two_moderate_not_promoted(self, classifier):
        """2 moderate items (< 3) → not promoted, max is moderate."""
        result = classifier.classify_claim(["moderate", "moderate", "weak"], 3)
        assert result == "moderate"

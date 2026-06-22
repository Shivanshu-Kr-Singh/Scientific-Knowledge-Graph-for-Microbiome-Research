"""
Unit tests for collectors/embedding_filter.py

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7**

Tests the EmbeddingFilter class decision logic using mocked embedding model
and store to isolate the threshold logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from collectors.embedding_filter import EmbeddingFilter, EmbeddingVerdict
from collectors.embedding_store import SimilarityResult, EmbeddingMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakePaper:
    """Minimal paper object for testing."""
    title: str = "Test Paper Title"
    abstract: Optional[str] = "Test paper abstract about microbiome."


def _make_similarity_result(score: float) -> SimilarityResult:
    """Create a SimilarityResult with the given score."""
    meta = EmbeddingMetadata(
        doi="10.1000/test",
        pmid="12345678",
        title="Some stored paper",
        partition="positive",
        added_at="2024-01-01T00:00:00Z",
    )
    return SimilarityResult(score=score, metadata=meta)


def _make_filter(pos_count: int = 100, neg_count: int = 100,
                 pos_sim: float = 0.5, neg_sim: float = 0.5) -> EmbeddingFilter:
    """Create an EmbeddingFilter with mocked dependencies."""
    mock_model = MagicMock()
    mock_model.encode_paper.return_value = np.zeros(384, dtype=np.float32)

    mock_store = MagicMock()
    mock_store.positive_count = pos_count
    mock_store.negative_count = neg_count

    def query_similar(vector, partition, top_k=1):
        if partition == "positive":
            return [_make_similarity_result(pos_sim)]
        else:
            return [_make_similarity_result(neg_sim)]

    mock_store.query_similar.side_effect = query_similar

    return EmbeddingFilter(embedding_model=mock_model, embedding_store=mock_store)


# ---------------------------------------------------------------------------
# Tests: INSUFFICIENT_DATA
# ---------------------------------------------------------------------------


class TestInsufficientData:
    """Tests for INSUFFICIENT_DATA verdict when partitions are too small."""

    def test_insufficient_positive_partition(self):
        """Validates: Requirement 5.7 — fewer than 50 in positive partition."""
        ef = _make_filter(pos_count=30, neg_count=100)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "INSUFFICIENT_DATA"
        assert verdict.pos_similarity == 0.0
        assert verdict.neg_similarity == 0.0

    def test_insufficient_negative_partition(self):
        """Validates: Requirement 5.7 — fewer than 50 in negative partition."""
        ef = _make_filter(pos_count=100, neg_count=10)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "INSUFFICIENT_DATA"

    def test_insufficient_both_partitions(self):
        """Validates: Requirement 5.7 — both partitions below minimum."""
        ef = _make_filter(pos_count=0, neg_count=0)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "INSUFFICIENT_DATA"

    def test_exactly_at_minimum_is_sufficient(self):
        """50 in each partition should NOT trigger INSUFFICIENT_DATA."""
        ef = _make_filter(pos_count=50, neg_count=50, pos_sim=0.5, neg_sim=0.5)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision != "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Tests: KEEP decision
# ---------------------------------------------------------------------------


class TestKeepDecision:
    """Tests for KEEP verdict."""

    def test_keep_high_positive_low_negative(self):
        """Validates: Requirement 5.3 — pos_sim >= 0.85 AND neg_sim < 0.60."""
        ef = _make_filter(pos_sim=0.90, neg_sim=0.30)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "KEEP"
        assert verdict.pos_similarity == 0.90
        assert verdict.neg_similarity == 0.30

    def test_keep_exact_threshold(self):
        """Validates: Requirement 5.3 — pos_sim exactly 0.85, neg_sim exactly 0.59."""
        ef = _make_filter(pos_sim=0.85, neg_sim=0.59)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "KEEP"

    def test_not_keep_when_neg_at_ceiling(self):
        """pos_sim >= 0.85 but neg_sim == 0.60 → not KEEP (BORDERLINE)."""
        ef = _make_filter(pos_sim=0.90, neg_sim=0.60)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "BORDERLINE"


# ---------------------------------------------------------------------------
# Tests: REJECT decision
# ---------------------------------------------------------------------------


class TestRejectDecision:
    """Tests for REJECT verdict."""

    def test_reject_high_negative_low_positive(self):
        """Validates: Requirement 5.4 — neg_sim >= 0.85 AND pos_sim < 0.60."""
        ef = _make_filter(pos_sim=0.30, neg_sim=0.90)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "REJECT"
        assert verdict.pos_similarity == 0.30
        assert verdict.neg_similarity == 0.90

    def test_reject_exact_threshold(self):
        """Validates: Requirement 5.4 — neg_sim exactly 0.85, pos_sim exactly 0.59."""
        ef = _make_filter(pos_sim=0.59, neg_sim=0.85)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "REJECT"

    def test_not_reject_when_pos_at_ceiling(self):
        """neg_sim >= 0.85 but pos_sim == 0.60 → not REJECT (BORDERLINE)."""
        ef = _make_filter(pos_sim=0.60, neg_sim=0.90)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "BORDERLINE"


# ---------------------------------------------------------------------------
# Tests: BORDERLINE decision
# ---------------------------------------------------------------------------


class TestBorderlineDecision:
    """Tests for BORDERLINE verdict."""

    def test_borderline_both_moderate(self):
        """Validates: Requirement 5.5 — neither threshold met."""
        ef = _make_filter(pos_sim=0.70, neg_sim=0.70)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "BORDERLINE"

    def test_borderline_both_high(self):
        """Both similarities high — conflicting signals → BORDERLINE."""
        ef = _make_filter(pos_sim=0.90, neg_sim=0.90)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "BORDERLINE"

    def test_borderline_both_low(self):
        """Both similarities low — no confident signal → BORDERLINE."""
        ef = _make_filter(pos_sim=0.40, neg_sim=0.40)
        verdict = ef.evaluate(FakePaper())
        assert verdict.decision == "BORDERLINE"


# ---------------------------------------------------------------------------
# Tests: Verdict structure
# ---------------------------------------------------------------------------


class TestVerdictStructure:
    """Tests for EmbeddingVerdict dataclass."""

    def test_verdict_stage_default(self):
        """Validates: Requirement 5.6 — stage is set correctly."""
        ef = _make_filter(pos_sim=0.90, neg_sim=0.30)
        verdict = ef.evaluate(FakePaper())
        assert verdict.stage == "stage3_5_embedding"

    def test_verdict_has_reason(self):
        """Verdict always includes a reason string."""
        ef = _make_filter(pos_sim=0.90, neg_sim=0.30)
        verdict = ef.evaluate(FakePaper())
        assert len(verdict.reason) > 0

    def test_verdict_dataclass_fields(self):
        """EmbeddingVerdict has all required fields."""
        v = EmbeddingVerdict(
            decision="KEEP",
            pos_similarity=0.9,
            neg_similarity=0.3,
            reason="test",
        )
        assert v.decision == "KEEP"
        assert v.pos_similarity == 0.9
        assert v.neg_similarity == 0.3
        assert v.reason == "test"
        assert v.stage == "stage3_5_embedding"


# ---------------------------------------------------------------------------
# Tests: Paper attribute handling
# ---------------------------------------------------------------------------


class TestPaperAttributes:
    """Tests for safe attribute access on paper objects."""

    def test_paper_with_none_abstract(self):
        """Handle paper with None abstract gracefully."""
        paper = FakePaper(title="Some Title", abstract=None)
        ef = _make_filter(pos_sim=0.70, neg_sim=0.30)
        verdict = ef.evaluate(paper)
        assert verdict.decision in ("KEEP", "REJECT", "BORDERLINE", "INSUFFICIENT_DATA")

    def test_paper_with_empty_title(self):
        """Handle paper with empty title."""
        paper = FakePaper(title="", abstract="Some abstract")
        ef = _make_filter(pos_sim=0.70, neg_sim=0.30)
        verdict = ef.evaluate(paper)
        assert verdict.decision in ("KEEP", "REJECT", "BORDERLINE", "INSUFFICIENT_DATA")

    def test_paper_without_abstract_attr(self):
        """Handle paper object that lacks abstract attribute entirely."""

        class MinimalPaper:
            title = "Only has title"

        ef = _make_filter(pos_sim=0.70, neg_sim=0.30)
        verdict = ef.evaluate(MinimalPaper())
        assert verdict.decision in ("KEEP", "REJECT", "BORDERLINE", "INSUFFICIENT_DATA")

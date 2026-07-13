"""
Property-based tests for PredicateRegistry paper-frequency tracking,
threshold promotion, and promoted predicate normalization.

Uses Hypothesis to verify correctness properties across a wide range
of generated inputs.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

import os
import string
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from graph.predicate_registry import (
    PredicateRegistry,
    PREDICATE_NORMALIZATION,
)


def _novel_predicate_strategy():
    """
    Strategy for generating predicate strings that will NOT match
    any existing key in PREDICATE_NORMALIZATION (including partial matches).

    Uses a prefix 'zxqj_' that doesn't appear in any existing predicate key,
    followed by random lowercase letters, to guarantee novelty.
    """
    return st.text(
        alphabet=string.ascii_lowercase,
        min_size=3,
        max_size=10,
    ).map(lambda s: f"zxqj_{s}")


# Feature: open-world-triple-promotion, Property 11: Predicate paper-frequency tracking and threshold promotion
class TestProperty11PredicatePromotionThreshold:
    """
    Property 11: Predicate paper-frequency tracking and threshold promotion

    For any novel predicate and a sequence of calls track_paper_occurrence
    with T distinct paper_ids (where T equals the configured promotion_threshold),
    after the T-th distinct paper, the predicate SHALL be marked promoted=1.
    Before the T-th distinct paper, it SHALL remain promoted=0.

    Validates: Requirements 4.1, 4.2
    """

    @given(
        threshold=st.integers(min_value=2, max_value=10),
        predicate_suffix=st.text(
            alphabet=string.ascii_lowercase,
            min_size=3,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_promoted_only_at_threshold(self, threshold, predicate_suffix):
        """
        Feed T distinct paper_ids for a novel predicate, verify
        is_newly_promoted is False for calls 1..T-1 and True at call T.

        Validates: Requirements 4.1, 4.2
        """
        # Novel predicate that won't match any existing key
        novel_predicate = f"zxqj_{predicate_suffix}"

        # Ensure this predicate is actually novel (not in PREDICATE_NORMALIZATION)
        assume(novel_predicate.lower().strip() not in PREDICATE_NORMALIZATION)
        # Also check partial matches don't trigger
        assume(not any(
            k in novel_predicate.lower() or novel_predicate.lower() in k
            for k in PREDICATE_NORMALIZATION
        ))

        # Use a unique temp DB for this test run and set threshold via env
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "pred_reg_test.db"

            with patch("graph.predicate_registry.REGISTRY_DB_PATH", db_path), \
                 patch.dict(os.environ, {"PREDICATE_PROMOTION_THRESHOLD": str(threshold)}):

                # Create a fresh PredicateRegistry with the temp DB
                registry = PredicateRegistry()

                # Feed T distinct paper IDs
                paper_ids = [f"paper_{i:04d}" for i in range(threshold)]

                for i, paper_id in enumerate(paper_ids):
                    canonical, is_known, is_newly_promoted = registry.track_paper_occurrence(
                        novel_predicate, paper_id
                    )

                    if i < threshold - 1:
                        # Before the T-th call, should NOT be newly promoted
                        assert is_newly_promoted is False, (
                            f"Expected is_newly_promoted=False at call {i+1}/{threshold}, "
                            f"but got True. predicate={novel_predicate!r}, paper_id={paper_id!r}"
                        )
                    else:
                        # At the T-th call (index T-1), should be newly promoted
                        assert is_newly_promoted is True, (
                            f"Expected is_newly_promoted=True at call {i+1}/{threshold}, "
                            f"but got False. predicate={novel_predicate!r}, paper_id={paper_id!r}"
                        )


# Feature: open-world-triple-promotion, Property 12: Promoted predicate normalization
class TestProperty12PromotedPredicateNormalization:
    """
    Property 12: Promoted predicate normalization

    For any novel predicate that has been promoted, calling normalize()
    SHALL return the promoted canonical form (not "RELATES_TO"), and
    subsequent PromotedTriple objects using that predicate SHALL have
    relationship_type equal to the promoted canonical form.

    Validates: Requirements 4.3, 4.4
    """

    @given(
        predicate_suffix=st.text(
            alphabet=string.ascii_lowercase,
            min_size=3,
            max_size=10,
        ),
    )
    @settings(max_examples=100)
    def test_normalize_returns_canonical_after_promotion(self, predicate_suffix):
        """
        After promoting a novel predicate, normalize() returns the promoted
        canonical form (uppercase, underscores), not "RELATES_TO".

        Validates: Requirements 4.3, 4.4
        """
        # Novel predicate that won't match existing keys
        novel_predicate = f"zxqj_{predicate_suffix}"

        # Ensure this predicate is actually novel
        assume(novel_predicate.lower().strip() not in PREDICATE_NORMALIZATION)
        assume(not any(
            k in novel_predicate.lower() or novel_predicate.lower() in k
            for k in PREDICATE_NORMALIZATION
        ))

        # Use a unique temp DB
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "pred_reg_norm_test.db"

            with patch("graph.predicate_registry.REGISTRY_DB_PATH", db_path):
                # Create a fresh PredicateRegistry
                registry = PredicateRegistry()

                # Verify it's novel before promotion (normalizes to RELATES_TO)
                canonical_before, is_known_before = registry.normalize(novel_predicate)
                assert canonical_before == "RELATES_TO", (
                    f"Expected 'RELATES_TO' before promotion, got {canonical_before!r}"
                )
                assert is_known_before is False

                # Promote the predicate
                promoted_canonical = registry.promote_predicate(novel_predicate)

                # The canonical form should be uppercase with underscores
                expected_canonical = (
                    novel_predicate.lower().strip().upper().replace(" ", "_").replace("-", "_")
                )
                assert promoted_canonical == expected_canonical, (
                    f"Expected canonical form {expected_canonical!r}, got {promoted_canonical!r}"
                )

                # After promotion, normalize() should return the promoted form, NOT "RELATES_TO"
                canonical_after, is_known_after = registry.normalize(novel_predicate)
                assert canonical_after == promoted_canonical, (
                    f"After promotion, normalize() should return {promoted_canonical!r}, "
                    f"got {canonical_after!r}"
                )
                assert canonical_after != "RELATES_TO", (
                    f"After promotion, normalize() should NOT return 'RELATES_TO', "
                    f"but got {canonical_after!r}"
                )
                assert is_known_after is True, (
                    f"After promotion, is_known should be True, got {is_known_after!r}"
                )

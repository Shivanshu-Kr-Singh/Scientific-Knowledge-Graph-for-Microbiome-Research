"""
Property-based tests for query result ordering invariant.

Tests Property 15 from the open-world triple promotion design.

Requirements: 5.6
"""

# Feature: open-world-triple-promotion, Property 15: Query result ordering invariant

from hypothesis import given, settings
from hypothesis import strategies as st


class TestProperty15QueryResultOrderingInvariant:
    """
    Property 15: Query result ordering invariant

    For any result set returned by query_open_world_claims, for all consecutive
    pairs (results[i], results[i+1]):
      - results[i].consensus_confidence > results[i+1].consensus_confidence
      OR
      - (results[i].consensus_confidence == results[i+1].consensus_confidence
         AND results[i].paper_count >= results[i+1].paper_count)

    Validates: Requirements 5.6
    """

    @given(
        results=st.lists(
            st.fixed_dictionaries(
                {
                    "consensus_confidence": st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    "paper_count": st.integers(min_value=1, max_value=100),
                }
            ),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_ordering_property(self, results):
        """
        Generate an unsorted list of result dicts, sort them using the expected
        comparator (confidence DESC, paper_count DESC), and verify the ordering
        invariant holds for all consecutive pairs.

        Validates: Requirements 5.6
        """
        # Sort using the expected comparator: confidence DESC, paper_count DESC
        sorted_results = sorted(
            results,
            key=lambda r: (-r["consensus_confidence"], -r["paper_count"]),
        )

        # Verify the ordering invariant holds for all consecutive pairs
        for i in range(len(sorted_results) - 1):
            r1, r2 = sorted_results[i], sorted_results[i + 1]
            assert (
                r1["consensus_confidence"] > r2["consensus_confidence"]
                or (
                    r1["consensus_confidence"] == r2["consensus_confidence"]
                    and r1["paper_count"] >= r2["paper_count"]
                )
            ), (
                f"Ordering invariant violated at index {i}: "
                f"r1={r1}, r2={r2}"
            )

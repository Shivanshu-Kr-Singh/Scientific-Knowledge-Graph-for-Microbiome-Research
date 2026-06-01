"""
FuzzyMatcher — Levenshtein edit-distance matching for the Deterministic
Entity Resolution Pipeline.

Applies fuzzy matching (edit distance ≤ 2) against all canonical entity
surface forms for a given entity type.  Short surface forms (< 4 Unicode
code points after normalisation) are skipped immediately.

Confidence formula:
    confidence = 1.0 - (edit_distance / max(len_surface, len_candidate)) * 0.5

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from pydantic import BaseModel, Field

from entity_resolution.utils import normalize_surface_form

if TYPE_CHECKING:
    from entity_resolution.canonical_registry import CanonicalRegistry


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class FuzzyCandidate(BaseModel):
    """
    A single candidate produced by the FuzzyMatcher.

    Requirements: 12.1, 12.3
    """

    canonical_id: str
    matched_surface_form: str
    edit_distance: int
    grounding_confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Levenshtein distance (pure-Python DP, no external dependency)
# ---------------------------------------------------------------------------


def _levenshtein(s: str, t: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings using a
    standard two-row dynamic-programming algorithm.

    Early-exit optimisation: if the absolute length difference already
    exceeds 2, return 3 immediately (above the threshold we care about).

    Args:
        s: First string (already normalised).
        t: Second string (already normalised).

    Returns:
        Integer edit distance.
    """
    len_s = len(s)
    len_t = len(t)

    # Fast path: length difference alone exceeds threshold
    if abs(len_s - len_t) > 2:
        return 3  # guaranteed > 2, no need to compute exactly

    # Trivial cases
    if s == t:
        return 0
    if len_s == 0:
        return len_t
    if len_t == 0:
        return len_s

    # Two-row DP
    prev = list(range(len_t + 1))
    curr = [0] * (len_t + 1)

    for i in range(1, len_s + 1):
        curr[0] = i
        row_min = curr[0]
        for j in range(1, len_t + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
            if curr[j] < row_min:
                row_min = curr[j]
        # Early exit: if the minimum value in this row already exceeds 2,
        # the final distance will also exceed 2.
        if row_min > 2:
            return 3
        prev, curr = curr, prev

    return prev[len_t]


# ---------------------------------------------------------------------------
# FuzzyMatcher
# ---------------------------------------------------------------------------


class FuzzyMatcher:
    """
    Levenshtein edit-distance matching with edit distance ≤ 2 threshold.

    Normalisation applied before matching (via ``normalize_surface_form``):
    - Unicode NFC normalisation
    - Case-folding (lowercase)
    - Punctuation stripping
    - Whitespace collapsing

    Confidence formula:
        confidence = 1.0 - (edit_distance / max(len_surface, len_candidate)) * 0.5
        - edit_distance=0 → confidence=1.0
        - Lengths measured in Unicode code points after normalisation

    Preconditions for match():
    - surface_form is non-empty
    - registry is a CanonicalRegistry instance

    Postconditions for match():
    - Returns empty list if len(normalized_surface) < 4 code points  (Req 12.5)
    - Returns empty list if no candidates within edit distance ≤ 2   (Req 12.6)
    - Results sorted by edit_distance ascending, then canonical_id lexicographic (Req 12.4)
    - All returned candidates have edit_distance in {0, 1, 2}        (Req 12.2)

    Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        surface_form: str,
        entity_type: str,
        registry: "CanonicalRegistry",
    ) -> List[FuzzyCandidate]:
        """
        Find candidates within Levenshtein edit distance ≤ 2.

        Steps:
        1. Normalise the input surface form.
        2. Return [] immediately if len(normalised) < 4 code points.
        3. Query the registry for all (surface_form_normalized, canonical_id)
           pairs for the given entity_type.
        4. Compute edit distance for each pair; keep those with distance ≤ 2.
        5. Compute grounding_confidence for each kept candidate.
        6. Sort by edit_distance ASC, then canonical_id ASC.

        Returns:
            List of :class:`FuzzyCandidate` objects (may be empty).

        Requirements: 12.1, 12.2, 12.4, 12.5, 12.6
        """
        # Step 1: normalise
        normalised = normalize_surface_form(surface_form)

        # Step 2: short-form skip (Requirement 12.5)
        if len(normalised) < 4:
            return []

        len_surface = len(normalised)

        # Step 3: fetch all surface forms for the entity type from the registry
        all_forms = registry.get_all_surface_forms(entity_type=entity_type)

        # Step 4 & 5: compute distances and filter
        candidates: List[FuzzyCandidate] = []
        for candidate_normalised, canonical_id in all_forms:
            dist = _levenshtein(normalised, candidate_normalised)
            if dist <= 2:  # Requirement 12.2
                len_candidate = len(candidate_normalised)
                confidence = self.compute_confidence(dist, len_surface, len_candidate)
                candidates.append(
                    FuzzyCandidate(
                        canonical_id=canonical_id,
                        matched_surface_form=candidate_normalised,
                        edit_distance=dist,
                        grounding_confidence=confidence,
                    )
                )

        # Step 6: sort by edit_distance ASC, then canonical_id ASC (Requirement 12.4)
        candidates.sort(key=lambda c: (c.edit_distance, c.canonical_id))

        return candidates

    @staticmethod
    def compute_confidence(
        edit_distance: int,
        len_surface: int,
        len_candidate: int,
    ) -> float:
        """
        Compute grounding confidence for a fuzzy match.

        Formula:
            confidence = 1.0 - (edit_distance / max(len_surface, len_candidate)) * 0.5

        Special case: edit_distance=0 always yields 1.0 (Requirement 12.3).
        Lengths are measured in Unicode code points after normalisation.

        Args:
            edit_distance: Levenshtein distance between the two strings.
            len_surface:   Length of the normalised surface form in code points.
            len_candidate: Length of the normalised candidate in code points.

        Returns:
            Float in [0.0, 1.0].

        Requirements: 12.3
        """
        if edit_distance == 0:
            return 1.0

        max_len = max(len_surface, len_candidate)
        if max_len == 0:
            return 1.0  # both empty strings — treat as identical

        confidence = 1.0 - (edit_distance / max_len) * 0.5
        # Clamp to [0.0, 1.0] for safety
        return max(0.0, min(1.0, confidence))

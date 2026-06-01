"""
Property 13: Fuzzy Skip for Short Forms

**Validates: Requirements 12.5**

For any surface form where ``len(normalize_surface_form(S)) < 4`` Unicode code
points, ``FuzzyMatcher.match()`` must return ``[]`` without performing any edit
distance computation.

The test generates surface forms that normalise to fewer than 4 code points and
asserts that ``FuzzyMatcher.match()`` returns an empty list regardless of what
is in the registry.

Requirements: 12.5
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.fuzzy_matcher import FuzzyMatcher
from entity_resolution.utils import normalize_surface_form


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_registry(surface_forms: List[Tuple[str, str]]) -> MagicMock:
    """
    Return a minimal stub CanonicalRegistry whose ``get_all_surface_forms``
    returns the provided list of ``(surface_form_normalized, canonical_id)``
    pairs.

    Using a stub avoids the overhead of a full SQLite database while still
    exercising the real ``FuzzyMatcher.match()`` code path.
    """
    registry = MagicMock()
    registry.get_all_surface_forms.return_value = surface_forms
    return registry


# A small set of plausible registry entries to ensure the matcher has
# something to compare against (even though it should skip immediately).
_REGISTRY_ENTRIES: List[Tuple[str, str]] = [
    ("escherichia coli", "562"),
    ("bacteroides fragilis", "817"),
    ("crohns disease", "D003424"),
    ("16s rrna sequencing", "METHOD-16S"),
    ("metagenomics", "METHOD-META"),
]


# ---------------------------------------------------------------------------
# Strategies: generate surface forms that normalise to < 4 code points
# ---------------------------------------------------------------------------

# Strategy 1: generate strings of 0–3 printable ASCII characters.
# After NFC + lowercase + strip punctuation + collapse whitespace, these
# will have at most 3 code points (often fewer, since punctuation is stripped).
_short_ascii = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),  # letters and digits only
    ),
    min_size=0,
    max_size=3,
)

# Strategy 2: strings that consist entirely of punctuation characters.
# After stripping punctuation, the normalised form is empty (length 0).
_punct_only = st.text(
    alphabet=st.characters(whitelist_categories=("Po", "Ps", "Pe", "Pd", "Pc")),
    min_size=1,
    max_size=10,
)

# Strategy 3: strings of 1–3 Unicode letters (including non-ASCII).
# NFC normalisation may compose some sequences, but 1–3 base characters
# will always normalise to ≤ 3 code points.
_short_unicode_letters = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Lt", "Lo")),
    min_size=1,
    max_size=3,
)

# Strategy 4: whitespace-only strings.
# After collapsing whitespace, the normalised form is empty (length 0).
_whitespace_only = st.text(
    alphabet=st.characters(whitelist_categories=("Zs",)),
    min_size=1,
    max_size=10,
)

# Combined strategy: any of the above
_short_surface_form = st.one_of(
    _short_ascii,
    _punct_only,
    _short_unicode_letters,
    _whitespace_only,
).filter(lambda s: len(normalize_surface_form(s)) < 4)


# ---------------------------------------------------------------------------
# Property 13: Fuzzy Skip for Short Forms
# **Validates: Requirements 12.5**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(surface_form=_short_surface_form)
def test_property_fuzzy_skip_short_forms(surface_form: str) -> None:
    """
    **Property 13: Fuzzy Skip for Short Forms**

    **Validates: Requirements 12.5**

    For any surface form where ``len(normalize_surface_form(S)) < 4`` Unicode
    code points, ``FuzzyMatcher.match()`` must return ``[]``.

    The registry stub contains several real-looking entries to ensure the
    matcher would find candidates if it did not skip — confirming that the
    empty result is due to the short-form guard, not an empty registry.
    """
    # Precondition: the normalised form is indeed shorter than 4 code points
    normalised = normalize_surface_form(surface_form)
    assert len(normalised) < 4, (
        f"Strategy filter failed: '{surface_form}' normalises to "
        f"'{normalised}' (len={len(normalised)}), expected < 4"
    )

    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)

    result = matcher.match(surface_form, "taxon", registry)

    assert result == [], (
        f"Expected [] for short surface form '{surface_form}' "
        f"(normalised='{normalised}', len={len(normalised)}), "
        f"but got {result}"
    )


@settings(max_examples=100)
@given(
    surface_form=_short_surface_form,
    entity_type=st.sampled_from(["taxon", "disease", "method"]),
)
def test_property_fuzzy_skip_short_forms_all_entity_types(
    surface_form: str, entity_type: str
) -> None:
    """
    **Property 13: Fuzzy Skip for Short Forms (all entity types)**

    **Validates: Requirements 12.5**

    The short-form skip applies regardless of entity_type.  For any surface
    form where ``len(normalize_surface_form(S)) < 4``, ``match()`` returns
    ``[]`` for all entity types.
    """
    normalised = normalize_surface_form(surface_form)
    assert len(normalised) < 4

    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)

    result = matcher.match(surface_form, entity_type, registry)

    assert result == [], (
        f"Expected [] for short surface form '{surface_form}' "
        f"(normalised='{normalised}', len={len(normalised)}, "
        f"entity_type='{entity_type}'), but got {result}"
    )


@settings(max_examples=100)
@given(surface_form=_short_surface_form)
def test_property_fuzzy_skip_does_not_call_get_all_surface_forms(
    surface_form: str,
) -> None:
    """
    **Property 13: Fuzzy Skip for Short Forms — no edit distance computation**

    **Validates: Requirements 12.5**

    When the normalised surface form is shorter than 4 code points,
    ``FuzzyMatcher.match()`` must return ``[]`` *without* querying the
    registry for surface forms (i.e. without performing any edit distance
    computation).

    This is verified by asserting that ``registry.get_all_surface_forms``
    is never called when the short-form guard triggers.
    """
    normalised = normalize_surface_form(surface_form)
    assert len(normalised) < 4

    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)

    result = matcher.match(surface_form, "taxon", registry)

    assert result == []
    # The registry should NOT have been queried — no edit distance computation
    registry.get_all_surface_forms.assert_not_called()


# ---------------------------------------------------------------------------
# Boundary / edge-case unit tests
# ---------------------------------------------------------------------------


def test_empty_string_returns_empty() -> None:
    """Empty string normalises to '' (length 0) — must return []."""
    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)
    assert matcher.match("", "taxon", registry) == []


def test_single_char_returns_empty() -> None:
    """Single character normalises to length 1 — must return []."""
    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)
    assert matcher.match("a", "taxon", registry) == []


def test_two_chars_returns_empty() -> None:
    """Two characters normalise to length 2 — must return []."""
    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)
    assert matcher.match("ab", "taxon", registry) == []


def test_three_chars_returns_empty() -> None:
    """Three characters normalise to length 3 — must return []."""
    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)
    assert matcher.match("abc", "taxon", registry) == []


def test_punctuation_only_returns_empty() -> None:
    """Punctuation-only string normalises to '' — must return []."""
    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)
    assert matcher.match("...", "taxon", registry) == []


def test_whitespace_only_returns_empty() -> None:
    """Whitespace-only string normalises to '' — must return []."""
    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)
    assert matcher.match("   ", "taxon", registry) == []


def test_four_chars_does_not_skip() -> None:
    """
    A surface form that normalises to exactly 4 code points must NOT be
    skipped — the matcher should proceed to query the registry.

    We use a registry stub that returns no entries, so the result is still
    [] but for the right reason (no candidates), not the short-form guard.
    """
    matcher = FuzzyMatcher()
    # Empty registry — no candidates to match against
    registry = _make_stub_registry([])

    # "abcd" normalises to "abcd" (length 4) — should NOT trigger the skip
    result = matcher.match("abcd", "taxon", registry)

    # Result is [] because the registry is empty, but get_all_surface_forms
    # MUST have been called (the skip did not fire).
    registry.get_all_surface_forms.assert_called_once()
    assert result == []


def test_three_chars_with_punctuation_returns_empty() -> None:
    """
    'a.b' normalises to 'ab' (length 2 after stripping punctuation) — must
    return [] due to the short-form guard.
    """
    matcher = FuzzyMatcher()
    registry = _make_stub_registry(_REGISTRY_ENTRIES)
    normalised = normalize_surface_form("a.b")
    assert len(normalised) < 4  # sanity check
    assert matcher.match("a.b", "taxon", registry) == []

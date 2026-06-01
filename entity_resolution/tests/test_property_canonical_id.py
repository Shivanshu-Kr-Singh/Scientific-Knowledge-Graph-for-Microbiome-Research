"""
Property 9: Canonical ID Format Validation

**Validates: Requirements 3.2, 3.3, 3.4**

For each entity type, generate arbitrary strings and verify that
`validate_canonical_id` correctly accepts valid IDs and rejects invalid ones.
No partial records are created on failure (the function is pure validation).
"""

from __future__ import annotations

import re
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from entity_resolution.utils import validate_canonical_id

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Valid taxon IDs: positive integer strings (no leading zeros unless the value
# itself is "0", but "0" is not positive — so we use integers >= 1)
valid_taxon_ids = st.integers(min_value=1).map(str)

# Invalid taxon IDs: anything that is NOT a positive integer string.
# We cover: zero, negative integers, floats, empty string, arbitrary text,
# strings with leading zeros (e.g. "007"), and strings with whitespace.
invalid_taxon_ids = st.one_of(
    st.just("0"),
    st.just(""),
    st.just("-1"),
    st.just("1.5"),
    st.just("1e3"),
    st.just(" 1"),
    st.just("1 "),
    # Arbitrary text that is not a pure positive integer
    st.text(
        alphabet=st.characters(blacklist_categories=("Nd",)),  # no decimal digits
        min_size=1,
        max_size=20,
    ).filter(lambda s: s.strip() != ""),
    # Strings that look numeric but have leading zeros
    st.integers(min_value=1, max_value=9999).map(lambda n: f"0{n}"),
    # Negative integer strings
    st.integers(max_value=-1).map(str),
)

# Valid disease IDs: one uppercase ASCII letter followed by one or more digits
valid_disease_ids = st.builds(
    lambda letter, digits: letter + digits,
    letter=st.sampled_from(string.ascii_uppercase),
    digits=st.text(alphabet=string.digits, min_size=1, max_size=10),
)

# Invalid disease IDs: anything that does NOT match ^[A-Z]\d+$
_DISEASE_RE = re.compile(r"^[A-Z]\d+$")
invalid_disease_ids = st.one_of(
    st.just(""),
    st.just("d006262"),          # lowercase letter
    st.just("D"),                # no digits
    st.just("D06262X"),          # trailing non-digit
    st.just("1D06262"),          # starts with digit
    st.just("DD06262"),          # two letters
    st.just("D 06262"),          # space inside
    # Arbitrary text filtered to exclude valid patterns
    st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
        min_size=1,
        max_size=15,
    ).filter(lambda s: not _DISEASE_RE.match(s)),
)

# Valid method IDs: "METHOD-" followed by one or more alphanumeric characters
_ALNUM = string.ascii_letters + string.digits
valid_method_ids = st.text(alphabet=_ALNUM, min_size=1, max_size=20).map(
    lambda suffix: "METHOD-" + suffix
)

# Invalid method IDs: anything that does NOT match ^METHOD-[A-Za-z0-9]+$
_METHOD_RE = re.compile(r"^METHOD-[A-Za-z0-9]+$")
invalid_method_ids = st.one_of(
    st.just(""),
    st.just("METHOD-"),          # empty suffix
    st.just("method-16S"),       # lowercase prefix
    st.just("METHOD_16S"),       # underscore instead of hyphen
    st.just("METHOD-16S!"),      # non-alphanumeric suffix char
    st.just("METHOD-16 S"),      # space in suffix
    st.just("METH-16S"),         # wrong prefix
    # Arbitrary text filtered to exclude valid patterns
    st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")),
        min_size=1,
        max_size=25,
    ).filter(lambda s: not _METHOD_RE.match(s)),
)


# ---------------------------------------------------------------------------
# Property tests — valid IDs are accepted
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(canonical_id=valid_taxon_ids)
def test_valid_taxon_id_accepted(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.2**

    For any positive integer string, validate_canonical_id("taxon") returns True.
    """
    assert validate_canonical_id(canonical_id, "taxon") is True


@settings(max_examples=100)
@given(canonical_id=valid_disease_ids)
def test_valid_disease_id_accepted(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.3**

    For any string matching ^[A-Z]\\d+$, validate_canonical_id("disease") returns True.
    """
    assert validate_canonical_id(canonical_id, "disease") is True


@settings(max_examples=100)
@given(canonical_id=valid_method_ids)
def test_valid_method_id_accepted(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.4**

    For any string matching ^METHOD-[A-Za-z0-9]+$, validate_canonical_id("method") returns True.
    """
    assert validate_canonical_id(canonical_id, "method") is True


# ---------------------------------------------------------------------------
# Property tests — invalid IDs are rejected (no partial records created)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(canonical_id=invalid_taxon_ids)
def test_invalid_taxon_id_rejected(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.2**

    For any string that is not a positive integer, validate_canonical_id("taxon")
    returns False without raising an exception (no partial records created).
    """
    result = validate_canonical_id(canonical_id, "taxon")
    assert result is False


@settings(max_examples=100)
@given(canonical_id=invalid_disease_ids)
def test_invalid_disease_id_rejected(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.3**

    For any string that does not match ^[A-Z]\\d+$, validate_canonical_id("disease")
    returns False without raising an exception (no partial records created).
    """
    result = validate_canonical_id(canonical_id, "disease")
    assert result is False


@settings(max_examples=100)
@given(canonical_id=invalid_method_ids)
def test_invalid_method_id_rejected(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.4**

    For any string that does not match ^METHOD-[A-Za-z0-9]+$,
    validate_canonical_id("method") returns False without raising an exception
    (no partial records created).
    """
    result = validate_canonical_id(canonical_id, "method")
    assert result is False


# ---------------------------------------------------------------------------
# Property test — cross-type rejection (valid ID for one type is invalid for others)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(canonical_id=valid_taxon_ids)
def test_taxon_id_rejected_for_other_types(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.2, 3.3, 3.4**

    A valid taxon ID (positive integer string) must be rejected for disease and
    method entity types, since it does not match their patterns.
    """
    # A pure positive integer string cannot match ^[A-Z]\d+$ (no leading letter)
    # nor ^METHOD-[A-Za-z0-9]+$ (no METHOD- prefix).
    assert validate_canonical_id(canonical_id, "disease") is False
    assert validate_canonical_id(canonical_id, "method") is False


@settings(max_examples=100)
@given(canonical_id=valid_disease_ids)
def test_disease_id_rejected_for_other_types(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.2, 3.3, 3.4**

    A valid disease ID (e.g. "D006262") must be rejected for taxon and method types.
    """
    # Disease IDs start with a letter — not a positive integer.
    assert validate_canonical_id(canonical_id, "taxon") is False
    # Disease IDs don't start with "METHOD-".
    assert validate_canonical_id(canonical_id, "method") is False


@settings(max_examples=100)
@given(canonical_id=valid_method_ids)
def test_method_id_rejected_for_other_types(canonical_id: str) -> None:
    """
    **Validates: Requirements 3.2, 3.3, 3.4**

    A valid method ID (e.g. "METHOD-16S") must be rejected for taxon and disease types.
    """
    # Method IDs start with "METHOD-" — not a positive integer.
    assert validate_canonical_id(canonical_id, "taxon") is False
    # Method IDs don't match ^[A-Z]\d+$ (they have a hyphen and more letters).
    assert validate_canonical_id(canonical_id, "disease") is False


# ---------------------------------------------------------------------------
# Property test — unknown entity type always rejected
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    canonical_id=st.text(min_size=1, max_size=30),
    entity_type=st.text(min_size=1, max_size=20).filter(
        lambda t: t.lower() not in ("taxon", "disease", "method")
    ),
)
def test_unknown_entity_type_always_rejected(canonical_id: str, entity_type: str) -> None:
    """
    **Validates: Requirements 3.2, 3.3, 3.4**

    For any unknown entity type, validate_canonical_id returns False regardless
    of the canonical_id value.
    """
    assert validate_canonical_id(canonical_id, entity_type) is False

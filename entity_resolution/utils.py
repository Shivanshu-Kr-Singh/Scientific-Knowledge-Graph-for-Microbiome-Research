"""
Shared utility functions for the Deterministic Entity Resolution Pipeline.

Requirements: 2.1, 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import re
import string
import unicodedata


# ---------------------------------------------------------------------------
# Canonical ID validation
# ---------------------------------------------------------------------------

# Pre-compiled patterns for performance
_DISEASE_PATTERN = re.compile(r"^[A-Z]\d+$")
_METHOD_PATTERN = re.compile(r"^METHOD-[A-Za-z0-9]+$")


def validate_canonical_id(canonical_id: str, entity_type: str) -> bool:
    """
    Validate that a canonical_id conforms to the format rules for its entity type.

    Rules:
    - taxon:   canonical_id must be a positive integer string (e.g., "562")
    - disease: canonical_id must match ``^[A-Z]\\d+$`` (e.g., "D006262")
    - method:  canonical_id must match ``^METHOD-[A-Za-z0-9]+$`` (e.g., "METHOD-16S")

    Returns True if valid, False otherwise.

    Requirements: 3.2, 3.3, 3.4
    """
    if not canonical_id or not isinstance(canonical_id, str):
        return False

    entity_type_lower = entity_type.lower() if entity_type else ""

    if entity_type_lower == "taxon":
        # Must be a positive integer string with no surrounding whitespace,
        # no leading zeros, and no other non-digit characters.
        # Valid examples: "1", "562", "1000000"
        # Invalid: " 1", "01", "007", "0", "-1", "1.5"
        if not canonical_id.isdigit():
            return False
        # Reject leading zeros (e.g. "01", "007") — a canonical integer has no
        # leading zeros unless the value itself is "0", which is not positive.
        if len(canonical_id) > 1 and canonical_id[0] == "0":
            return False
        return int(canonical_id) > 0

    elif entity_type_lower == "disease":
        # Must match ^[A-Z]\d+$ (one uppercase letter followed by one or more digits)
        return bool(_DISEASE_PATTERN.match(canonical_id))

    elif entity_type_lower == "method":
        # Must match ^METHOD-[A-Za-z0-9]+$
        return bool(_METHOD_PATTERN.match(canonical_id))

    # Unknown entity type — reject
    return False


# ---------------------------------------------------------------------------
# Surface form normalisation
# ---------------------------------------------------------------------------

# Build a translation table that maps every punctuation character to None (remove it)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_surface_form(surface_form: str) -> str:
    """
    Normalise a surface form for consistent comparison.

    Steps applied in order:
    1. Unicode NFC normalisation
    2. Lowercase
    3. Strip punctuation (all characters in ``string.punctuation``)
    4. Collapse whitespace (multiple spaces → single space, strip leading/trailing)

    Returns the normalised string.

    Requirements: 2.1
    """
    if not surface_form:
        return surface_form

    # 1. Unicode NFC normalisation
    normalised = unicodedata.normalize("NFC", surface_form)

    # 2. Lowercase
    normalised = normalised.lower()

    # 3. Strip punctuation
    normalised = normalised.translate(_PUNCT_TABLE)

    # 4. Collapse whitespace
    normalised = " ".join(normalised.split())

    return normalised

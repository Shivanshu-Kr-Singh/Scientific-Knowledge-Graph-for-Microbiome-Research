"""
Property Tests for Embedding Model

**Validates: Requirements 1.3, 1.5**

Property 1: Embedding Model Output Consistency
  For any non-empty (title, abstract), the Embedding Model SHALL return a NumPy
  array of shape (dimension,) where dimension is fixed for a given model instance,
  all values are finite floats, and the vector has non-zero norm.

Property 2: Batch Encoding Equivalence
  For any list of texts and any batch size, encoding the list in one batch call
  SHALL produce vectors identical (within float32 tolerance) to encoding each text
  individually and stacking the results.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from collectors.embedding_model import EmbeddingModel


# ---------------------------------------------------------------------------
# Module-scoped fixture: load model ONCE for all tests in this module.
# Uses all-MiniLM-L6-v2 for speed (lightweight, already in project deps).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_model():
    """Load a lightweight embedding model once for the entire test module."""
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    return model


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-empty text for paper titles: at least 1 character, up to 200.
_title_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
        whitelist_characters=" -_.,;:()[]",
    ),
    min_size=1,
    max_size=200,
).filter(lambda s: len(s.strip()) > 0)

# Optional abstract: either None or a non-empty string.
_abstract_st = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(
            whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
            whitelist_characters=" -_.,;:()[]",
        ),
        min_size=1,
        max_size=200,
    ).filter(lambda s: len(s.strip()) > 0),
)

# Non-empty text for batch encoding tests.
_text_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
        whitelist_characters=" -_.,;:()[]",
    ),
    min_size=1,
    max_size=200,
).filter(lambda s: len(s.strip()) > 0)

# List of 1-10 non-empty texts for batch testing.
_text_list_st = st.lists(_text_st, min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Property 1: Embedding Model Output Consistency
# **Validates: Requirements 1.3**
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=30000)
@given(title=_title_st, abstract=_abstract_st)
def test_property_embedding_output_consistency(
    embedding_model: EmbeddingModel,
    title: str,
    abstract: str | None,
) -> None:
    """
    **Property 1: Embedding Model Output Consistency**

    **Validates: Requirements 1.3**

    For any non-empty (title, abstract), encode_paper returns:
    - Shape (dimension,) where dimension matches model.dimension
    - All values are finite (no NaN/Inf)
    - Non-zero L2 norm
    """
    embedding = embedding_model.encode_paper(title, abstract)

    # Shape must be (dimension,)
    assert embedding.shape == (embedding_model.dimension,), (
        f"Expected shape ({embedding_model.dimension},), got {embedding.shape}"
    )

    # All values must be finite (no NaN or Inf)
    assert np.all(np.isfinite(embedding)), (
        f"Embedding contains non-finite values. "
        f"NaN count: {np.isnan(embedding).sum()}, "
        f"Inf count: {np.isinf(embedding).sum()}"
    )

    # Non-zero L2 norm (embedding is meaningful, not a zero vector)
    norm = np.linalg.norm(embedding)
    assert norm > 0.0, (
        f"Embedding has zero norm for title={title!r}, abstract={abstract!r}"
    )


# ---------------------------------------------------------------------------
# Property 2: Batch Encoding Equivalence
# **Validates: Requirements 1.5**
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=30000)
@given(texts=_text_list_st)
def test_property_batch_encoding_equivalence(
    embedding_model: EmbeddingModel,
    texts: list[str],
) -> None:
    """
    **Property 2: Batch Encoding Equivalence**

    **Validates: Requirements 1.5**

    For any list of 1-10 non-empty strings, encoding the full list in one batch
    call produces the same result as encoding each individually and stacking
    (within float32 tolerance of 1e-5).
    """
    # Batch encode all texts at once
    batch_result = embedding_model.encode(texts)

    # Encode each text individually and stack
    individual_results = []
    for text in texts:
        single = embedding_model.encode([text])
        individual_results.append(single[0])
    stacked_result = np.stack(individual_results)

    # Shapes must match
    assert batch_result.shape == stacked_result.shape, (
        f"Shape mismatch: batch={batch_result.shape}, stacked={stacked_result.shape}"
    )

    # Values must be equal within float32 tolerance
    assert np.allclose(batch_result, stacked_result, atol=1e-5, rtol=1e-5), (
        f"Batch encoding differs from individual encoding. "
        f"Max absolute difference: {np.max(np.abs(batch_result - stacked_result))}"
    )

"""
Property Tests for Embedding Store

**Validates: Requirements 2.1, 2.2, 2.3, 2.6**

Property 3: Embedding Store Round-Trip
  For any valid embedding vector and metadata (doi, pmid, title), after appending
  to the store and reloading from disk, the retrieved vector SHALL be equal to
  the original (within float32 tolerance) and the metadata fields SHALL be identical.

Property 4: Partition Isolation
  For any embedding appended to the "positive" partition, querying the "negative"
  partition SHALL never return that embedding in results, and vice versa. The
  partitions are strictly disjoint.

Property 5: Cosine Similarity Correctness
  For any query vector and set of stored vectors in a partition, the top-k results
  returned by query_similar SHALL be the k vectors with the highest cosine similarity
  to the query, ordered descending by similarity score, where each score matches the
  manually computed cosine similarity within float32 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from collectors.embedding_store import EmbeddingStore, EmbeddingMetadata


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Float32 vector of dimension 384 with values in [-1, 1], no NaN/Inf.
# Use width=32 and allow_subnormal=False to avoid float64→float32 representation issues.
_vector_st = arrays(
    dtype=np.float32,
    shape=(384,),
    elements=st.floats(
        min_value=-1.0, max_value=1.0,
        allow_nan=False, allow_infinity=False,
        allow_subnormal=False, width=32,
    ),
)

# Non-empty title string
_title_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
        whitelist_characters=" -_.,;:()",
    ),
    min_size=1,
    max_size=100,
).filter(lambda s: len(s.strip()) > 0)

# Optional DOI string
_doi_st = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="./:-"),
        min_size=5,
        max_size=50,
    ).filter(lambda s: len(s.strip()) > 0),
)

# Optional PMID string (numeric-like)
_pmid_st = st.one_of(
    st.none(),
    st.from_regex(r"[0-9]{5,10}", fullmatch=True),
)

# Partition choice
_partition_st = st.sampled_from(["positive", "negative"])


# ---------------------------------------------------------------------------
# Property 3: Embedding Store Round-Trip
# **Validates: Requirements 2.1, 2.6**
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=10000)
@given(vector=_vector_st, title=_title_st, doi=_doi_st, pmid=_pmid_st)
def test_property_embedding_store_round_trip(
    tmp_path_factory,
    vector: np.ndarray,
    title: str,
    doi: str | None,
    pmid: str | None,
) -> None:
    """
    **Property 3: Embedding Store Round-Trip**

    **Validates: Requirements 2.1, 2.6**

    For any valid float32 vector (dim 384) and metadata (doi, pmid, title),
    appending to the store and creating a NEW store instance from the same dir
    yields identical vector (within float32 tolerance) and metadata fields.
    """
    # Skip zero-norm vectors since they won't produce meaningful results
    assume(np.linalg.norm(vector) > 1e-7)

    # Create a unique temp dir for this test instance
    store_dir = tmp_path_factory.mktemp("store_roundtrip")

    metadata = EmbeddingMetadata(
        doi=doi,
        pmid=pmid,
        title=title,
        partition="positive",
        added_at="2024-01-01T00:00:00Z",
    )

    # Append to store
    store1 = EmbeddingStore(store_dir=store_dir)
    store1.append(vector, metadata)

    # Create a NEW store instance from same directory (reload from disk)
    store2 = EmbeddingStore(store_dir=store_dir)

    # Verify the store reloaded the data
    assert store2.positive_count == 1, (
        f"Expected 1 embedding after reload, got {store2.positive_count}"
    )

    # Query with the same vector to retrieve it
    results = store2.query_similar(vector, partition="positive", top_k=1)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    result = results[0]

    # Vector should match within float32 tolerance (score should be ~1.0 for same vector)
    assert result.score >= 0.999, (
        f"Self-similarity score should be ~1.0, got {result.score}"
    )

    # Metadata fields must be identical
    assert result.metadata.doi == doi, (
        f"DOI mismatch: expected {doi!r}, got {result.metadata.doi!r}"
    )
    assert result.metadata.pmid == pmid, (
        f"PMID mismatch: expected {pmid!r}, got {result.metadata.pmid!r}"
    )
    assert result.metadata.title == title, (
        f"Title mismatch: expected {title!r}, got {result.metadata.title!r}"
    )
    assert result.metadata.partition == "positive", (
        f"Partition mismatch: expected 'positive', got {result.metadata.partition!r}"
    )


# ---------------------------------------------------------------------------
# Property 4: Partition Isolation
# **Validates: Requirements 2.2**
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=10000)
@given(vector=_vector_st, title=_title_st)
def test_property_partition_isolation(
    tmp_path_factory,
    vector: np.ndarray,
    title: str,
) -> None:
    """
    **Property 4: Partition Isolation**

    **Validates: Requirements 2.2**

    For any embedding appended to "positive", querying "negative" SHALL NOT
    return that embedding, and vice versa.
    """
    # Skip zero-norm vectors
    assume(np.linalg.norm(vector) > 1e-7)

    store_dir = tmp_path_factory.mktemp("store_isolation")
    store = EmbeddingStore(store_dir=store_dir)

    # Append vector to "positive" partition
    pos_meta = EmbeddingMetadata(
        doi="10.1000/pos-test",
        pmid="11111111",
        title=title,
        partition="positive",
        added_at="2024-01-01T00:00:00Z",
    )
    store.append(vector, pos_meta)

    # Query the "negative" partition — should return empty (no vectors there)
    neg_results = store.query_similar(vector, partition="negative", top_k=5)
    assert len(neg_results) == 0, (
        f"Positive vector appeared in negative results: got {len(neg_results)} results"
    )

    # Now append a different vector to "negative"
    neg_vector = -vector  # Use negated vector to ensure it's different
    neg_meta = EmbeddingMetadata(
        doi="10.1000/neg-test",
        pmid="22222222",
        title=f"Negative {title}",
        partition="negative",
        added_at="2024-01-01T00:00:00Z",
    )
    store.append(neg_vector, neg_meta)

    # Query positive partition — should only find the positive vector
    pos_results = store.query_similar(vector, partition="positive", top_k=5)
    assert len(pos_results) == 1, (
        f"Expected 1 positive result, got {len(pos_results)}"
    )
    assert pos_results[0].metadata.doi == "10.1000/pos-test", (
        f"Wrong vector returned from positive partition"
    )

    # Query negative partition — should only find the negative vector
    neg_results = store.query_similar(vector, partition="negative", top_k=5)
    assert len(neg_results) == 1, (
        f"Expected 1 negative result, got {len(neg_results)}"
    )
    assert neg_results[0].metadata.doi == "10.1000/neg-test", (
        f"Wrong vector returned from negative partition"
    )


# ---------------------------------------------------------------------------
# Property 5: Cosine Similarity Correctness
# **Validates: Requirements 2.3**
# ---------------------------------------------------------------------------


# Strategy for generating 5-20 random vectors (a matrix)
_num_vectors_st = st.integers(min_value=5, max_value=20)


@settings(max_examples=20, deadline=10000)
@given(
    num_vectors=_num_vectors_st,
    query_vector=_vector_st,
    data=st.data(),
)
def test_property_cosine_similarity_correctness(
    tmp_path_factory,
    num_vectors: int,
    query_vector: np.ndarray,
    data: st.DataObject,
) -> None:
    """
    **Property 5: Cosine Similarity Correctness**

    **Validates: Requirements 2.3**

    Insert 5-20 random vectors into one partition. Query with a random query
    vector. Assert:
    1. Results are in descending order by score
    2. Each score matches manually computed cosine similarity within tolerance
    """
    # Skip zero-norm query vectors
    assume(np.linalg.norm(query_vector) > 1e-7)

    store_dir = tmp_path_factory.mktemp("store_cosine")
    store = EmbeddingStore(store_dir=store_dir)

    # Generate and insert random vectors
    stored_vectors = []
    for i in range(num_vectors):
        vec = data.draw(
            arrays(
                dtype=np.float32,
                shape=(384,),
                elements=st.floats(
                    min_value=-1.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False,
                    allow_subnormal=False, width=32,
                ),
            ),
            label=f"vector_{i}",
        )
        # Skip zero-norm vectors
        if np.linalg.norm(vec) < 1e-7:
            continue

        meta = EmbeddingMetadata(
            doi=f"10.1000/paper-{i}",
            pmid=str(10000000 + i),
            title=f"Paper {i}",
            partition="positive",
            added_at="2024-01-01T00:00:00Z",
        )
        store.append(vec, meta)
        stored_vectors.append(vec)

    # Need at least 2 vectors to test ordering
    assume(len(stored_vectors) >= 2)

    # Query with top_k = all stored vectors
    top_k = len(stored_vectors)
    results = store.query_similar(query_vector, partition="positive", top_k=top_k)

    assert len(results) == len(stored_vectors), (
        f"Expected {len(stored_vectors)} results, got {len(results)}"
    )

    # 1. Assert descending order by score
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, (
            f"Results not in descending order: "
            f"results[{i}].score={results[i].score} < results[{i+1}].score={results[i+1].score}"
        )

    # 2. Assert each score matches manually computed cosine similarity
    query_norm = np.linalg.norm(query_vector)
    for result in results:
        # Find the matching stored vector by metadata
        idx = int(result.metadata.pmid) - 10000000
        stored_vec = stored_vectors[idx]

        # Manually compute cosine similarity
        vec_norm = np.linalg.norm(stored_vec)
        if vec_norm == 0:
            expected_cosine = 0.0
        else:
            dot_product = np.dot(query_vector, stored_vec)
            expected_cosine = float(dot_product / (query_norm * vec_norm))
            # Clamp to [-1, 1] as the store does
            expected_cosine = max(-1.0, min(1.0, expected_cosine))

        assert abs(result.score - expected_cosine) < 1e-5, (
            f"Score mismatch for paper {idx}: "
            f"returned={result.score}, expected={expected_cosine}, "
            f"diff={abs(result.score - expected_cosine)}"
        )

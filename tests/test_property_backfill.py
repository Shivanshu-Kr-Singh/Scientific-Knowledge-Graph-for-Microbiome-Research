"""
Property Tests for Backfill Seeding Script

**Validates: Requirements 4.2, 4.3, 4.4, 4.5**

Property 7: Backfill Partition Correctness
  For any set of audit records where each has a decision (keep/reject) and an
  abstract, the Backfill Script SHALL encode only records with non-empty abstracts,
  place kept papers in the positive partition and rejected papers in the negative
  partition, and the count of skipped records SHALL equal the count of records with
  empty or missing abstracts.

Property 8: Backfill Idempotence
  For any set of audit records, running the Backfill Script twice on identical
  input SHALL produce an Embedding Store of the same size after both runs — the
  second run adds zero new embeddings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for generating a non-empty abstract string
_abstract_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
        whitelist_characters=" -_.,;:()",
    ),
    min_size=5,
    max_size=200,
).filter(lambda s: len(s.strip()) > 0)

# Strategy for generating an empty or None abstract
_empty_abstract_st = st.one_of(
    st.none(),
    st.just(""),
    st.just("   "),  # whitespace-only counts as empty per _has_abstract()
)

# Strategy for generating a single audit record
_record_st = st.fixed_dictionaries(
    {
        "title": st.text(
            alphabet=st.characters(
                whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
                whitelist_characters=" -_.,;:()",
            ),
            min_size=3,
            max_size=80,
        ).filter(lambda s: len(s.strip()) > 0),
        "doi": st.from_regex(r"10\.[0-9]{4}/[a-z0-9]{5,10}", fullmatch=True),
        "pmid": st.from_regex(r"[0-9]{7,9}", fullmatch=True),
    }
)


def _record_with_abstract(has_abstract: bool):
    """Build a strategy for an audit record with or without abstract."""
    if has_abstract:
        return st.builds(
            lambda rec, abstract: {**rec, "abstract": abstract},
            _record_st,
            _abstract_st,
        )
    else:
        return st.builds(
            lambda rec, abstract: {**rec, "abstract": abstract} if abstract is not None else rec,
            _record_st,
            _empty_abstract_st,
        )


# Strategy for a list of kept records (mix of with/without abstract)
_kept_records_st = st.lists(
    st.one_of(
        _record_with_abstract(True),
        _record_with_abstract(False),
    ),
    min_size=1,
    max_size=8,
)

# Strategy for a list of rejected records (mix of with/without abstract)
_rejected_records_st = st.lists(
    st.one_of(
        _record_with_abstract(True),
        _record_with_abstract(False),
    ),
    min_size=1,
    max_size=8,
)


def _has_abstract(record: dict) -> bool:
    """Mirror the BackfillSeeder._has_abstract logic."""
    abstract = record.get("abstract")
    return bool(abstract and abstract.strip())


def _make_mock_embedding_model():
    """Create a mock EmbeddingModel that returns fixed-dimension random vectors."""
    mock_model = MagicMock()
    mock_model.dimension = 384

    def _encode_paper(title, abstract):
        # Use a deterministic seed based on title for reproducibility within a run,
        # but produce different vectors for different papers.
        seed = hash(title) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(384).astype(np.float32)

    mock_model.encode_paper = MagicMock(side_effect=_encode_paper)
    return mock_model


# ---------------------------------------------------------------------------
# Property 7: Backfill Partition Correctness
# **Validates: Requirements 4.2, 4.3, 4.4**
# ---------------------------------------------------------------------------


@settings(max_examples=15, deadline=30000)
@given(
    kept_records=_kept_records_st,
    rejected_records=_rejected_records_st,
)
def test_property_backfill_partition_correctness(
    tmp_path_factory,
    kept_records: list[dict],
    rejected_records: list[dict],
) -> None:
    """
    **Property 7: Backfill Partition Correctness**

    **Validates: Requirements 4.2, 4.3, 4.4**

    Generate audit records (mix of kept/rejected, some with abstracts, some without).
    Run BackfillSeeder. Assert:
    - positive_added = count of kept records with non-empty abstract
    - negative_added = count of rejected records with non-empty abstract
    - skipped_no_abstract = count of records with empty/None abstract
    """
    # Ensure at least one record has an abstract so the script doesn't sys.exit
    all_records = kept_records + rejected_records
    has_any_abstract = any(_has_abstract(r) for r in all_records)
    assume(has_any_abstract)

    # Ensure DOIs are unique across all records to avoid deduplication
    seen_dois = set()
    for record in all_records:
        if record["doi"] in seen_dois:
            assume(False)
        seen_dois.add(record["doi"])

    # Set up temp directory for audit files and embedding store
    base_dir = tmp_path_factory.mktemp("backfill_partition")
    audit_dir = base_dir / "audit"
    audit_dir.mkdir()
    store_dir = base_dir / "embeddings"
    store_dir.mkdir()

    # Write audit files
    with open(audit_dir / "kept.json", "w") as f:
        json.dump(kept_records, f)
    with open(audit_dir / "rejected.json", "w") as f:
        json.dump(rejected_records, f)
    with open(audit_dir / "llm_verified.json", "w") as f:
        json.dump([], f)  # Empty for this test

    # Expected counts
    expected_positive = sum(1 for r in kept_records if _has_abstract(r))
    expected_negative = sum(1 for r in rejected_records if _has_abstract(r))
    expected_skipped = sum(1 for r in all_records if not _has_abstract(r))

    # Create mock model and real store
    mock_model = _make_mock_embedding_model()

    from collectors.embedding_store import EmbeddingStore

    real_store = EmbeddingStore(store_dir=store_dir)

    # Patch and run BackfillSeeder
    with patch("scripts.seed_embedding_store.AUDIT_DIR", audit_dir), \
         patch("scripts.seed_embedding_store.EmbeddingModel", return_value=mock_model), \
         patch("scripts.seed_embedding_store.EmbeddingStore", return_value=real_store):

        from scripts.seed_embedding_store import BackfillSeeder

        seeder = BackfillSeeder()
        stats = seeder.run()

    # Assert partition correctness
    assert stats["positive_added"] == expected_positive, (
        f"positive_added: expected {expected_positive}, got {stats['positive_added']}"
    )
    assert stats["negative_added"] == expected_negative, (
        f"negative_added: expected {expected_negative}, got {stats['negative_added']}"
    )
    assert stats["skipped_no_abstract"] == expected_skipped, (
        f"skipped_no_abstract: expected {expected_skipped}, got {stats['skipped_no_abstract']}"
    )

    # Also verify the store contains the correct partition counts
    assert real_store.positive_count == expected_positive, (
        f"Store positive_count: expected {expected_positive}, got {real_store.positive_count}"
    )
    assert real_store.negative_count == expected_negative, (
        f"Store negative_count: expected {expected_negative}, got {real_store.negative_count}"
    )


# ---------------------------------------------------------------------------
# Property 8: Backfill Idempotence
# **Validates: Requirements 4.5**
# ---------------------------------------------------------------------------


@settings(max_examples=15, deadline=30000)
@given(
    kept_records=st.lists(
        _record_with_abstract(True),
        min_size=1,
        max_size=6,
    ),
    rejected_records=st.lists(
        _record_with_abstract(True),
        min_size=1,
        max_size=6,
    ),
)
def test_property_backfill_idempotence(
    tmp_path_factory,
    kept_records: list[dict],
    rejected_records: list[dict],
) -> None:
    """
    **Property 8: Backfill Idempotence**

    **Validates: Requirements 4.5**

    Generate audit records (all with abstracts), run BackfillSeeder twice.
    After the second run:
    - Store size should not increase
    - skipped_duplicate = number of records that were added in first run
    - positive_added == 0 on second run
    - negative_added == 0 on second run
    """
    # Ensure DOIs are unique across all records
    all_records = kept_records + rejected_records
    seen_dois = set()
    for record in all_records:
        if record["doi"] in seen_dois:
            assume(False)
        seen_dois.add(record["doi"])

    # Set up temp directory
    base_dir = tmp_path_factory.mktemp("backfill_idempotence")
    audit_dir = base_dir / "audit"
    audit_dir.mkdir()
    store_dir = base_dir / "embeddings"
    store_dir.mkdir()

    # Write audit files
    with open(audit_dir / "kept.json", "w") as f:
        json.dump(kept_records, f)
    with open(audit_dir / "rejected.json", "w") as f:
        json.dump(rejected_records, f)
    with open(audit_dir / "llm_verified.json", "w") as f:
        json.dump([], f)

    # Create mock model and a REAL store (to test actual deduplication)
    mock_model = _make_mock_embedding_model()

    from collectors.embedding_store import EmbeddingStore

    real_store = EmbeddingStore(store_dir=store_dir)

    # --- First run ---
    with patch("scripts.seed_embedding_store.AUDIT_DIR", audit_dir), \
         patch("scripts.seed_embedding_store.EmbeddingModel", return_value=mock_model), \
         patch("scripts.seed_embedding_store.EmbeddingStore", return_value=real_store):

        from scripts.seed_embedding_store import BackfillSeeder

        seeder1 = BackfillSeeder()
        stats1 = seeder1.run()

    size_after_first = real_store.positive_count + real_store.negative_count
    total_added_first = stats1["positive_added"] + stats1["negative_added"]

    # All records have abstracts, so all should be added
    assert total_added_first == len(all_records), (
        f"First run should add all {len(all_records)} records, "
        f"but added {total_added_first}"
    )

    # --- Second run (same data, same store) ---
    with patch("scripts.seed_embedding_store.AUDIT_DIR", audit_dir), \
         patch("scripts.seed_embedding_store.EmbeddingModel", return_value=mock_model), \
         patch("scripts.seed_embedding_store.EmbeddingStore", return_value=real_store):

        seeder2 = BackfillSeeder()
        stats2 = seeder2.run()

    size_after_second = real_store.positive_count + real_store.negative_count

    # Store size should NOT increase after second run
    assert size_after_second == size_after_first, (
        f"Store size changed after second run: "
        f"first={size_after_first}, second={size_after_second}"
    )

    # Second run should add zero new embeddings
    assert stats2["positive_added"] == 0, (
        f"Second run positive_added should be 0, got {stats2['positive_added']}"
    )
    assert stats2["negative_added"] == 0, (
        f"Second run negative_added should be 0, got {stats2['negative_added']}"
    )

    # Second run's skipped_duplicate should equal total records added in first run
    assert stats2["skipped_duplicate"] == total_added_first, (
        f"Second run skipped_duplicate should equal {total_added_first} "
        f"(all records from first run), got {stats2['skipped_duplicate']}"
    )

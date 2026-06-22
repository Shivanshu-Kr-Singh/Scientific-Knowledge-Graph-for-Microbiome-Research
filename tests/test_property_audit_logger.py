"""
Property 6: Audit Logger Field Completeness

**Validates: Requirements 3.1, 3.2, 3.3**

For any PaperRecord with doi, pmid, and abstract fields, the audit log record
produced by `AuditLogger.log()` SHALL contain the paper's DOI, PMID, and
abstract truncated to at most 2000 characters, while preserving all previously
existing fields (title, source, year, decision, stage, score, reason).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from collectors.audit_logger import AuditLogger


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-empty printable strings for paper text fields
_text_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z", "P")),
    min_size=1,
    max_size=100,
)

# DOI strings (simplified pattern)
_doi_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="/.:-"),
    min_size=5,
    max_size=80,
)

# PMID strings (numeric-ish)
_pmid_st = st.from_regex(r"[0-9]{1,10}", fullmatch=True)

# Abstract can be any length (including > 2000 chars) to test truncation
_abstract_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z", "P")),
    min_size=0,
    max_size=5000,
)

# Publication year
_year_st = st.integers(min_value=1900, max_value=2030)

# Verdict fields
_stage_st = st.sampled_from([
    "stage1_mesh", "stage2_rules", "stage3_ml", "stage3_5_embedding", "stage4_llm",
])
_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_reason_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    min_size=1,
    max_size=200,
)
_keep_st = st.booleans()
_review_st = st.booleans()


# ---------------------------------------------------------------------------
# Property 6: Audit Logger Field Completeness
# **Validates: Requirements 3.1, 3.2, 3.3**
# ---------------------------------------------------------------------------


@settings(max_examples=30)
@given(
    title=_text_st,
    source=_text_st,
    year=_year_st,
    doi=_doi_st,
    pmid=_pmid_st,
    abstract=_abstract_st,
    review=_review_st,
    keep=_keep_st,
    stage=_stage_st,
    score=_score_st,
    reason=_reason_st,
)
def test_property_audit_logger_field_completeness(
    title: str,
    source: str,
    year: int,
    doi: str,
    pmid: str,
    abstract: str,
    review: bool,
    keep: bool,
    stage: str,
    score: float,
    reason: str,
) -> None:
    """
    **Property 6: Audit Logger Field Completeness**

    **Validates: Requirements 3.1, 3.2, 3.3**

    For any PaperRecord with doi, pmid, and abstract fields, the audit log
    record produced by AuditLogger.log() SHALL contain the paper's DOI, PMID,
    and abstract truncated to at most 2000 characters, while preserving all
    previously existing fields (title, source, year, decision, stage, score,
    reason).
    """
    # Build mock paper and verdict as SimpleNamespace objects
    paper = SimpleNamespace(
        title=title,
        source=source,
        publication_year=year,
        doi=doi,
        pmid=pmid,
        abstract=abstract,
    )

    verdict = SimpleNamespace(
        review=review,
        keep=keep,
        stage=stage,
        score=score,
        reason=reason,
    )

    # Determine expected decision based on AuditLogger logic
    if review:
        expected_decision = "review"
    elif keep:
        expected_decision = "keep"
    else:
        expected_decision = "reject"

    # Use a fresh temporary directory for each invocation to isolate file writes
    tmp_dir = Path(tempfile.mkdtemp())

    # Patch AuditLogger.MAP to use tmp_dir so tests are isolated
    original_map = AuditLogger.MAP.copy()
    AuditLogger.MAP = {
        "keep": tmp_dir / "kept.json",
        "reject": tmp_dir / "rejected.json",
        "review": tmp_dir / "review.json",
        "llm": tmp_dir / "llm_verified.json",
    }

    try:
        AuditLogger.log(paper, verdict)

        # Read the written JSON file
        expected_path = AuditLogger.MAP[expected_decision]
        assert expected_path.exists(), f"Audit file not created at {expected_path}"

        with open(expected_path) as f:
            records = json.load(f)

        assert len(records) >= 1, "No records written"
        record = records[-1]  # Last record is the one we just wrote

        # ── Assert all existing fields are preserved ──────────────────────
        assert record["title"] == title, (
            f"title mismatch: expected {title!r}, got {record['title']!r}"
        )
        assert record["source"] == source, (
            f"source mismatch: expected {source!r}, got {record['source']!r}"
        )
        assert record["year"] == year, (
            f"year mismatch: expected {year}, got {record['year']}"
        )
        assert record["decision"] == expected_decision, (
            f"decision mismatch: expected {expected_decision!r}, got {record['decision']!r}"
        )
        assert record["stage"] == stage, (
            f"stage mismatch: expected {stage!r}, got {record['stage']!r}"
        )
        assert record["score"] == score, (
            f"score mismatch: expected {score}, got {record['score']}"
        )
        assert record["reason"] == reason, (
            f"reason mismatch: expected {reason!r}, got {record['reason']!r}"
        )

        # ── Assert new fields are present (Requirements 3.1, 3.2, 3.3) ───
        assert record["doi"] == doi, (
            f"doi mismatch: expected {doi!r}, got {record['doi']!r}"
        )
        assert record["pmid"] == pmid, (
            f"pmid mismatch: expected {pmid!r}, got {record['pmid']!r}"
        )

        # Abstract must be truncated to at most 2000 characters
        expected_abstract = abstract[:2000]
        assert record["abstract"] == expected_abstract, (
            f"abstract mismatch or not truncated properly.\n"
            f"  Expected length: {len(expected_abstract)}\n"
            f"  Got length: {len(record['abstract'])}"
        )
        assert len(record["abstract"]) <= 2000, (
            f"abstract exceeds 2000 chars: length={len(record['abstract'])}"
        )

    finally:
        # Restore original MAP to avoid cross-test pollution
        AuditLogger.MAP = original_map

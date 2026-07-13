"""
Unit tests for PredicateRegistry paper-frequency tracking and promotion detection.

Tests the new `track_paper_occurrence`, `get_promotion_threshold` methods,
and the `predicate_paper_occurrences` table.

Requirements: 4.1, 4.5
"""

import os
import sqlite3
import pytest
from unittest.mock import patch
from pathlib import Path

from graph.predicate_registry import PredicateRegistry, REGISTRY_DB_PATH


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    """Use a temporary database for each test to avoid side effects."""
    test_db = tmp_path / "predicate_registry.db"
    monkeypatch.setattr("graph.predicate_registry.REGISTRY_DB_PATH", test_db)
    yield test_db


@pytest.fixture
def registry(clean_db):
    """Create a fresh PredicateRegistry instance with a clean DB."""
    return PredicateRegistry()


class TestPredicatePaperOccurrencesTable:
    """Tests for the predicate_paper_occurrences table creation."""

    def test_table_exists(self, registry, clean_db):
        """The predicate_paper_occurrences table is created on init."""
        conn = sqlite3.connect(clean_db)
        cur = conn.cursor()
        cur.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='predicate_paper_occurrences'
        """)
        result = cur.fetchone()
        conn.close()
        assert result is not None
        assert result[0] == "predicate_paper_occurrences"

    def test_table_has_unique_constraint(self, registry, clean_db):
        """The table enforces UNIQUE(raw_predicate, paper_id)."""
        conn = sqlite3.connect(clean_db)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predicate_paper_occurrences (raw_predicate, paper_id)
            VALUES ('test_pred', 'paper1')
        """)
        conn.commit()
        # Inserting the same pair again should raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            cur.execute("""
                INSERT INTO predicate_paper_occurrences (raw_predicate, paper_id)
                VALUES ('test_pred', 'paper1')
            """)
        conn.close()


class TestGetPromotionThreshold:
    """Tests for get_promotion_threshold method."""

    def test_default_threshold_is_5(self, registry, monkeypatch):
        """Default promotion threshold is 5 when env var is not set."""
        monkeypatch.delenv('PREDICATE_PROMOTION_THRESHOLD', raising=False)
        assert registry.get_promotion_threshold() == 5

    def test_threshold_from_env_var(self, registry, monkeypatch):
        """Promotion threshold can be configured via environment variable."""
        monkeypatch.setenv('PREDICATE_PROMOTION_THRESHOLD', '10')
        assert registry.get_promotion_threshold() == 10

    def test_threshold_env_var_returns_int(self, registry, monkeypatch):
        """get_promotion_threshold returns an integer."""
        monkeypatch.setenv('PREDICATE_PROMOTION_THRESHOLD', '3')
        result = registry.get_promotion_threshold()
        assert isinstance(result, int)
        assert result == 3


class TestTrackPaperOccurrence:
    """Tests for track_paper_occurrence method."""

    def test_returns_tuple_of_three(self, registry):
        """track_paper_occurrence returns (canonical, is_known, is_newly_promoted)."""
        result = registry.track_paper_occurrence("some novel predicate xyz", "paper1")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_known_predicate_is_known_true(self, registry):
        """Known predicates return is_known=True."""
        canonical, is_known, is_newly_promoted = registry.track_paper_occurrence(
            "produces", "paper1"
        )
        assert canonical == "PRODUCES"
        assert is_known is True
        assert is_newly_promoted is False

    def test_novel_predicate_is_known_false(self, registry):
        """Novel predicates return is_known=False."""
        canonical, is_known, is_newly_promoted = registry.track_paper_occurrence(
            "xylophagous interaction", "paper1"
        )
        assert canonical == "RELATES_TO"
        assert is_known is False

    def test_idempotent_same_paper(self, registry, clean_db):
        """Calling with the same (predicate, paper_id) doesn't increment count."""
        registry.track_paper_occurrence("novel pred abc", "paper1")
        registry.track_paper_occurrence("novel pred abc", "paper1")
        registry.track_paper_occurrence("novel pred abc", "paper1")

        # Check the count is still 1
        conn = sqlite3.connect(clean_db)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM predicate_paper_occurrences
            WHERE raw_predicate = 'novel pred abc'
        """)
        count = cur.fetchone()[0]
        conn.close()
        assert count == 1

    def test_distinct_papers_counted(self, registry, clean_db):
        """Each distinct paper_id is counted once."""
        for i in range(4):
            registry.track_paper_occurrence("novel pred abc", f"paper{i}")

        conn = sqlite3.connect(clean_db)
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM predicate_paper_occurrences
            WHERE raw_predicate = 'novel pred abc'
        """)
        count = cur.fetchone()[0]
        conn.close()
        assert count == 4

    def test_promotion_at_threshold(self, registry, monkeypatch):
        """Predicate is newly promoted when threshold is reached."""
        monkeypatch.setenv('PREDICATE_PROMOTION_THRESHOLD', '3')

        # Papers 1 and 2 should not trigger promotion
        _, _, promoted1 = registry.track_paper_occurrence("novel pred xyz", "paper1")
        _, _, promoted2 = registry.track_paper_occurrence("novel pred xyz", "paper2")
        assert promoted1 is False
        assert promoted2 is False

        # Paper 3 reaches the threshold
        _, _, promoted3 = registry.track_paper_occurrence("novel pred xyz", "paper3")
        assert promoted3 is True

    def test_no_promotion_for_known_predicates(self, registry, monkeypatch):
        """Known predicates are never flagged as newly promoted."""
        monkeypatch.setenv('PREDICATE_PROMOTION_THRESHOLD', '1')

        canonical, is_known, is_newly_promoted = registry.track_paper_occurrence(
            "produces", "paper1"
        )
        assert canonical == "PRODUCES"
        assert is_known is True
        assert is_newly_promoted is False

    def test_not_newly_promoted_if_already_promoted(self, registry, clean_db, monkeypatch):
        """If predicate was already promoted, is_newly_promoted remains False."""
        monkeypatch.setenv('PREDICATE_PROMOTION_THRESHOLD', '2')

        # Reach threshold (newly promoted)
        registry.track_paper_occurrence("novel pred qrs", "paper1")
        _, _, promoted = registry.track_paper_occurrence("novel pred qrs", "paper2")
        assert promoted is True

        # Manually mark as promoted in DB
        conn = sqlite3.connect(clean_db)
        cur = conn.cursor()
        cur.execute("""
            UPDATE novel_predicates SET promoted = 1
            WHERE raw_predicate = 'novel pred qrs'
        """)
        conn.commit()
        conn.close()

        # Adding another paper should not flag as newly promoted
        _, _, promoted_again = registry.track_paper_occurrence("novel pred qrs", "paper3")
        assert promoted_again is False

    def test_predicate_normalized_to_lowercase(self, registry, clean_db):
        """raw_predicate is stored as lowercase in paper_occurrences."""
        registry.track_paper_occurrence("Novel Pred XYZ", "paper1")

        conn = sqlite3.connect(clean_db)
        cur = conn.cursor()
        cur.execute("SELECT raw_predicate FROM predicate_paper_occurrences")
        row = cur.fetchone()
        conn.close()
        assert row[0] == "novel pred xyz"

    def test_existing_methods_still_work(self, registry):
        """Existing normalize, get_category, get_novel_predicates methods still work."""
        # normalize
        canonical, is_known = registry.normalize("produces")
        assert canonical == "PRODUCES"
        assert is_known is True

        # get_category
        category = registry.get_category("PRODUCES")
        assert category == "biosynthetic"

        # get_novel_predicates (should return empty initially)
        novels = registry.get_novel_predicates(min_frequency=1)
        assert isinstance(novels, list)


class TestPromotePredicate:
    """Tests for promote_predicate method."""

    def test_returns_canonical_form(self, registry):
        """promote_predicate returns the canonical form string."""
        result = registry.promote_predicate("modulates mtor")
        assert result == "MODULATES_MTOR"

    def test_canonical_form_uppercase_underscores(self, registry):
        """Canonical form is uppercase with underscores replacing spaces."""
        canonical = registry.promote_predicate("xylophagous interaction")
        assert canonical == "XYLOPHAGOUS_INTERACTION"

    def test_canonical_form_hyphens_replaced(self, registry):
        """Hyphens are replaced with underscores in canonical form."""
        canonical = registry.promote_predicate("cross-reacts")
        assert canonical == "CROSS_REACTS"

    def test_sets_promoted_in_db(self, registry, clean_db):
        """promote_predicate sets promoted=1 in SQLite."""
        registry.promote_predicate("novel interaction xyz")

        conn = sqlite3.connect(clean_db)
        cur = conn.cursor()
        cur.execute("""
            SELECT promoted, canonical_form FROM novel_predicates
            WHERE raw_predicate = 'novel interaction xyz'
        """)
        row = cur.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1  # promoted
        assert row[1] == "NOVEL_INTERACTION_XYZ"

    def test_adds_to_normalization_dict(self, registry):
        """After promotion, the predicate is added to PREDICATE_NORMALIZATION."""
        from graph.predicate_registry import PREDICATE_NORMALIZATION

        registry.promote_predicate("novel pred abc")
        assert "novel pred abc" in PREDICATE_NORMALIZATION
        assert PREDICATE_NORMALIZATION["novel pred abc"] == "NOVEL_PRED_ABC"

    def test_normalize_returns_promoted_form(self, registry):
        """After promotion, normalize() returns the promoted canonical form (not RELATES_TO)."""
        registry.promote_predicate("modulates mtor")

        canonical, is_known = registry.normalize("modulates mtor")
        assert canonical == "MODULATES_MTOR"
        assert is_known is True

    def test_assigns_category_biosynthetic(self, registry):
        """Predicate with biosynthetic keywords gets 'biosynthetic' category."""
        from graph.predicate_registry import PREDICATE_CATEGORIES

        registry.promote_predicate("synthesizes butyrate")
        assert PREDICATE_CATEGORIES.get("SYNTHESIZES_BUTYRATE") == "biosynthetic"

    def test_assigns_category_regulatory(self, registry):
        """Predicate with regulatory keywords gets 'regulatory' category."""
        from graph.predicate_registry import PREDICATE_CATEGORIES

        registry.promote_predicate("modulates signaling")
        assert PREDICATE_CATEGORIES.get("MODULATES_SIGNALING") == "regulatory"

    def test_assigns_category_generic_for_unknown(self, registry):
        """Predicate with no matching keywords gets 'generic' category."""
        from graph.predicate_registry import PREDICATE_CATEGORIES

        registry.promote_predicate("xylophagous interaction")
        assert PREDICATE_CATEGORIES.get("XYLOPHAGOUS_INTERACTION") == "generic"

    def test_handles_leading_trailing_whitespace(self, registry):
        """Leading/trailing whitespace is stripped before promotion."""
        canonical = registry.promote_predicate("  some pred  ")
        assert canonical == "SOME_PRED"


class TestGetPromotedPredicates:
    """Tests for get_promoted_predicates method."""

    def test_returns_empty_list_initially(self, registry):
        """With no promoted predicates, returns empty list."""
        result = registry.get_promoted_predicates()
        assert result == []

    def test_returns_promoted_predicate(self, registry):
        """After promotion, the predicate appears in get_promoted_predicates."""
        registry.promote_predicate("modulates mtor")
        result = registry.get_promoted_predicates()
        assert len(result) == 1
        assert result[0]["raw_predicate"] == "modulates mtor"
        assert result[0]["canonical_form"] == "MODULATES_MTOR"

    def test_returns_category(self, registry):
        """get_promoted_predicates includes the category for each predicate."""
        registry.promote_predicate("synthesizes butyrate")
        result = registry.get_promoted_predicates()
        assert len(result) == 1
        assert result[0]["category"] == "biosynthetic"

    def test_multiple_promoted_predicates(self, registry):
        """Multiple promoted predicates are all returned."""
        registry.promote_predicate("synthesizes butyrate")
        registry.promote_predicate("modulates signaling")
        result = registry.get_promoted_predicates()
        assert len(result) == 2
        raw_preds = {r["raw_predicate"] for r in result}
        assert "synthesizes butyrate" in raw_preds
        assert "modulates signaling" in raw_preds

    def test_unpromoted_not_returned(self, registry):
        """Predicates that are not promoted are not returned."""
        registry.normalize("xylophagous interaction")  # logs but does not promote
        result = registry.get_promoted_predicates()
        assert len(result) == 0


class TestAssignCategory:
    """Tests for the _assign_category helper method."""

    def test_biosynthetic_keywords(self, registry):
        """Predicates with biosynthetic keywords map to 'biosynthetic'."""
        assert registry._assign_category("synthesizes something") == "biosynthetic"
        assert registry._assign_category("generates metabolites") == "biosynthetic"

    def test_regulatory_keywords(self, registry):
        """Predicates with regulatory keywords map to 'regulatory'."""
        assert registry._assign_category("activates pathway") == "regulatory"
        assert registry._assign_category("suppresses expression") == "regulatory"

    def test_clinical_keywords(self, registry):
        """Predicates with clinical keywords map to 'clinical'."""
        assert registry._assign_category("improves symptoms") == "clinical"
        assert registry._assign_category("treats disease") == "clinical"

    def test_generic_fallback(self, registry):
        """Predicates with no matching keywords default to 'generic'."""
        assert registry._assign_category("xyzzy foobar") == "generic"

    def test_handles_hyphens_and_underscores(self, registry):
        """Hyphens and underscores are treated as word separators."""
        assert registry._assign_category("cross-activates") == "regulatory"
        assert registry._assign_category("co_produces") == "biosynthetic"

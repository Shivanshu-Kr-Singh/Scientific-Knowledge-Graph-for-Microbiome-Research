"""
graph/predicate_registry.py
----------------------------
Open-world predicate registry for the scientific knowledge graph.

Maps raw predicate strings extracted from text to canonical predicate names.
Novel predicates not in the registry are stored as RELATES_TO with the raw
predicate string preserved as a property, allowing promotion to first-class
types when they accumulate sufficient evidence.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone
from loguru import logger

REGISTRY_DB_PATH = Path(__file__).parent / "predicate_registry.db"

# ── Canonical predicate normalization map ─────────────────────────────────────
# Maps raw predicate strings (lowercased) → canonical predicate name
# This covers the most common microbiome research predicates.
PREDICATE_NORMALIZATION: Dict[str, str] = {
    # Association / correlation
    "associated with": "ASSOCIATED_WITH",
    "is associated with": "ASSOCIATED_WITH",
    "correlates with": "CORRELATES_WITH",
    "correlates positively": "POSITIVELY_CORRELATES_WITH",
    "correlates negatively": "NEGATIVELY_CORRELATES_WITH",
    "linked to": "ASSOCIATED_WITH",
    "related to": "ASSOCIATED_WITH",

    # Abundance changes
    "increases": "INCREASES",
    "increased": "INCREASES",
    "elevates": "INCREASES",
    "elevated": "INCREASES",
    "upregulates": "UPREGULATES",
    "upregulated": "UPREGULATES",
    "enriched in": "ENRICHED_IN",
    "overrepresented in": "ENRICHED_IN",
    "more abundant": "ENRICHED_IN",
    "decreases": "DECREASES",
    "decreased": "DECREASES",
    "reduces": "DECREASES",
    "reduced": "DECREASES",
    "depleted in": "DEPLETED_IN",
    "underrepresented in": "DEPLETED_IN",
    "less abundant": "DEPLETED_IN",
    "downregulates": "DOWNREGULATES",
    "downregulated": "DOWNREGULATES",

    # Causal / mechanistic
    "produces": "PRODUCES",
    "synthesizes": "PRODUCES",
    "generates": "PRODUCES",
    "inhibits": "INHIBITS",
    "suppresses": "INHIBITS",
    "activates": "ACTIVATES",
    "promotes": "PROMOTES",
    "enhances": "PROMOTES",
    "modulates": "MODULATES",
    "regulates": "REGULATES",
    "mediates": "MEDIATES",
    "influences": "INFLUENCES",
    "affects": "INFLUENCES",
    "degrades": "DEGRADES",
    "metabolizes": "METABOLIZES",
    "ferments": "FERMENTS",
    "colonizes": "COLONIZES",

    # Clinical outcomes
    "improves": "IMPROVES",
    "worsens": "WORSENS",
    "treats": "TREATS",
    "prevents": "PREVENTS",
    "causes": "CAUSES",
    "predicts": "PREDICTS",
    "biomarker for": "BIOMARKER_FOR",

    # Intervention effects
    "administered to": "ADMINISTERED_TO",
    "supplemented with": "SUPPLEMENTED_WITH",
    "treated with": "TREATED_WITH",

    # Methodology
    "uses": "USES_METHODOLOGY",
    "sequenced with": "USES_METHODOLOGY",
    "analyzed by": "USES_METHODOLOGY",
    "measured by": "USES_METHODOLOGY",

    # Existing 3 canonical types (map to same values)
    "reports_association": "REPORTS_ASSOCIATION",
    "reports_intervention_effect": "REPORTS_INTERVENTION_EFFECT",
    "uses_methodology": "USES_METHODOLOGY",
}

# ── Predicate categories ──────────────────────────────────────────────────────
PREDICATE_CATEGORIES: Dict[str, str] = {
    "REPORTS_ASSOCIATION": "associative",
    "ASSOCIATED_WITH": "associative",
    "CORRELATES_WITH": "associative",
    "POSITIVELY_CORRELATES_WITH": "associative",
    "NEGATIVELY_CORRELATES_WITH": "associative",
    "ENRICHED_IN": "abundance",
    "DEPLETED_IN": "abundance",
    "INCREASES": "quantitative",
    "DECREASES": "quantitative",
    "UPREGULATES": "regulatory",
    "DOWNREGULATES": "regulatory",
    "PRODUCES": "biosynthetic",
    "METABOLIZES": "biosynthetic",
    "FERMENTS": "biosynthetic",
    "INHIBITS": "regulatory",
    "ACTIVATES": "regulatory",
    "PROMOTES": "regulatory",
    "MODULATES": "regulatory",
    "REGULATES": "regulatory",
    "MEDIATES": "mechanistic",
    "INFLUENCES": "mechanistic",
    "DEGRADES": "mechanistic",
    "COLONIZES": "ecological",
    "IMPROVES": "clinical",
    "WORSENS": "clinical",
    "TREATS": "clinical",
    "PREVENTS": "clinical",
    "CAUSES": "causal",
    "PREDICTS": "predictive",
    "BIOMARKER_FOR": "biomarker",
    "ADMINISTERED_TO": "intervention",
    "SUPPLEMENTED_WITH": "intervention",
    "TREATED_WITH": "intervention",
    "REPORTS_INTERVENTION_EFFECT": "intervention",
    "USES_METHODOLOGY": "methodology",
    "RELATES_TO": "generic",  # catch-all for novel predicates
}


class PredicateRegistry:
    """
    Registry for predicate normalization and novel predicate tracking.

    Normalizes raw predicate strings to canonical forms.
    Novel predicates are stored in SQLite with frequency counts.
    High-frequency novel predicates can be promoted to first-class types.
    """

    def __init__(self) -> None:
        self._init_db()

    def _init_db(self) -> None:
        try:
            conn = sqlite3.connect(REGISTRY_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS novel_predicates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_predicate TEXT NOT NULL UNIQUE,
                    frequency INTEGER NOT NULL DEFAULT 1,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    example_subject TEXT,
                    example_object TEXT,
                    promoted BOOLEAN DEFAULT 0,
                    canonical_form TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predicate_paper_occurrences (
                    raw_predicate TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    UNIQUE(raw_predicate, paper_id)
                )
            """)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("PredicateRegistry: could not initialize DB — {}", exc)

    def normalize(self, raw_predicate: str) -> Tuple[str, bool]:
        """
        Normalize a raw predicate string to its canonical form.

        Returns:
            (canonical_predicate, is_known) tuple.
            If not in registry: returns ("RELATES_TO", False) and logs the novel predicate.
        """
        if not raw_predicate or not raw_predicate.strip():
            return "RELATES_TO", False

        lower = raw_predicate.lower().strip()

        # Direct lookup
        if lower in PREDICATE_NORMALIZATION:
            return PREDICATE_NORMALIZATION[lower], True

        # Partial match — check if any known predicate is contained in the raw
        for known, canonical in PREDICATE_NORMALIZATION.items():
            if known in lower or lower in known:
                return canonical, True

        # Novel predicate — log and return RELATES_TO
        self._log_novel(raw_predicate)
        return "RELATES_TO", False

    def get_category(self, canonical_predicate: str) -> str:
        """Return the category for a canonical predicate."""
        return PREDICATE_CATEGORIES.get(canonical_predicate, "generic")

    def _log_novel(self, raw_predicate: str, subject: str = "", object_: str = "") -> None:
        """Log a novel predicate to the registry DB."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn = sqlite3.connect(REGISTRY_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO novel_predicates
                    (raw_predicate, frequency, first_seen, last_seen, example_subject, example_object)
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(raw_predicate) DO UPDATE SET
                    frequency = frequency + 1,
                    last_seen = excluded.last_seen
            """, (raw_predicate.lower().strip(), now, now, subject or None, object_ or None))
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("PredicateRegistry: could not log novel predicate — {}", exc)

    def get_novel_predicates(self, min_frequency: int = 2) -> List[Dict]:
        """Return novel predicates with frequency >= min_frequency for review."""
        try:
            conn = sqlite3.connect(REGISTRY_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                SELECT raw_predicate, frequency, first_seen, last_seen
                FROM novel_predicates
                WHERE frequency >= ? AND promoted = 0
                ORDER BY frequency DESC
            """, (min_frequency,))
            rows = cur.fetchall()
            conn.close()
            return [
                {"raw_predicate": r[0], "frequency": r[1], "first_seen": r[2], "last_seen": r[3]}
                for r in rows
            ]
        except Exception:
            return []

    def get_promotion_threshold(self) -> int:
        """
        Return the configured promotion threshold.
        Reads PREDICATE_PROMOTION_THRESHOLD env var, defaults to 5.
        """
        return int(os.environ.get('PREDICATE_PROMOTION_THRESHOLD', '10'))

    # ── Category keyword map for semantic similarity assignment ──────────────
    _CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "biosynthetic": [
            "produce", "produces", "synthesize", "synthesizes", "generate",
            "generates", "ferment", "ferments", "metabolize", "metabolizes",
        ],
        "regulatory": [
            "inhibit", "inhibits", "suppress", "suppresses", "activate",
            "activates", "promote", "promotes", "enhance", "enhances",
            "modulate", "modulates", "regulate", "regulates", "upregulate",
            "upregulates", "downregulate", "downregulates",
        ],
        "abundance": [
            "enrich", "enriched", "deplete", "depleted", "abundant",
            "overrepresented", "underrepresented",
        ],
        "quantitative": [
            "increase", "increases", "decrease", "decreases", "elevate",
            "elevates", "reduce", "reduces",
        ],
        "clinical": [
            "improve", "improves", "worsen", "worsens", "treat", "treats",
            "prevent", "prevents", "cure", "cures",
        ],
        "causal": [
            "cause", "causes", "induce", "induces", "trigger", "triggers",
        ],
        "associative": [
            "associate", "associated", "correlate", "correlates", "link",
            "linked", "relate", "related",
        ],
        "mechanistic": [
            "mediate", "mediates", "influence", "influences", "affect",
            "affects", "degrade", "degrades",
        ],
        "ecological": [
            "colonize", "colonizes", "inhabit", "inhabits", "dominate",
            "dominates",
        ],
        "predictive": [
            "predict", "predicts", "biomarker", "indicate", "indicates",
        ],
        "intervention": [
            "administer", "administered", "supplement", "supplemented",
            "treated",
        ],
        "methodology": [
            "use", "uses", "sequence", "sequenced", "analyze", "analyzed",
            "measure", "measured",
        ],
    }

    def _assign_category(self, raw_predicate: str) -> str:
        """
        Assign a predicate category via semantic similarity (keyword matching).

        Compares words in the raw predicate against category keyword lists.
        Returns the best matching category, or "generic" if no match found.
        """
        words = raw_predicate.lower().replace("_", " ").replace("-", " ").split()

        best_category = "generic"
        best_score = 0

        for category, keywords in self._CATEGORY_KEYWORDS.items():
            score = sum(1 for word in words if word in keywords)
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    def promote_predicate(self, raw_predicate: str) -> str:
        """
        Promote a novel predicate to first-class status.
        - Sets promoted=1 in SQLite
        - Assigns canonical form (uppercase, underscores)
        - Adds to PREDICATE_NORMALIZATION mapping
        - Assigns a category via semantic similarity

        Quality gates reject predicates that are clearly not scientific relationships:
        - Longer than 60 characters (likely a sentence fragment from a small LLM)
        - Pure stop-words with no scientific content
        - Contain metadata markers (author, doi, title, etc.)

        Returns the new canonical form, or "RELATES_TO" if rejected.
        """
        normalized_pred = raw_predicate.lower().strip()

        # ── Quality gates ─────────────────────────────────────────────────────
        # Reject predicates longer than 60 chars — always sentence fragments
        if len(normalized_pred) > 60:
            logger.debug(
                "PredicateRegistry: rejected promotion of long predicate ({} chars): '{}'",
                len(normalized_pred), normalized_pred[:60]
            )
            return "RELATES_TO"

        # Reject single-word stop predicates with no scientific meaning
        _STOP_PREDICATES = {
            "are", "is", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "the", "a", "an", "in", "on", "at", "to", "of", "and", "or",
        }
        if normalized_pred in _STOP_PREDICATES:
            logger.debug(
                "PredicateRegistry: rejected stop-word predicate: '{}'", normalized_pred
            )
            return "RELATES_TO"

        # Reject metadata predicates (paper authorship, bibliographic info)
        _METADATA_MARKERS = {"author", "doi", "title", "journal", "published", "cited", "reference"}
        if any(marker in normalized_pred for marker in _METADATA_MARKERS):
            logger.debug(
                "PredicateRegistry: rejected metadata predicate: '{}'", normalized_pred
            )
            return "RELATES_TO"

        # Generate canonical form: uppercase, replace spaces/hyphens with underscores
        canonical_form = normalized_pred.upper().replace(" ", "_").replace("-", "_")

        # Assign category via semantic similarity to existing categories
        category = self._assign_category(normalized_pred)

        # Update SQLite: mark as promoted with canonical form
        # If the predicate doesn't already exist in novel_predicates, insert it first
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn = sqlite3.connect(REGISTRY_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO novel_predicates
                    (raw_predicate, frequency, first_seen, last_seen, promoted, canonical_form)
                VALUES (?, 1, ?, ?, 1, ?)
                ON CONFLICT(raw_predicate) DO UPDATE SET
                    promoted = 1,
                    canonical_form = excluded.canonical_form
            """, (normalized_pred, now, now, canonical_form))
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("PredicateRegistry: could not promote predicate in DB — {}", exc)

        # Add to runtime PREDICATE_NORMALIZATION dict
        PREDICATE_NORMALIZATION[normalized_pred] = canonical_form

        # Add to PREDICATE_CATEGORIES
        PREDICATE_CATEGORIES[canonical_form] = category

        logger.info(
            "PredicateRegistry: promoted '{}' → '{}' (category: {})",
            normalized_pred, canonical_form, category,
        )

        return canonical_form

    def get_promoted_predicates(self) -> List[Dict]:
        """Return all promoted predicates with their canonical forms and categories."""
        try:
            conn = sqlite3.connect(REGISTRY_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                SELECT raw_predicate, canonical_form
                FROM novel_predicates
                WHERE promoted = 1
            """)
            rows = cur.fetchall()
            conn.close()
            return [
                {
                    "raw_predicate": r[0],
                    "canonical_form": r[1],
                    "category": PREDICATE_CATEGORIES.get(r[1], "generic"),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("PredicateRegistry: could not get promoted predicates — {}", exc)
            return []

    def track_paper_occurrence(self, raw_predicate: str, paper_id: str) -> Tuple[str, bool, bool]:
        """
        Track that a paper uses a predicate and check for promotion.

        Inserts the (raw_predicate, paper_id) pair into predicate_paper_occurrences
        using INSERT OR IGNORE for idempotency. Then counts distinct papers for
        this predicate and checks if the promotion threshold has been reached.

        Returns:
            (canonical_predicate, is_known, is_newly_promoted)
        """
        # First normalize the predicate
        canonical, is_known = self.normalize(raw_predicate)

        normalized_predicate = raw_predicate.lower().strip()

        try:
            conn = sqlite3.connect(REGISTRY_DB_PATH)
            cur = conn.cursor()

            # Insert the paper occurrence (idempotent via UNIQUE constraint)
            cur.execute("""
                INSERT OR IGNORE INTO predicate_paper_occurrences (raw_predicate, paper_id)
                VALUES (?, ?)
            """, (normalized_predicate, paper_id))

            # Count distinct papers for this predicate
            cur.execute("""
                SELECT COUNT(*) FROM predicate_paper_occurrences
                WHERE raw_predicate = ?
            """, (normalized_predicate,))
            paper_count = cur.fetchone()[0]

            # Check if predicate is already promoted
            cur.execute("""
                SELECT promoted FROM novel_predicates
                WHERE raw_predicate = ?
            """, (normalized_predicate,))
            row = cur.fetchone()
            already_promoted = row[0] == 1 if row else False

            conn.commit()
            conn.close()

            # Determine if newly promoted (threshold reached AND not already promoted)
            threshold = self.get_promotion_threshold()
            is_newly_promoted = (
                paper_count >= threshold
                and not is_known
                and not already_promoted
            )

            return (canonical, is_known, is_newly_promoted)

        except Exception as exc:
            logger.warning("PredicateRegistry: could not track paper occurrence — {}", exc)
            return (canonical, is_known, False)

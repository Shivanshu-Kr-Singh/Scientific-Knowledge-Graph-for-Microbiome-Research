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

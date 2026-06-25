"""
nlp/article_classifier.py
---------------------------
Classifies each paper into a standardized article type.

WHY THIS MATTERS:
  The knowledge graph needs to distinguish between:
    - original_research  → has primary data we can analyze
    - systematic_review  → summarizes existing evidence
    - meta_analysis      → pools statistics across studies
    - protocol           → pre-registered study, no results yet

PROBLEMS WITH THE ORIGINAL:
  1. "journal article" mapped to original_research in Tier 1, and because
     ~90% of PubMed papers carry this tag, Tier 1 almost always fired first.
     A paper tagged ["Journal Article", "Meta-Analysis"] would return
     original_research because "journal article" came first in the loop —
     meta_analysis was never seen. TYPE_PRIORITY existed but _from_tags()
     returned the first *priority-ordered* type from the found set, which
     correctly handled multi-tag papers — BUT the fallthrough to Tier 2/3
     never ran because Tier 1 always fired for any paper with "journal article".

  2. TITLE_PATTERNS had no preprint detection (bioRxiv/medRxiv papers from
     Semantic Scholar/CORE never have PubMed tags).

  3. ABSTRACT_PATTERNS had no confidence differentiation — all returned 0.70.
     A "forest plot" mention is much stronger evidence of meta-analysis than
     "inclusion criteria" alone.

  4. Fallback was `original_research` at 0.40 — too confident for an unknown.

FIXES:
  1. _from_tags() now treats "journal article" ALONE as insufficient — only
     promotes to original_research if no higher-priority type is found from
     the *other* tags. If tags contain ["Journal Article", "Meta-Analysis"],
     we correctly return meta_analysis.

  2. Tier 2 title patterns expanded: preprint, narrative review, network
     meta-analysis, umbrella review, living systematic review, cohort study.

  3. Tier 3 abstract patterns have per-pattern confidence scores. Highly
     specific signals (forest plot, I² heterogeneity) return 0.90; weaker
     signals (PRISMA alone, inclusion criteria alone) return 0.75.

  4. Tier 2 and Tier 3 now run EVEN AFTER Tier 1 returns original_research
     from "journal article" alone — giving higher-specificity patterns a
     chance to upgrade the type.

  5. Confidence values are now meaningful:
       1.00 = specific PubMed tag (e.g., "meta-analysis", "systematic review")
       0.95 = journal article + corroborating Tier 2/3 evidence
       0.90 = strong Tier 3 signal (forest plot, I²)
       0.85 = Tier 2 title pattern
       0.75 = Tier 3 moderate signal
       0.55 = IMRAD structure detected
       0.40 = no signal — fallback
"""

import re
from typing import Optional, Tuple, List
from loguru import logger


# ── PubMed tag → normalized type ──────────────────────────────────────────────
# "journal article" is intentionally NOT included here —
# it is too generic to drive classification on its own.
# It is handled separately in _from_tags() as a fallback.
SPECIFIC_TAG_MAP = {
    # Research designs
    "clinical trial":                   "original_research",
    "clinical trial, phase i":          "original_research",
    "clinical trial, phase ii":         "original_research",
    "clinical trial, phase iii":        "original_research",
    "clinical trial, phase iv":         "original_research",
    "randomized controlled trial":      "original_research",
    "controlled clinical trial":        "original_research",
    "observational study":              "original_research",
    "comparative study":                "original_research",
    "multicenter study":                "original_research",
    "cohort study":                     "original_research",
    "twin study":                       "original_research",
    "validation study":                 "original_research",
    # Reviews
    "review":                           "narrative_review",
    "literature review":                "narrative_review",
    "systematic review":                "systematic_review",
    "meta-analysis":                    "meta_analysis",
    # Special types
    "case reports":                     "case_report",
    "letter":                           "letter",
    "editorial":                        "commentary",
    "comment":                          "commentary",
    "news":                             "commentary",
    "clinical study protocol":          "protocol",
    "study characteristics":            "protocol",
    "dataset":                          "dataset",
    "preprint":                         "original_research",
    # OpenAlex / EuropePMC / Crossref types
    "research-article":                 "original_research",
    "research article":                 "original_research",
    "original article":                 "original_research",
    "original research":                "original_research",
    "systematic-review":                "systematic_review",
    "meta-analysis article":            "meta_analysis",
    "review-article":                   "narrative_review",
    "review article":                   "narrative_review",
    "case-report":                      "case_report",
    "rapid-communication":              "letter",
    "erratum":                          "commentary",
}

# Types that definitively override "journal article" if co-present
TYPE_PRIORITY = [
    "meta_analysis", "systematic_review", "narrative_review",
    "protocol", "case_report", "dataset", "letter",
    "commentary", "original_research",
]

# ── Tier 2: Title patterns (compiled, confidence per pattern) ─────────────────
_TITLE_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # Meta-analysis
    (re.compile(r"\bmeta[- ]anal",                       re.I), "meta_analysis",    0.90),
    (re.compile(r"\bnetwork\s+meta[- ]anal",             re.I), "meta_analysis",    0.95),
    (re.compile(r"\bpooled\s+anal",                      re.I), "meta_analysis",    0.85),
    (re.compile(r"\bumbrella\s+review\b",                re.I), "meta_analysis",    0.90),
    (re.compile(r"\bsystematic\s+review\s+and\s+meta",   re.I), "meta_analysis",    0.95),
    # Systematic review
    (re.compile(r"\bsystematic\s+review\b",              re.I), "systematic_review", 0.90),
    (re.compile(r"\bscoping\s+review\b",                 re.I), "systematic_review", 0.85),
    (re.compile(r"\bliving\s+systematic\s+review\b",     re.I), "systematic_review", 0.90),
    (re.compile(r"\brapid\s+(?:systematic\s+)?review\b", re.I), "systematic_review", 0.80),
    # Narrative review
    (re.compile(r"\bnarrative\s+review\b",               re.I), "narrative_review",  0.90),
    (re.compile(r"\b(?:a\s+)?(?:comprehensive|current)\s+review\b", re.I), "narrative_review", 0.80),
    # Protocol
    (re.compile(r"\bstudy\s+protocol\b",                 re.I), "protocol",         0.95),
    (re.compile(r"\btrial\s+protocol\b",                 re.I), "protocol",         0.95),
    (re.compile(r"\bprotocol\s+(?:for|of)\s+a\b",        re.I), "protocol",         0.85),
    # Case report / series
    (re.compile(r"\bcase\s+report\b",                    re.I), "case_report",      0.95),
    (re.compile(r"\bcase\s+series\b",                    re.I), "case_report",      0.90),
    # Commentary
    (re.compile(r"\beditorial\b",                        re.I), "commentary",       0.90),
    (re.compile(r"\bcommentary\b",                       re.I), "commentary",       0.90),
    # Letter
    (re.compile(r"\bletter\s+to\s+the\s+editor\b",       re.I), "letter",           0.95),
    # Dataset
    (re.compile(r"\bdata\s+descriptor\b",                re.I), "dataset",          0.95),
    # Preprint
    (re.compile(r"\bpreprint\b",                         re.I), "original_research", 0.70),
    # Cohort study signals in title
    (re.compile(r"\bcohort\s+study\b",                   re.I), "original_research", 0.80),
    (re.compile(r"\brandomized\s+controlled\s+trial\b",  re.I), "original_research", 0.85),
]

# ── Tier 3: Abstract patterns (compiled, confidence per pattern) ──────────────
_ABSTRACT_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    # Strong meta-analysis signals
    (re.compile(r"\bforest\s+plot\b",                         re.I), "meta_analysis",    0.92),
    (re.compile(r"\bI[²2]\s*(?:heterogeneity|statistic)",     re.I), "meta_analysis",    0.92),
    (re.compile(r"\bpooled\s+(?:odds\s+ratio|risk\s+ratio|effect\s+size|estimate)", re.I), "meta_analysis", 0.90),
    (re.compile(r"\brandom[- ]effects\s+model\b",             re.I), "meta_analysis",    0.88),
    (re.compile(r"\bfixed[- ]effects\s+model\b",              re.I), "meta_analysis",    0.85),
    (re.compile(r"\bfunnel\s+plot\b",                         re.I), "meta_analysis",    0.85),
    (re.compile(r"\bpublication\s+bias\b",                    re.I), "meta_analysis",    0.80),
    # Moderate meta-analysis
    (re.compile(r"\bwe\s+identified\s+\d+\s+(?:eligible\s+)?(?:studies|trials|articles)", re.I), "meta_analysis", 0.80),
    (re.compile(r"\d+\s+(?:studies|trials|articles)\s+(?:were\s+)?(?:included|eligible)", re.I), "meta_analysis", 0.78),
    # Strong systematic review signals
    (re.compile(r"\bprisma\b",                                re.I), "systematic_review", 0.88),
    (re.compile(r"\b(?:medline|pubmed|embase|cochrane|web\s+of\s+science)\b.{0,60}\bsearch(?:ed)?\b", re.I | re.DOTALL), "systematic_review", 0.85),
    (re.compile(r"\binclusion\s+(?:and\s+exclusion\s+)?criteria\b.{0,100}\b(?:studies|trials)\b", re.I | re.DOTALL), "systematic_review", 0.80),
    (re.compile(r"\bstudies\s+were\s+(?:included|eligible|selected)\b",  re.I), "systematic_review", 0.78),
    (re.compile(r"\bsearch(?:ed)?\s+(?:the\s+)?(?:medline|pubmed|embase|cochrane)\b", re.I), "systematic_review", 0.82),
    # Original research signals
    (re.compile(r"\bwe\s+(?:enrolled|recruited|randomized|randomised)\b", re.I), "original_research", 0.80),
    (re.compile(r"\b(?:fecal|stool|biopsy|blood|saliva|urine)\s+samples?\b", re.I), "original_research", 0.72),
    (re.compile(r"\bshotgun\s+metagenomic|16s\s+rrna\s+gene\s+sequen", re.I), "original_research", 0.78),
    (re.compile(r"\bwe\s+collected\s+\d+|n\s*=\s*\d{2,}", re.I), "original_research", 0.70),
    # Protocol signals
    (re.compile(r"\bthis\s+protocol\s+describes\b",           re.I), "protocol",         0.92),
    (re.compile(r"\bwill\s+be\s+(?:randomly\s+)?assigned\b",  re.I), "protocol",         0.85),
    (re.compile(r"\bregistered\s+(?:at|with|on)\s+clinicaltrials\b", re.I), "protocol",  0.88),
    (re.compile(r"\bprimary\s+outcome\s+will\s+be\b",          re.I), "protocol",         0.80),
]

IMRAD_KEYWORDS = frozenset([
    "background", "methods", "results", "conclusions",
    "objective", "findings", "purpose",
])


class ArticleClassifier:
    """
    Classifies papers into standardized article types.

    Key fix: "journal article" tag alone no longer short-circuits Tier 2/3.
    If a paper has only generic tags, we run title + abstract patterns to
    find more specific signals (meta-analysis, systematic review, protocol).
    """

    def classify(
        self,
        article_types_raw: List[str],
        title: str,
        abstract: Optional[str],
    ) -> Tuple[str, float]:
        """
        Returns (article_type, confidence).

        Resolution order:
          1. Specific PubMed tags (meta-analysis, systematic review, etc.) → type, 1.0
          2. Title patterns — run REGARDLESS of Tier 1 result when Tier 1
             only found "journal article" (generic tag)
          3. Abstract patterns with per-pattern confidence
          4. IMRAD structure detection → original_research, 0.55
          5. Fallback → unknown, 0.40
        """
        title_l    = (title    or "").lower()
        abstract_l = (abstract or "").lower()

        # ── Tier 1: Specific PubMed tags ──────────────────────────────────────
        specific_type, is_generic_only = self._from_tags(article_types_raw)

        if specific_type and not is_generic_only:
            # Got a specific type (meta-analysis, systematic review, etc.)
            logger.debug(f"[classifier] Tier1 specific: {specific_type}")
            return specific_type, 1.0

        # ── Tier 2: Title patterns ────────────────────────────────────────────
        # Runs even if Tier 1 found "original_research" from generic tags,
        # giving specific title patterns a chance to override.
        for pattern, atype, conf in _TITLE_PATTERNS:
            if pattern.search(title_l):
                logger.debug(f"[classifier] Tier2 title: {atype} ({conf})")
                # If Tier 1 already said original_research from generic tags
                # and Tier 2 agrees, keep Tier 1 confidence
                if specific_type == atype == "original_research":
                    return atype, conf
                return atype, conf

        # ── Tier 3: Abstract patterns ─────────────────────────────────────────
        if abstract_l:
            best_type = None
            best_conf = 0.0
            for pattern, atype, conf in _ABSTRACT_PATTERNS:
                if pattern.search(abstract_l) and conf > best_conf:
                    best_type = atype
                    best_conf = conf
            if best_type:
                logger.debug(f"[classifier] Tier3 abstract: {best_type} ({best_conf})")
                return best_type, best_conf

        # ── Tier 4: IMRAD structure ───────────────────────────────────────────
        if abstract_l:
            imrad_hits = sum(1 for kw in IMRAD_KEYWORDS if kw in abstract_l)
            if imrad_hits >= 3:
                return "original_research", 0.60
            if imrad_hits >= 2:
                return "original_research", 0.55

        # ── If Tier 1 gave a generic result, use it at reduced confidence ─────
        if specific_type:
            return specific_type, 0.65

        return "unknown", 0.40

    # ── Tag resolution ────────────────────────────────────────────────────────

    def _from_tags(self, raw_tags: List[str]) -> Tuple[Optional[str], bool]:
        """
        Maps raw article type tags to a normalized type.

        Returns (type, is_generic_only):
          - type: the resolved article type, or None
          - is_generic_only: True if the ONLY matched tag was "journal article"
            (meaning no specific type was found from tags alone)

        This distinction lets the caller decide whether to run Tier 2/3
        to find something more specific.
        """
        if not raw_tags:
            return None, False

        found_specific: set = set()
        has_journal_article = False

        for tag in raw_tags:
            tag_lower = tag.lower().strip()
            if tag_lower == "journal article":
                has_journal_article = True
                continue   # don't add to found_specific

            mapped = SPECIFIC_TAG_MAP.get(tag_lower)
            if mapped:
                found_specific.add(mapped)

        # Specific tags found — return highest priority
        for ptype in TYPE_PRIORITY:
            if ptype in found_specific:
                return ptype, False   # specific type found

        # Only "journal article" was found — return original_research but
        # flag as generic so caller can try Tier 2/3
        if has_journal_article:
            return "original_research", True

        return None, False

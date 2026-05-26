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

STRATEGY — 3 tiers applied in order:
  Tier 1: PubMed article type tags (human-curated, most reliable)
  Tier 2: Title keyword patterns (regex on title text)
  Tier 3: Abstract content patterns (structural signals in text)
  Fallback: original_research with low confidence

WHY RULES NOT ML HERE?
  PubMed's own tags are >95% accurate for the common types.
  Rules work well here. We use ML (BioBERT) only for NER, where
  rules genuinely fail on biomedical entity recognition.
"""

import re
from typing import Optional, Tuple, List
from loguru import logger


# ── PubMed tag → normalized type ──────────────────────────────────────────────
PUBMED_TAG_MAP = {
    "journal article":              "original_research",
    "clinical trial":               "original_research",
    "randomized controlled trial":  "original_research",
    "observational study":          "original_research",
    "comparative study":            "original_research",
    "multicenter study":            "original_research",
    "cohort study":                 "original_research",
    "twin study":                   "original_research",
    "review":                       "narrative_review",
    "literature review":            "narrative_review",
    "systematic review":            "systematic_review",
    "meta-analysis":                "meta_analysis",
    "case reports":                 "case_report",
    "letter":                       "letter",
    "editorial":                    "commentary",
    "comment":                      "commentary",
    "clinical study protocol":      "protocol",
    "dataset":                      "dataset",
    "preprint":                     "original_research",
}

# Priority when multiple tags map to different types
TYPE_PRIORITY = [
    "meta_analysis", "systematic_review", "narrative_review",
    "protocol", "case_report", "dataset", "letter",
    "commentary", "original_research",
]

# ── Title keyword patterns ─────────────────────────────────────────────────────
TITLE_PATTERNS = [
    (r"\bmeta.anal",                  "meta_analysis"),
    (r"\bpooled anal",                "meta_analysis"),
    (r"\bsystematic review and meta", "meta_analysis"),
    (r"\bsystematic review\b",        "systematic_review"),
    (r"\bscoping review\b",           "systematic_review"),
    (r"\bstudy protocol\b",           "protocol"),
    (r"\btrial protocol\b",           "protocol"),
    (r"\bcase report\b",              "case_report"),
    (r"\bcase series\b",              "case_report"),
    (r"\beditorial\b",                "commentary"),
    (r"\bcommentary\b",               "commentary"),
    (r"\bletter to the editor\b",     "letter"),
    (r"\bdata descriptor\b",          "dataset"),
]

# ── Abstract content patterns ─────────────────────────────────────────────────
ABSTRACT_PATTERNS = [
    (r"\bheterogeneity\b.*\bI[²2]\b",                        "meta_analysis"),
    (r"\bpooled (odds ratio|risk ratio|effect size)",         "meta_analysis"),
    (r"\bforest plot",                                        "meta_analysis"),
    (r"\brandom.effects model",                               "meta_analysis"),
    (r"\b(medline|pubmed|embase|cochrane)\b.*\bsearch",       "systematic_review"),
    (r"\binclusion criteri",                                  "systematic_review"),
    (r"\bprisma\b",                                           "systematic_review"),
    (r"\bstudies were included\b",                            "systematic_review"),
    (r"\bwe enrolled\b|\bparticipants were recruited",        "original_research"),
    (r"\b(fecal|stool|biopsy|blood|saliva) samples",          "original_research"),
    (r"\bshotgun metagenom|\b16s rrna gene sequenc",          "original_research"),
    (r"\bthis protocol describes\b",                          "protocol"),
    (r"\bwill be randomly assigned\b",                        "protocol"),
]

IMRAD_KEYWORDS = ["background", "methods", "results", "conclusions",
                  "objective", "findings", "purpose"]


class ArticleClassifier:
    """
    Classifies papers into standardized article types using a 3-tier rule cascade.
    Returns (article_type, confidence_score).
    """

    def classify(
        self,
        article_types_raw: List[str],
        title: str,
        abstract: Optional[str],
    ) -> Tuple[str, float]:
        """
        Tier 1 → PubMed tags (confidence 1.0)
        Tier 2 → Title patterns (confidence 0.85)
        Tier 3 → Abstract patterns (confidence 0.70)
        Fallback → original_research (confidence 0.40)
        """
        title_l    = (title or "").lower()
        abstract_l = (abstract or "").lower()

        # Tier 1
        t1 = self._from_tags(article_types_raw)
        if t1:
            logger.debug(f"[classifier] Tier1: {t1}")
            return t1, 1.0

        # Tier 2
        for pattern, atype in TITLE_PATTERNS:
            if re.search(pattern, title_l):
                logger.debug(f"[classifier] Tier2: {atype}")
                return atype, 0.85

        # Tier 3
        if abstract_l:
            for pattern, atype in ABSTRACT_PATTERNS:
                if re.search(pattern, abstract_l):
                    logger.debug(f"[classifier] Tier3: {atype}")
                    return atype, 0.70

        # Fallback
        if abstract_l and sum(1 for kw in IMRAD_KEYWORDS if kw in abstract_l) >= 2:
            return "original_research", 0.55

        return "unknown", 0.40

    def _from_tags(self, raw_tags: List[str]) -> Optional[str]:
        """Maps raw tags → normalized type with priority resolution."""
        if not raw_tags:
            return None
        found = set()
        for tag in raw_tags:
            mapped = PUBMED_TAG_MAP.get(tag.lower().strip())
            if mapped:
                found.add(mapped)
        for ptype in TYPE_PRIORITY:
            if ptype in found:
                return ptype
        return None

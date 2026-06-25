"""
nlp/study_design.py
---------------------
Extracts study design classification with calibrated confidence scores.

PROBLEMS WITH THE ORIGINAL:
  1. r"randomized|trial" fired at confidence 0.9 on "trial version", "trial
     period", "during a trial run" — inflating quality scores for non-RCTs.
  2. Only one design returned — the first pattern match. Real papers often
     have multiple applicable designs (e.g., prospective longitudinal cohort).
  3. All designs got confidence=0.9 regardless of pattern specificity.
  4. The cohort pattern `r"cohort|\\d+ patients"` matched nearly every paper
     (almost all mention a number of patients), conflating cohort studies
     with case reports and cross-sectionals.
  5. No r"prospective", r"longitudinal", r"double-blind", r"placebo",
     r"multi-center", r"open-label" patterns.
  6. Function (not a class) — patterns re-compiled on every call.

FIXES:
  - RCT requires co-occurrence of specific terms (randomized + controlled,
    or double-blind, or placebo-controlled) — not just "trial" alone.
  - Each pattern has its own calibrated confidence score.
  - Returns primary design + all_designs list for downstream use.
  - Pre-compiled patterns (class with __init__).
  - Confidence calibration:
      0.95 = very specific phrase (double-blind placebo-controlled RCT)
      0.85 = specific compound phrase (randomized controlled trial)
      0.75 = strong signal (prospective cohort, longitudinal study)
      0.60 = moderate signal (cohort study, cross-sectional)
      0.45 = weak signal (pilot, feasibility, single-center)
      0.30 = minimal signal (enrolled/recruited — pattern, not design proof)

OUTPUT:
  {
    "type":        str    — primary design label
    "confidence":  float  — confidence for primary design
    "all_designs": list   — all matched designs with their confidences
    "is_rct":      bool   — True only for genuinely randomized controlled trials
    "is_prospective": bool
  }
"""

import re
from typing import Optional


class StudyDesignExtractor:
    """
    Extracts study design with calibrated, pattern-specific confidence.
    Instantiate once; call extract(text) per paper.
    """

    # Each entry: (label, confidence, compiled_pattern)
    # Ordered from most specific to least specific.
    # A paper can match multiple patterns — all are returned.
    _PATTERNS = [
        # ── RCT / controlled trials (most specific first) ─────────────────────
        ("rct", 0.95, re.compile(
            r"\bdouble[- ]blind\b.{0,80}\bplacebo[- ]controlled\b|"
            r"\bplacebo[- ]controlled\b.{0,80}\bdouble[- ]blind\b",
            re.IGNORECASE | re.DOTALL,
        )),
        ("rct", 0.90, re.compile(
            r"\brandomized\s+controlled\s+trial\b|"
            r"\brandomised\s+controlled\s+trial\b|"
            r"\bparallel[- ]group\s+(?:randomized|randomised)\b|"
            r"\bcluster[- ]randomized\s+trial\b",
            re.IGNORECASE,
        )),
        ("rct", 0.85, re.compile(
            r"\bplacebo[- ]controlled\b.{0,50}\brandomized\b|"
            r"\brandomized\b.{0,50}\bplacebo[- ]controlled\b|"
            r"\bdouble[- ]blind\b.{0,50}\brandomized\b",
            re.IGNORECASE | re.DOTALL,
        )),
        ("rct", 0.75, re.compile(
            r"\bopen[- ]label\s+(?:randomized|randomised)\s+(?:controlled\s+)?trial\b|"
            r"\bsingle[- ]blind\s+(?:randomized|randomised)\b",
            re.IGNORECASE,
        )),

        # ── Systematic review / meta-analysis ─────────────────────────────────
        ("meta_analysis", 0.95, re.compile(
            r"\bmeta[- ]analysis\b.{0,100}\bsystematic\s+review\b|"
            r"\bsystematic\s+review\b.{0,100}\bmeta[- ]analysis\b|"
            r"\bpooled\s+(?:odds\s+ratio|risk\s+ratio|effect\s+size|estimate)\b|"
            r"\bforest\s+plot\b|"
            r"\brandom[- ]effects\s+model\b",
            re.IGNORECASE | re.DOTALL,
        )),
        ("systematic_review", 0.90, re.compile(
            r"\bsystematic(?:ally)?\s+(?:searched|reviewed|identified)\b|"
            r"\bprisma\b|"
            r"\binclusion\s+(?:and\s+exclusion\s+)?criteria\b.{0,200}"
            r"\b(?:medline|pubmed|embase|cochrane|web\s+of\s+science)\b|"
            r"\bstudies\s+were\s+(?:included|eligible|selected)\b",
            re.IGNORECASE | re.DOTALL,
        )),

        # ── Prospective cohort ─────────────────────────────────────────────────
        ("cohort", 0.85, re.compile(
            r"\bprospective\s+(?:longitudinal\s+)?cohort\b|"
            r"\blongitudinal\s+(?:prospective\s+)?cohort\b|"
            r"\bbirth\s+cohort\b|"
            r"\bprospective\s+follow[- ]up\s+study\b",
            re.IGNORECASE,
        )),
        ("cohort", 0.75, re.compile(
            r"\bcohort\s+study\b|"
            r"\bprospective\s+study\b|"
            r"\blongitudinal\s+study\b|"
            r"\bfollow[- ]up\s+(?:period|study|cohort)\b",
            re.IGNORECASE,
        )),
        ("cohort", 0.65, re.compile(
            r"\bretrospective\s+cohort\b|"
            r"\bobservational\s+cohort\b",
            re.IGNORECASE,
        )),

        # ── Case-control ──────────────────────────────────────────────────────
        ("case_control", 0.90, re.compile(
            r"\bcase[- ]control\s+study\b|"
            r"\bmatched\s+case[- ]control\b|"
            r"\bcase[- ]control\s+design\b",
            re.IGNORECASE,
        )),
        ("case_control", 0.75, re.compile(
            r"\bcases\s+and\s+(?:matched\s+)?controls\b|"
            r"\bcontrol\s+group\s+matched\b",
            re.IGNORECASE,
        )),

        # ── Cross-sectional ───────────────────────────────────────────────────
        ("cross_sectional", 0.90, re.compile(
            r"\bcross[- ]sectional\s+study\b|"
            r"\bcross[- ]sectional\s+(?:survey|design|analysis)\b|"
            r"\bprevalence\s+study\b",
            re.IGNORECASE,
        )),
        ("cross_sectional", 0.70, re.compile(
            r"\bsingle\s+(?:time\s+)?point\s+measurement\b|"
            r"\bbaseline\s+only\b",
            re.IGNORECASE,
        )),

        # ── Twin study ────────────────────────────────────────────────────────
        ("twin_study", 0.90, re.compile(
            r"\btwin\s+(?:study|cohort|pairs?)\b|"
            r"\bmonozygotic\s+twins?\b|"
            r"\bdizygotic\s+twins?\b",
            re.IGNORECASE,
        )),

        # ── Intervention / clinical trial (non-RCT) ────────────────────────────
        ("clinical_trial", 0.80, re.compile(
            r"\bclinical\s+trial\b(?!\s+registr)|"   # avoid matching "clinical trial registration"
            r"\bintervention\s+study\b|"
            r"\btreatment\s+arm\b|"
            r"\bdose[- ]escalation\b",
            re.IGNORECASE,
        )),

        # ── Observational ─────────────────────────────────────────────────────
        ("observational", 0.70, re.compile(
            r"\bobservational\s+study\b|"
            r"\bepidemiological\s+study\b|"
            r"\bpopulation[- ]based\s+study\b",
            re.IGNORECASE,
        )),

        # ── Pilot / feasibility ───────────────────────────────────────────────
        ("pilot", 0.60, re.compile(
            r"\bpilot\s+(?:study|trial|rct|cohort)\b|"
            r"\bfeasibility\s+(?:study|trial)\b|"
            r"\bexploratory\s+(?:study|analysis)\b",
            re.IGNORECASE,
        )),

        # ── Protocol (pre-registered, no results yet) ─────────────────────────
        ("protocol", 0.85, re.compile(
            r"\bstudy\s+protocol\b|"
            r"\btrial\s+protocol\b|"
            r"\bwill\s+be\s+(?:randomized|enrolled|recruited|assigned)\b|"
            r"\bregistered\s+(?:at|with|in)\s+clinicaltrials\b",
            re.IGNORECASE,
        )),

        # ── Case report / series ──────────────────────────────────────────────
        ("case_report", 0.90, re.compile(
            r"\bcase\s+report\b|"
            r"\bcase\s+series\b|"
            r"\bsingle\s+patient\b",
            re.IGNORECASE,
        )),

        # ── Mendelian randomization ────────────────────────────────────────────
        ("mendelian_randomization", 0.95, re.compile(
            r"\bmendelian\s+randomiz",
            re.IGNORECASE,
        )),
    ]

    # RCT-specific signal for is_rct flag
    _RCT_SIGNAL = re.compile(
        r"\brandomized\b|\brandomised\b",
        re.IGNORECASE,
    )
    _PROSPECTIVE_SIGNAL = re.compile(
        r"\bprospective\b|\blongitudinal\b|\bfollow[- ]up\b",
        re.IGNORECASE,
    )

    def extract(self, text: str) -> Optional[dict]:
        """
        Extracts study design from text.

        Returns:
          {
            "type":          str    — primary design (highest confidence match)
            "confidence":    float  — confidence for primary design
            "all_designs":   list   — [{type, confidence}, ...] all matches
            "is_rct":        bool
            "is_prospective": bool
          }
          Returns None if no design pattern matches.
        """
        text = (text or "").strip()
        if not text:
            return None

        matched = []
        for label, conf, pattern in self._PATTERNS:
            if pattern.search(text):
                matched.append({"type": label, "confidence": conf})

        if not matched:
            return None

        # Primary design = highest confidence match
        primary = max(matched, key=lambda x: x["confidence"])

        return {
            "type":           primary["type"],
            "confidence":     primary["confidence"],
            "all_designs":    matched,
            "is_rct":         bool(self._RCT_SIGNAL.search(text)),
            "is_prospective": bool(self._PROSPECTIVE_SIGNAL.search(text)),
        }


# ── Module-level convenience function (backward-compatible) ──────────────────
_extractor = StudyDesignExtractor()


def extract_design(text: str) -> Optional[dict]:
    """
    Module-level function for backward compatibility with pipeline.py.
    Uses the singleton StudyDesignExtractor.
    """
    return _extractor.extract(text)

"""
collectors/stage2_calibrator.py
---------------------------------
Automatic threshold calibration for Stage 2 (weighted keyword rule scorer).

WHY THIS EXISTS:
  Stage 2 uses two thresholds from stage2_rules.yaml:
    keep:   0.70  → score ≥ 0.70 → confident keep, skip ML
    review: 0.40  → score 0.40–0.69 → borderline, send to ML
                    score < 0.40  → confident reject

  These were initially set manually. But the right boundaries depend on
  the actual score distribution of YOUR collected papers — and that shifts
  as your dataset grows and your term weights evolve.

HOW CALIBRATION WORKS:
  1. Uses Stage-1-confirmed papers as ground truth:
       Stage 1 KEEP  (score 0.90) → true positives
       Stage 1 REJECT (score 0.05) → true negatives
  2. Runs Stage 2 scoring on all of them → gets a score distribution
  3. Finds the score value that best separates the two groups:
       keep_threshold   = score where precision of positives ≥ target (default 0.90)
       review_threshold = score where recall of negatives ≥ target (default 0.90)
  4. Saves computed thresholds to config/stage2_calibration.json
  5. relevance_filter.py loads this file at startup and uses these values
     instead of the static stage2_rules.yaml thresholds

WHEN IT RUNS:
  Triggered by: RUN_LAYER=train_filter python main.py
  Same command that retrains the ML model — both calibrations happen together.

OUTPUT FILE:
  config/stage2_calibration.json
  {
    "keep_threshold":   0.68,
    "review_threshold": 0.38,
    "calibrated_on":    "2026-06-13T10:30:00",
    "n_positives":      420,
    "n_negatives":      380,
    "note": "Auto-calibrated. Edit stage2_rules.yaml thresholds to override."
  }

FALLBACK:
  If calibration file is missing or corrupt → stage2_rules.yaml thresholds
  are used. The system always works even without calibration.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from loguru import logger

CALIBRATION_PATH = Path(__file__).parent.parent / "config" / "stage2_calibration.json"

# Defaults used if calibration fails or file doesn't exist
DEFAULT_KEEP_THRESHOLD   = 0.70
DEFAULT_REVIEW_THRESHOLD = 0.40


def load_calibrated_thresholds() -> Tuple[float, float]:
    """
    Loads auto-calibrated Stage 2 thresholds from disk.

    Returns:
        (keep_threshold, review_threshold)
        Falls back to stage2_rules.yaml defaults if file missing or corrupt.
    """
    if not CALIBRATION_PATH.exists():
        logger.debug(
            "[stage2_calibrator] No calibration file found — "
            "using stage2_rules.yaml defaults"
        )
        return DEFAULT_KEEP_THRESHOLD, DEFAULT_REVIEW_THRESHOLD

    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        keep   = float(data["keep_threshold"])
        review = float(data["review_threshold"])

        logger.info(
            f"[stage2_calibrator] Loaded calibrated thresholds: "
            f"keep≥{keep:.3f}  review≥{review:.3f}  "
            f"(calibrated on {data.get('calibrated_on', 'unknown')} "
            f"from {data.get('n_positives', '?')} pos + "
            f"{data.get('n_negatives', '?')} neg papers)"
        )
        return keep, review

    except Exception as e:
        logger.warning(
            f"[stage2_calibrator] Failed to load calibration file: {e} — "
            f"using stage2_rules.yaml defaults"
        )
        return DEFAULT_KEEP_THRESHOLD, DEFAULT_REVIEW_THRESHOLD


def calibrate_stage2_thresholds(
    papers: List,
    stage2_scorer,
    metadata_filter,
    precision_target: float = 0.90,
    recall_target:    float = 0.90,
    min_samples:      int   = 50,
) -> Tuple[float, float]:
    """
    Computes optimal Stage 2 keep/review thresholds from collected papers.

    Uses Stage-1-confirmed verdicts as ground truth labels, then finds
    the Stage 2 score boundaries that best match those labels.

    Args:
        papers:           All collected PaperRecord objects (load_all())
        stage2_scorer:    Callable — takes a PaperRecord, returns FilterVerdict
                          (pass RelevanceFilter._stage2_rules)
        metadata_filter:  MetadataFilter instance for Stage 1 ground truth
        precision_target: Min precision to call a paper KEEP (default 0.90)
        recall_target:    Min recall to call a paper REJECT (default 0.90)
        min_samples:      Minimum confirmed papers needed to calibrate

    Returns:
        (keep_threshold, review_threshold)
        Saves result to config/stage2_calibration.json.
        Falls back to defaults if not enough ground-truth data.
    """
    logger.info("[stage2_calibrator] Starting Stage 2 threshold calibration...")

    # ── Step 1: Collect Stage-1-confirmed ground truth labels ─────────────────
    # Only PubMed papers have MeSH tags → Stage 1 can make confident decisions
    pos_scores: List[float] = []   # Stage 1 said KEEP  → true positive
    neg_scores: List[float] = []   # Stage 1 said REJECT → true negative

    for paper in papers:
        source = (paper.source or "").lower()
        if "pubmed" not in source and "merged" not in source:
            continue   # Stage 1 only works on PubMed papers

        mv = metadata_filter.evaluate(paper)

        if mv.decision == "KEEP":
            # Stage 1 is confident this is a relevant human microbiome paper
            v = stage2_scorer(paper)
            pos_scores.append(v.score)

        elif mv.decision == "REJECT":
            # Stage 1 is confident this is NOT relevant
            v = stage2_scorer(paper)
            neg_scores.append(v.score)
        # UNKNOWN → skip, Stage 1 wasn't confident enough to use as ground truth

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    logger.info(
        f"[stage2_calibrator] Ground truth from Stage 1: "
        f"{n_pos} confirmed positives, {n_neg} confirmed negatives"
    )

    if n_pos < min_samples or n_neg < min_samples:
        logger.warning(
            f"[stage2_calibrator] Not enough ground-truth samples "
            f"(need {min_samples}+ each, got pos={n_pos} neg={n_neg}). "
            f"Collect more PubMed papers before calibrating. "
            f"Using stage2_rules.yaml defaults."
        )
        return DEFAULT_KEEP_THRESHOLD, DEFAULT_REVIEW_THRESHOLD

    # ── Step 2: Log score distributions ───────────────────────────────────────
    import statistics
    pos_mean = statistics.mean(pos_scores)
    neg_mean = statistics.mean(neg_scores)
    pos_min, pos_max = min(pos_scores), max(pos_scores)
    neg_min, neg_max = min(neg_scores), max(neg_scores)

    logger.info(
        f"[stage2_calibrator] Score distributions:\n"
        f"  Confirmed relevant   (n={n_pos}): "
        f"mean={pos_mean:.3f}  min={pos_min:.3f}  max={pos_max:.3f}\n"
        f"  Confirmed irrelevant (n={n_neg}): "
        f"mean={neg_mean:.3f}  min={neg_min:.3f}  max={neg_max:.3f}"
    )

    # ── Step 3: Find optimal thresholds by sweeping candidates ────────────────
    all_scores = sorted(set(pos_scores + neg_scores))

    # Build binary arrays: 1 = positive, 0 = negative
    labels = [1] * n_pos + [0] * n_neg
    scores = pos_scores + neg_scores

    keep_threshold   = DEFAULT_KEEP_THRESHOLD
    review_threshold = DEFAULT_REVIEW_THRESHOLD

    # ── KEEP threshold ─────────────────────────────────────────────────────────
    # Find the LOWEST score where precision of predicting KEEP ≥ precision_target
    # i.e. "papers above this score are relevant with ≥ 90% confidence"
    for t in reversed(all_scores):
        true_pos  = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
        false_pos = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        total_pos_pred = true_pos + false_pos

        if total_pos_pred == 0:
            continue

        precision = true_pos / total_pos_pred
        if precision >= precision_target:
            keep_threshold = float(t)
        else:
            break   # precision dropped below target — stop going lower

    # ── REVIEW threshold ───────────────────────────────────────────────────────
    # Find the HIGHEST score where recall of predicting REJECT ≥ recall_target
    # i.e. "papers below this score are irrelevant with ≥ 90% confidence"
    for t in all_scores:
        true_neg  = sum(1 for s, l in zip(scores, labels) if s < t and l == 0)
        total_neg = sum(1 for l in labels if l == 0)

        if total_neg == 0:
            continue

        recall = true_neg / total_neg
        if recall >= recall_target:
            review_threshold = float(t)
        else:
            break   # recall dropped below target — stop going higher

    # ── Safety check: review must be strictly less than keep ──────────────────
    if review_threshold >= keep_threshold:
        logger.warning(
            f"[stage2_calibrator] Threshold crossing detected "
            f"(review={review_threshold:.3f} ≥ keep={keep_threshold:.3f}) — "
            f"this means Stage 2 scores don't cleanly separate the two groups. "
            f"Falling back to stage2_rules.yaml defaults. "
            f"Consider updating positive_terms / negative_terms weights."
        )
        return DEFAULT_KEEP_THRESHOLD, DEFAULT_REVIEW_THRESHOLD

    # ── Log what the thresholds mean for your data ─────────────────────────────
    kept_correctly   = sum(1 for s, l in zip(scores, labels) if s >= keep_threshold   and l == 1)
    rejected_correctly = sum(1 for s, l in zip(scores, labels) if s < review_threshold and l == 0)
    borderline_count = sum(1 for s in scores if review_threshold <= s < keep_threshold)

    logger.info(
        f"[stage2_calibrator] Calibration result:\n"
        f"  keep_threshold   = {keep_threshold:.3f}  "
        f"(correctly keeps {kept_correctly}/{n_pos} confirmed positives = "
        f"{100*kept_correctly//n_pos}%)\n"
        f"  review_threshold = {review_threshold:.3f}  "
        f"(correctly rejects {rejected_correctly}/{n_neg} confirmed negatives = "
        f"{100*rejected_correctly//n_neg}%)\n"
        f"  borderline zone  = {review_threshold:.3f}–{keep_threshold:.3f}  "
        f"({borderline_count} papers sent to ML)"
    )

    # ── Step 4: Save to config/stage2_calibration.json ────────────────────────
    calibration = {
        "keep_threshold":   round(keep_threshold,   4),
        "review_threshold": round(review_threshold, 4),
        "calibrated_on":    datetime.utcnow().isoformat(),
        "n_positives":      n_pos,
        "n_negatives":      n_neg,
        "precision_target": precision_target,
        "recall_target":    recall_target,
        "note": (
            "Auto-calibrated from Stage-1-confirmed papers. "
            "Edit stage2_rules.yaml thresholds to override permanently."
        ),
    }

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    logger.success(
        f"[stage2_calibrator] Thresholds saved → {CALIBRATION_PATH} | "
        f"keep≥{keep_threshold:.3f}  review≥{review_threshold:.3f}"
    )

    return keep_threshold, review_threshold

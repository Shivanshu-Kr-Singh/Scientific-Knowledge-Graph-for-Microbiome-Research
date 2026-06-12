"""
collectors/ml_classifier.py
-----------------------------
Stage 3 — ML relevance classifier.
sentence-transformers + LogisticRegression.

STATUS: Inactive until trained. Train after collecting 500+ papers:
RUN_LAYER=train_filter python main.py

WHAT IT DOES:
Encodes title+abstract as a 384-dim semantic vector using
sentence-transformers (all-MiniLM-L6-v2), then applies a
LogisticRegression classifier trained on pseudo-labeled papers.

TRAINING STRATEGY — self-supervised bootstrap:
1. Run Stage 2 rules on collected papers to get confidence scores
2. score > 0.85 → pseudo-label as 1 (relevant)
3. score < 0.15 → pseudo-label as 0 (not relevant)
4. Borderline papers (0.15–0.85) excluded from training
5. Train + cross-validate on the high-confidence subset
6. Automatically find optimal keep/reject thresholds from training data
7. Save model + thresholds to config/relevance_model.pkl

WHY AUTO THRESHOLDS:
  Hardcoded 0.85/0.15 ignores the actual probability distribution of
  your data. If 80% of papers are relevant, the natural separation
  point is much lower than 0.85. Auto thresholds find the real boundary
  from the data and adjust per training run.

  KEEP threshold   = maximise precision (avoid false keeps)
  REJECT threshold = maximise recall   (avoid false rejects)
  Middle zone      = BORDERLINE → pass to Stage 4 LLM

WHY LOGISTIC REGRESSION NOT BERT:
For this classification task with ~300-1000 training samples,
LogisticRegression on sentence-transformer embeddings achieves
F1 > 0.90 and is 100x faster than fine-tuning BERT.
Upgrade to SVM or fine-tuned BERT if F1 drops below 0.85.
"""

import pickle
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from loguru import logger

MODEL_PATH = Path(__file__).parent.parent / "config" / "relevance_model.pkl"

# Fallback thresholds used ONLY when no trained model exists yet.
# These are overwritten by auto-computed values after every training run.
_DEFAULT_KEEP_THRESHOLD   = 0.85
_DEFAULT_REJECT_THRESHOLD = 0.15


@dataclass
class MLVerdict:
    """Result from Stage 3 ML classifier."""
    decision:   str      # "KEEP" | "REJECT" | "BORDERLINE" | "UNTRAINED"
    prob:       float    # Raw model probability (class 1 = relevant)
    reason:     str
    stage:      str = "stage3_ml"


class MLClassifier:
    """
    Stage 3: sentence-transformers + LogisticRegression.
    Thresholds are computed automatically from training data distribution.
    Falls back gracefully when model is not yet trained.
    """

    def __init__(self):
        self._model          = None
        self._encoder        = None
        self.trained         = False
        self.keep_threshold   = _DEFAULT_KEEP_THRESHOLD
        self.reject_threshold = _DEFAULT_REJECT_THRESHOLD
        self._try_load()

    def _try_load(self):
        """Load trained model + computed thresholds from disk. Silent if not found."""
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    saved = pickle.load(f)
                from sentence_transformers import SentenceTransformer
                self._encoder        = SentenceTransformer(
                    saved.get("encoder_name", "all-MiniLM-L6-v2")
                )
                self._model          = saved["model"]
                self.trained         = True

                # Load auto-computed thresholds if they exist in the saved file
                # Falls back to defaults for models trained before this feature
                self.keep_threshold   = saved.get("keep_threshold",   _DEFAULT_KEEP_THRESHOLD)
                self.reject_threshold = saved.get("reject_threshold",  _DEFAULT_REJECT_THRESHOLD)

                logger.info(
                    f"[ml_classifier] Model loaded | "
                    f"trained on {saved.get('n_samples', '?')} samples | "
                    f"keep≥{self.keep_threshold:.3f} "
                    f"reject≤{self.reject_threshold:.3f}"
                )
            except Exception as e:
                logger.warning(f"[ml_classifier] Failed to load model: {e}")
                self.trained = False
        else:
            logger.info("[ml_classifier] No model at config/relevance_model.pkl — Stage 3 inactive")
            logger.info("  Train with: RUN_LAYER=train_filter python main.py")

    def evaluate(self, paper) -> MLVerdict:
        """
        Classify one paper using auto-computed thresholds.
        Returns UNTRAINED if model not available — pipeline passes to Stage 4.
        """
        if not self.trained or self._model is None:
            return MLVerdict("UNTRAINED", 0.5, "model_not_trained_yet")

        try:
            text = (
                f"{getattr(paper, 'title',    '') or ''} "
                f"{getattr(paper, 'abstract', '') or ''}"
            )
            if len(text.strip()) < 10:
                return MLVerdict("UNTRAINED", 0.5, "insufficient_text")

            emb  = self._encoder.encode([text])
            prob = float(self._model.predict_proba(emb)[0][1])

            if prob >= self.keep_threshold:
                return MLVerdict(
                    "KEEP", prob,
                    f"ml_prob={prob:.3f} (threshold≥{self.keep_threshold:.3f})"
                )
            elif prob <= self.reject_threshold:
                return MLVerdict(
                    "REJECT", prob,
                    f"ml_prob={prob:.3f} (threshold≤{self.reject_threshold:.3f})"
                )
            else:
                return MLVerdict(
                    "BORDERLINE", prob,
                    f"ml_prob={prob:.3f} "
                    f"(between {self.reject_threshold:.3f}–{self.keep_threshold:.3f})"
                )

        except Exception as e:
            logger.warning(f"[ml_classifier] Inference error: {e}")
            return MLVerdict("UNTRAINED", 0.5, f"error:{e!s:.50}")

    # ── Threshold optimisation ────────────────────────────────────────────────

    @staticmethod
    def find_optimal_thresholds(
        clf,
        X,
        y,
        precision_target: float = 0.90,
        recall_target:    float = 0.90,
    ) -> Tuple[float, float]:
        """
        Automatically finds KEEP and REJECT thresholds from the training data.

        HOW IT WORKS:
          1. Run the trained model on all training samples → get probabilities
          2. Sweep every possible threshold value (all unique probabilities)
          3. KEEP threshold  = lowest probability where precision ≥ precision_target
             "Only say KEEP when we're at least 90% sure it's relevant"
          4. REJECT threshold = highest probability where recall of negatives ≥ recall_target
             "Only say REJECT when we're at least 90% sure it's irrelevant"
          5. Everything between the two thresholds → BORDERLINE → passes to LLM

        WHY THIS IS BETTER THAN HARDCODED:
          The optimal boundary shifts with your data:
          - Mostly relevant papers → keep threshold naturally lower
          - Balanced dataset      → keep threshold near 0.5
          - Mostly irrelevant     → keep threshold higher
          The model tells you what it actually learned, not what you assumed.

        Args:
            clf:               Trained LogisticRegression
            X:                 Encoded training vectors
            y:                 True labels (0/1)
            precision_target:  Minimum precision to call KEEP (default 0.90)
            recall_target:     Minimum recall to call REJECT (default 0.90)

        Returns:
            (keep_threshold, reject_threshold)
        """
        import numpy as np
        from sklearn.metrics import precision_score, recall_score

        probs = clf.predict_proba(X)[:, 1]   # prob of class 1 (relevant)

        # All unique probability values as candidate thresholds
        candidates = sorted(set(probs))

        # ── Find KEEP threshold ───────────────────────────────────────────────
        # Sweep from high to low: find the lowest threshold where
        # precision is still ≥ precision_target
        keep_threshold = _DEFAULT_KEEP_THRESHOLD
        for t in reversed(candidates):
            preds = (probs >= t).astype(int)
            if preds.sum() == 0:
                continue
            prec = precision_score(y, preds, zero_division=0)
            if prec >= precision_target:
                keep_threshold = float(t)
                # Keep going lower — we want the lowest t that still hits target
            else:
                break   # precision dropped below target — stop

        # ── Find REJECT threshold ─────────────────────────────────────────────
        # Sweep from low to high: find the highest threshold where
        # recall on negatives is still ≥ recall_target
        # (i.e. we correctly reject at least recall_target % of true negatives)
        reject_threshold = _DEFAULT_REJECT_THRESHOLD
        for t in candidates:
            preds = (probs < t).astype(int)   # predicting "reject"
            true_neg = (y == 0).astype(int)
            if true_neg.sum() == 0:
                continue
            rec = recall_score(true_neg, preds, zero_division=0)
            if rec >= recall_target:
                reject_threshold = float(t)
            else:
                break

        # Safety: ensure reject < keep (they should never cross)
        if reject_threshold >= keep_threshold:
            logger.warning(
                f"[ml_classifier] Threshold crossing detected "
                f"(reject={reject_threshold:.3f} ≥ keep={keep_threshold:.3f}) "
                f"— falling back to defaults"
            )
            return _DEFAULT_KEEP_THRESHOLD, _DEFAULT_REJECT_THRESHOLD

        return keep_threshold, reject_threshold

    def train(self, papers: list, rule_scores: List[float]) -> Tuple[float, int]:
        """
        Train the ML classifier using rule scores as pseudo-labels,
        then automatically compute optimal keep/reject thresholds.

        Args:
            papers:      List of PaperRecord objects
            rule_scores: Corresponding Stage 2 scores (0.0–1.0)

        Returns:
            (f1_score, n_training_samples) — for reporting
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            import numpy as np
        except ImportError as e:
            logger.error(f"[ml_classifier] Missing dependency: {e}")
            logger.error("Run: pip install sentence-transformers scikit-learn")
            return 0.0, 0

        # ── Build pseudo-labeled dataset ──────────────────────────────────────
        texts, labels = [], []
        skipped = 0

        for paper, score in zip(papers, rule_scores):
            text = (
                f"{getattr(paper, 'title',    '') or ''} "
                f"{getattr(paper, 'abstract', '') or ''}"
            )
            if score >= 0.85:
                texts.append(text)
                labels.append(1)
            elif score <= 0.15:
                texts.append(text)
                labels.append(0)
            else:
                skipped += 1   # Borderline — uncertain label, excluded

        pos   = labels.count(1)
        neg   = labels.count(0)
        total = pos + neg

        logger.info(
            f"[ml_classifier] Training data: {pos} relevant, {neg} off-topic, "
            f"{skipped} borderline excluded"
        )

        if total < 50:
            logger.warning(
                f"[ml_classifier] Only {total} samples — need 50+. "
                "Run with MAX_PER_SOURCE=500 first."
            )
            return 0.0, 0

        if min(pos, neg) < 5:
            logger.warning(
                f"[ml_classifier] Class imbalance too severe "
                f"(pos={pos}, neg={neg}). Collect more diverse papers."
            )
            return 0.0, 0

        # ── Encode ────────────────────────────────────────────────────────────
        logger.info("[ml_classifier] Encoding with all-MiniLM-L6-v2...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        X = encoder.encode(texts, show_progress_bar=True, batch_size=128,
                           convert_to_numpy=True)
        y = np.array(labels)

        # ── Cross-validate ────────────────────────────────────────────────────
        n_splits  = min(5, min(pos, neg))
        clf = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced",
            random_state=42, solver="saga", n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        try:
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1", n_jobs=-1)
            f1_mean   = float(cv_scores.mean())
            f1_std    = float(cv_scores.std())
            logger.info(f"[ml_classifier] Cross-val F1: {f1_mean:.3f} ± {f1_std:.3f}")
        except Exception as e:
            logger.warning(f"[ml_classifier] Cross-val failed: {e}")
            f1_mean = 0.0

        # ── Train final model on full dataset ─────────────────────────────────
        clf.fit(X, y)

        # ── Auto-compute thresholds from this model + this data ───────────────
        logger.info("[ml_classifier] Computing optimal thresholds from training distribution...")
        keep_t, reject_t = self.find_optimal_thresholds(clf, X, y)

        # Log the probability distribution for transparency
        probs     = clf.predict_proba(X)[:, 1]
        pos_probs = probs[y == 1]
        neg_probs = probs[y == 0]
        logger.info(
            f"[ml_classifier] Probability distribution:\n"
            f"  Relevant papers  (n={pos}): "
            f"mean={pos_probs.mean():.3f}  "
            f"min={pos_probs.min():.3f}  "
            f"max={pos_probs.max():.3f}\n"
            f"  Irrelevant papers (n={neg}): "
            f"mean={neg_probs.mean():.3f}  "
            f"min={neg_probs.min():.3f}  "
            f"max={neg_probs.max():.3f}\n"
            f"  → Auto thresholds: "
            f"KEEP≥{keep_t:.3f}  REJECT≤{reject_t:.3f}  "
            f"BORDERLINE={reject_t:.3f}–{keep_t:.3f}"
        )

        # ── Save model + thresholds ───────────────────────────────────────────
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":            clf,
                "encoder_name":     "all-MiniLM-L6-v2",
                "n_samples":        total,
                "f1":               f1_mean,
                "keep_threshold":   keep_t,
                "reject_threshold": reject_t,
                "pos_samples":      pos,
                "neg_samples":      neg,
            }, f)

        # Hot-reload thresholds into this instance immediately
        self._model           = clf
        self._encoder         = encoder
        self.trained          = True
        self.keep_threshold   = keep_t
        self.reject_threshold = reject_t

        quality = (
            "excellent"                          if f1_mean > 0.92 else
            "good — ready to use"                if f1_mean > 0.85 else
            "moderate — retrain with more data"
        )
        logger.success(
            f"[ml_classifier] Model saved → {MODEL_PATH} | "
            f"F1={f1_mean:.3f} ({quality}) | "
            f"thresholds: keep≥{keep_t:.3f} reject≤{reject_t:.3f}"
        )
        return f1_mean, total

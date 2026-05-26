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
6. Save to config/relevance_model.pkl

WHY LOGISTIC REGRESSION NOT BERT:
For this classification task with ~300-1000 training samples,
LogisticRegression on sentence-transformer embeddings achieves
F1 > 0.90 and is 100x faster than fine-tuning BERT.
Upgrade to SVM or fine-tuned BERT if F1 drops below 0.85.

DECISION THRESHOLDS:
prob >= 0.85 → KEEP
prob <= 0.15 → REJECT
0.15–0.85   → BORDERLINE → pass to Stage 4 LLM
"""

import pickle
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger

MODEL_PATH = Path(__file__).parent.parent / "config" / "relevance_model.pkl"

KEEP_THRESHOLD   = 0.85
REJECT_THRESHOLD = 0.15


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
    Falls back gracefully when model is not yet trained.
    """

    def __init__(self):
        self._model   = None
        self._encoder = None
        self.trained  = False
        self._try_load()

    def _try_load(self):
        """Load trained model if it exists. Silent if not found."""
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    saved = pickle.load(f)
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(saved.get("encoder_name",
                                                              "all-MiniLM-L6-v2"))
                self._model   = saved["model"]
                self.trained  = True
                logger.info(f"[ml_classifier] Model loaded | trained on {saved.get('n_samples','?')} samples")
            except Exception as e:
                logger.warning(f"[ml_classifier] Failed to load model: {e}")
                self.trained = False
        else:
            logger.info("[ml_classifier] No model at config/relevance_model.pkl — Stage 3 inactive")
            logger.info("  Train with: RUN_LAYER=train_filter python main.py")

    def evaluate(self, paper) -> MLVerdict:
        """
        Classify one paper.
        Returns UNTRAINED if model not available — pipeline passes to Stage 4.
        """
        if not self.trained or self._model is None:
            return MLVerdict("UNTRAINED", 0.5, "model_not_trained_yet")

        try:
            text = f"{getattr(paper, 'title', '') or ''} {getattr(paper, 'abstract', '') or ''}"
            if len(text.strip()) < 10:
                return MLVerdict("UNTRAINED", 0.5, "insufficient_text")

            emb  = self._encoder.encode([text])
            prob = float(self._model.predict_proba(emb)[0][1])

            if prob >= KEEP_THRESHOLD:
                return MLVerdict("KEEP",       prob, f"ml_prob={prob:.3f}")
            elif prob <= REJECT_THRESHOLD:
                return MLVerdict("REJECT",     prob, f"ml_prob={prob:.3f}")
            else:
                return MLVerdict("BORDERLINE", prob,
                                 f"ml_prob={prob:.3f}_uncertain_pass_to_llm")

        except Exception as e:
            logger.warning(f"[ml_classifier] Inference error: {e}")
            return MLVerdict("UNTRAINED", 0.5, f"error:{e!s:.50}")

    def train(self, papers: list, rule_scores: List[float]) -> Tuple[float, int]:
        """
        Train the ML classifier using rule scores as pseudo-labels.

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

        # Build pseudo-labeled dataset
        texts, labels = [], []
        skipped = 0

        for paper, score in zip(papers, rule_scores):
            text = f"{getattr(paper, 'title', '') or ''} {getattr(paper, 'abstract', '') or ''}"
            if score >= 0.85:
                texts.append(text)
                labels.append(1)
            elif score <= 0.15:
                texts.append(text)
                labels.append(0)
            else:
                skipped += 1  # Borderline — don't train on uncertain labels

        pos = labels.count(1)
        neg = labels.count(0)
        total = pos + neg

        logger.info(f"[ml_classifier] Training data: {pos} relevant, {neg} off-topic, "
                    f"{skipped} borderline excluded")

        if total < 50:
            logger.warning(f"[ml_classifier] Only {total} samples. Need 50+. "
                           "Run with MAX_PER_SOURCE=500 first to collect more papers.")
            return 0.0, 0

        if min(pos, neg) < 5:
            logger.warning(f"[ml_classifier] Class imbalance too severe (pos={pos}, neg={neg}). "
                           "Collect more papers from diverse sources.")
            return 0.0, 0

        # Encode with sentence-transformers
        logger.info("[ml_classifier] Encoding with all-MiniLM-L6-v2...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        X = encoder.encode(texts, show_progress_bar=True, batch_size=32)
        y = np.array(labels)

        # Cross-validate then train final model
        n_splits = min(5, min(pos, neg))
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 random_state=42)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        try:
            cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
            f1_mean   = float(cv_scores.mean())
            f1_std    = float(cv_scores.std())
            logger.info(f"[ml_classifier] Cross-val F1: {f1_mean:.3f} ± {f1_std:.3f}")
        except Exception as e:
            logger.warning(f"[ml_classifier] Cross-val failed: {e} — training without CV")
            f1_mean = 0.0

        clf.fit(X, y)

        # Save model
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":        clf,
                "encoder_name": "all-MiniLM-L6-v2",
                "n_samples":    total,
                "f1":           f1_mean,
            }, f)

        self._model   = clf
        self._encoder = encoder
        self.trained  = True

        quality = ("excellent" if f1_mean > 0.9
                   else "good — ready to use" if f1_mean > 0.8
                   else "moderate — retrain with more data when available")

        logger.success(
            f"[ml_classifier] Trained on {total} samples | "
            f"F1={f1_mean:.3f} ({quality}) | "
            f"Saved → {MODEL_PATH}"
        )
        return f1_mean, total

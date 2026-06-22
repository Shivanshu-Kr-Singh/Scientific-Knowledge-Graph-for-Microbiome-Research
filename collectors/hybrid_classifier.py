"""
collectors/hybrid_classifier.py
────────────────────────────────
Stacked meta-classifier combining all signal sources into one calibrated
confidence score.

WHY THIS EXISTS:
  Individual stages (rules, embeddings, ML) each provide a partial signal.
  After accumulating ~2000 papers, we have enough labeled data to train a
  stacked model that combines all four signals into a single, well-calibrated
  probability. This improves routing decisions and reduces unnecessary LLM
  calls.

HOW IT WORKS:
  1. Feature vector: [rule_score, pos_similarity, neg_similarity, ml_probability]
  2. Model: LogisticRegression trained on LLM-verified papers (ground truth)
  3. Output: PlattCalibrator-scaled probability → HybridVerdict
  4. Activation gate: only active when embedding store has ≥ 2000 papers total
  5. Quality gate: model discarded if hold-out F1 < 0.80

PERSISTENCE:
  Model + metadata saved to data/models/hybrid_classifier.pkl
  Calibration parameters saved via PlattCalibrator (calibration_params.json)

RETRAINING:
  retrain_if_needed() checks if ≥ 100 new LLM-verified papers have accumulated
  since last training. If so, retrains and evaluates. Called by weekly_refresh job.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from config import (
    HYBRID_MIN_STORE_SIZE,
    HYBRID_MIN_TRAIN_SAMPLES,
    HYBRID_MIN_RETRAIN_NEW,
    DATA_DIR,
)
from collectors.calibration import PlattCalibrator


# ── Persistence paths ─────────────────────────────────────────────────────────
MODELS_DIR = DATA_DIR / "models"


@dataclass
class HybridVerdict:
    """Result from the stacked meta-classifier."""

    confidence: float  # Calibrated probability [0, 1]
    keep: bool
    raw_logit: float  # Pre-calibration logit
    feature_vector: list  # [rule_score, pos_sim, neg_sim, ml_prob]
    reason: str


class HybridClassifier:
    """
    Stacked meta-classifier (LogisticRegression).
    Combines: rule_score, embedding_pos_similarity, embedding_neg_similarity,
              ml_probability.

    Activation: requires >= 2000 papers in Embedding Store.
    Training data: LLM-verified papers as ground truth labels.
    Output: Platt-scaled calibrated probability.

    Model persistence: data/models/hybrid_classifier.pkl
    """

    MODEL_PATH: Path = MODELS_DIR / "hybrid_classifier.pkl"
    MIN_STORE_SIZE: int = HYBRID_MIN_STORE_SIZE  # 2000
    MIN_TRAINING_SAMPLES: int = HYBRID_MIN_TRAIN_SAMPLES  # 200
    MIN_RETRAIN_NEW: int = HYBRID_MIN_RETRAIN_NEW  # 100
    MIN_F1_THRESHOLD: float = 0.80

    def __init__(self) -> None:
        self._model = None  # sklearn LogisticRegression or None
        self._calibrator = PlattCalibrator()
        self._metadata: dict = {}  # Persisted alongside model
        self._store_size: int = 0  # Cached store size for is_active check

        # Attempt to load existing model from disk
        self._load_model()

    @property
    def is_active(self) -> bool:
        """
        True when a trained model exists AND the embedding store has
        >= 2000 papers total.

        The store size is updated via set_store_size() or retrain_if_needed().
        """
        return self._model is not None and self._store_size >= self.MIN_STORE_SIZE

    def set_store_size(self, total_papers: int) -> None:
        """
        Update the cached store size. Called by the pipeline to keep the
        activation gate current without requiring a direct store reference.
        """
        self._store_size = total_papers

    def predict(
        self,
        rule_score: float,
        pos_sim: float,
        neg_sim: float,
        ml_prob: float,
    ) -> HybridVerdict:
        """
        Predict relevance using the stacked meta-classifier.

        Args:
            rule_score: Output from Stage 2 weighted rules [0, 1]
            pos_sim: Cosine similarity to positive partition [0, 1]
            neg_sim: Cosine similarity to negative partition [0, 1]
            ml_prob: Output from Stage 3 ML classifier [0, 1]

        Returns:
            HybridVerdict with calibrated confidence and keep decision.

        Raises:
            RuntimeError: If the classifier is not active.
        """
        if not self.is_active:
            raise RuntimeError(
                "HybridClassifier is not active. "
                f"Requires trained model and store size >= {self.MIN_STORE_SIZE}. "
                f"Current store size: {self._store_size}, model loaded: {self._model is not None}"
            )

        feature_vector = [rule_score, pos_sim, neg_sim, ml_prob]
        X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)

        # Get raw logit (log-odds) from the model
        raw_logit = float(self._model.decision_function(X)[0])

        # Calibrate using Platt scaling if available
        if self._calibrator.is_fitted:
            confidence = self._calibrator.calibrate(raw_logit)
        else:
            # Fallback: use model's own predict_proba
            confidence = float(self._model.predict_proba(X)[0, 1])

        keep = confidence >= 0.5

        reason = (
            f"Hybrid meta-classifier: confidence={confidence:.3f} "
            f"(raw_logit={raw_logit:.3f}), "
            f"features=[rule={rule_score:.3f}, pos_sim={pos_sim:.3f}, "
            f"neg_sim={neg_sim:.3f}, ml_prob={ml_prob:.3f}]"
        )

        return HybridVerdict(
            confidence=confidence,
            keep=keep,
            raw_logit=raw_logit,
            feature_vector=feature_vector,
            reason=reason,
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """
        Train the meta-classifier on LLM-verified paper features.

        Args:
            features: (N, 4) array of [rule_score, pos_sim, neg_sim, ml_prob]
            labels: (N,) binary array (1 = relevant, 0 = irrelevant)

        Returns:
            dict with training metrics: f1, accuracy, auc, n_samples, accepted.
            'accepted' is True if model passed F1 threshold and was persisted.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64).ravel()

        n_samples = len(labels)

        if n_samples < self.MIN_TRAINING_SAMPLES:
            logger.warning(
                f"[hybrid_classifier] Insufficient training data: "
                f"{n_samples} samples < minimum {self.MIN_TRAINING_SAMPLES}. "
                f"Skipping training."
            )
            return {
                "f1": 0.0,
                "accuracy": 0.0,
                "auc": 0.0,
                "n_samples": n_samples,
                "accepted": False,
                "reason": "insufficient_samples",
            }

        # Validate feature dimensions
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError(
                f"features must have shape (N, 4), got {features.shape}"
            )

        # Need both classes for meaningful training
        unique_labels = set(np.unique(labels).tolist())
        if len(unique_labels) < 2:
            logger.warning(
                "[hybrid_classifier] Only one class present in training data. "
                "Cannot train. Skipping."
            )
            return {
                "f1": 0.0,
                "accuracy": 0.0,
                "auc": 0.0,
                "n_samples": n_samples,
                "accepted": False,
                "reason": "single_class",
            }

        # ── Train/test split (80/20) ─────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # ── Train LogisticRegression ──────────────────────────────────────────
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # ── Evaluate on hold-out set ──────────────────────────────────────────
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        f1 = float(f1_score(y_test, y_pred))
        accuracy = float(accuracy_score(y_test, y_pred))

        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            # Can happen if test set has only one class after split
            auc = 0.0

        metrics = {
            "f1": f1,
            "accuracy": accuracy,
            "auc": auc,
            "n_samples": n_samples,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

        # ── F1 quality gate ───────────────────────────────────────────────────
        if f1 < self.MIN_F1_THRESHOLD:
            logger.warning(
                f"[hybrid_classifier] Model F1={f1:.3f} < threshold "
                f"{self.MIN_F1_THRESHOLD}. Discarding model."
            )
            metrics["accepted"] = False
            metrics["reason"] = "f1_below_threshold"
            return metrics

        # ── Model accepted — persist ──────────────────────────────────────────
        self._model = clf
        metrics["accepted"] = True
        metrics["reason"] = "accepted"

        # ── Fit PlattCalibrator on hold-out logits ────────────────────────────
        holdout_logits = clf.decision_function(X_test)
        calibration_result = self._calibrator.fit(holdout_logits, y_test)

        if calibration_result is not None:
            self._calibrator.save()
            metrics["calibration_error"] = calibration_result.calibration_error
        else:
            metrics["calibration_error"] = None

        # ── Save model to disk ────────────────────────────────────────────────
        self._metadata = {
            "n_samples_trained": n_samples,
            "metrics": metrics,
        }
        self._save_model()

        logger.info(
            f"[hybrid_classifier] Model trained and accepted: "
            f"F1={f1:.3f}, AUC={auc:.3f}, accuracy={accuracy:.3f}, "
            f"n_samples={n_samples}"
        )

        return metrics

    def retrain_if_needed(self, store) -> Optional[dict]:
        """
        Check if retraining is needed and perform it if so.

        Retrains when >= MIN_RETRAIN_NEW (100) new LLM-verified papers have
        accumulated since last training.

        Args:
            store: The EmbeddingStore instance (used for store size check
                   and to count LLM-verified papers via metadata).

        Returns:
            Metrics dict if retrained, None otherwise.
        """
        import json

        # Update store size for activation gate
        total_papers = store.positive_count + store.negative_count
        self.set_store_size(total_papers)

        # Check minimum store size
        if total_papers < self.MIN_STORE_SIZE:
            logger.debug(
                f"[hybrid_classifier] Store has {total_papers} papers, "
                f"need {self.MIN_STORE_SIZE}. Skipping retrain."
            )
            return None

        # ── Count LLM-verified papers ─────────────────────────────────────────
        # LLM-verified papers are those in the audit trail with llm_verified flag
        llm_verified_path = DATA_DIR / "audit" / "llm_verified.json"

        if not llm_verified_path.exists():
            logger.debug(
                "[hybrid_classifier] No llm_verified.json found. "
                "Cannot retrain without labeled data."
            )
            return None

        try:
            with open(llm_verified_path, "r", encoding="utf-8") as f:
                llm_verified = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"[hybrid_classifier] Failed to read llm_verified.json: {e}"
            )
            return None

        if not isinstance(llm_verified, list):
            logger.warning(
                "[hybrid_classifier] llm_verified.json is not a list. Skipping."
            )
            return None

        current_count = len(llm_verified)
        last_trained_count = self._metadata.get("n_samples_trained", 0)
        new_papers = current_count - last_trained_count

        if new_papers < self.MIN_RETRAIN_NEW:
            logger.debug(
                f"[hybrid_classifier] Only {new_papers} new papers since last "
                f"training (need {self.MIN_RETRAIN_NEW}). Skipping retrain."
            )
            return None

        # ── Build training features ───────────────────────────────────────────
        features_list = []
        labels_list = []

        for record in llm_verified:
            # Each record should have feature scores and a label
            rule_score = record.get("rule_score")
            pos_sim = record.get("pos_similarity", record.get("pos_sim"))
            neg_sim = record.get("neg_similarity", record.get("neg_sim"))
            ml_prob = record.get("ml_probability", record.get("ml_prob"))
            label = record.get("label", record.get("verdict"))

            # Skip records missing required features
            if any(v is None for v in [rule_score, pos_sim, neg_sim, ml_prob, label]):
                continue

            # Normalize label to 0/1
            if isinstance(label, str):
                label_val = 1.0 if label.lower() in ("keep", "relevant", "true", "1") else 0.0
            else:
                label_val = float(label)

            features_list.append([
                float(rule_score),
                float(pos_sim),
                float(neg_sim),
                float(ml_prob),
            ])
            labels_list.append(label_val)

        if len(features_list) < self.MIN_TRAINING_SAMPLES:
            logger.warning(
                f"[hybrid_classifier] Only {len(features_list)} usable records "
                f"with complete features (need {self.MIN_TRAINING_SAMPLES}). "
                f"Skipping retrain."
            )
            return None

        features = np.array(features_list, dtype=np.float64)
        labels = np.array(labels_list, dtype=np.float64)

        logger.info(
            f"[hybrid_classifier] Retraining with {len(features_list)} samples "
            f"({new_papers} new since last training)."
        )

        return self.train(features, labels)

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load model and metadata from pickle file if it exists."""
        if not self.MODEL_PATH.exists():
            logger.debug(
                f"[hybrid_classifier] No model file at {self.MODEL_PATH}. "
                f"Classifier inactive until trained."
            )
            return

        try:
            with open(self.MODEL_PATH, "rb") as f:
                data = pickle.load(f)

            self._model = data.get("model")
            self._metadata = data.get("metadata", {})

            # Load calibrator parameters
            self._calibrator.load()

            logger.info(
                f"[hybrid_classifier] Model loaded from {self.MODEL_PATH}. "
                f"Trained on {self._metadata.get('n_samples_trained', '?')} samples."
            )

        except (pickle.UnpicklingError, EOFError, KeyError, TypeError) as e:
            logger.warning(
                f"[hybrid_classifier] Failed to load model from "
                f"{self.MODEL_PATH}: {e}. Starting fresh."
            )
            self._model = None
            self._metadata = {}

    def _save_model(self) -> None:
        """Persist model and metadata to pickle file."""
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self._model,
            "metadata": self._metadata,
        }

        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"[hybrid_classifier] Model saved → {self.MODEL_PATH}")

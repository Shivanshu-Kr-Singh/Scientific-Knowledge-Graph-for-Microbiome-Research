"""
collectors/calibration.py
─────────────────────────
Platt scaling calibration for the hybrid meta-classifier.

WHY THIS EXISTS:
  Raw classifier logits don't correspond to true probabilities.
  A score of 0.90 doesn't mean 90% precision unless we explicitly calibrate.
  Platt scaling fits a sigmoid (logistic) function on held-out logits
  so that output probabilities reflect empirical class frequencies.

HOW IT WORKS:
  1. Collect held-out logits from the hybrid classifier (validation set)
  2. Fit: P(y=1|logit) = 1 / (1 + exp(-(slope * logit + intercept)))
  3. Enforce slope > 0 for monotonicity guarantee
  4. Output is always clamped to [0, 1]

MINIMUM DATA:
  200 LLM-verified papers required for reliable calibration.
  Below this threshold, calibration is skipped with a warning.

PERSISTENCE:
  Parameters saved to data/models/calibration_params.json alongside
  hybrid_classifier.pkl.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from config import HYBRID_MIN_TRAIN_SAMPLES, DATA_DIR


# ── Persistence path ──────────────────────────────────────────────────────────
MODELS_DIR = DATA_DIR / "models"
CALIBRATION_PARAMS_PATH = MODELS_DIR / "calibration_params.json"


@dataclass
class CalibrationResult:
    """Result from fitting the Platt scaling layer."""

    slope: float
    intercept: float
    calibration_error: float  # Expected Calibration Error (ECE)
    n_samples: int


class PlattCalibrator:
    """
    Platt scaling: fits a logistic regression on held-out logits to produce
    calibrated probabilities.

        P(y=1|logit) = 1 / (1 + exp(-(slope * logit + intercept)))

    Guarantees:
      - Monotonicity: slope is constrained to be positive, so calibrate(a) <= calibrate(b)
        whenever a < b.
      - Range: output is always in [0, 1].

    Persistence: parameters saved alongside hybrid_classifier.pkl.
    Minimum data: 200 LLM-verified papers for reliable calibration.
    """

    MIN_CALIBRATION_SAMPLES: int = HYBRID_MIN_TRAIN_SAMPLES  # 200

    def __init__(self) -> None:
        self._slope: Optional[float] = None
        self._intercept: Optional[float] = None
        self._calibration_error: Optional[float] = None
        self._n_samples: int = 0

    @property
    def is_fitted(self) -> bool:
        """True when slope and intercept have been set via fit()."""
        return self._slope is not None and self._intercept is not None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> Optional[CalibrationResult]:
        """
        Fit Platt scaling on a validation set of logits and binary labels.

        Args:
            logits: 1D array of raw classifier logits (float).
            labels: 1D array of binary ground-truth labels (0 or 1).

        Returns:
            CalibrationResult if fitting succeeds, None if insufficient data.

        Raises:
            ValueError: If logits and labels have different lengths or labels
                        contain values other than 0/1.
        """
        logits = np.asarray(logits, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"logits and labels must have same length, "
                f"got {logits.shape[0]} vs {labels.shape[0]}"
            )

        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"labels must be binary (0 or 1), got unique values: {unique_labels}"
            )

        n_samples = len(logits)

        # ── Minimum sample check ──────────────────────────────────────────────
        if n_samples < self.MIN_CALIBRATION_SAMPLES:
            logger.warning(
                f"[calibration] Insufficient data for Platt scaling: "
                f"{n_samples} samples < minimum {self.MIN_CALIBRATION_SAMPLES}. "
                f"Skipping calibration — raw probabilities will be used."
            )
            return None

        # ── Need both classes present ─────────────────────────────────────────
        if len(unique_labels) < 2:
            logger.warning(
                f"[calibration] Only one class present in labels. "
                f"Cannot fit calibration. Skipping."
            )
            return None

        # ── Fit logistic regression (Platt scaling) ───────────────────────────
        # Using sklearn LogisticRegression with very high C (no regularization)
        # on 1D feature (the raw logit) to get slope and intercept.
        from sklearn.linear_model import LogisticRegression

        X = logits.reshape(-1, 1)
        y = labels

        clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        clf.fit(X, y)

        raw_slope = float(clf.coef_[0, 0])
        raw_intercept = float(clf.intercept_[0])

        # ── Enforce monotonicity: slope must be positive ──────────────────────
        # A positive slope means higher logits → higher P(y=1), which is
        # the expected monotonic behavior. If sklearn fits a negative slope
        # (unlikely but possible with degenerate data), we take abs().
        if raw_slope < 0:
            logger.warning(
                f"[calibration] Fitted slope is negative ({raw_slope:.4f}). "
                f"Forcing positive for monotonicity."
            )
            raw_slope = abs(raw_slope)
            raw_intercept = -raw_intercept

        self._slope = raw_slope
        self._intercept = raw_intercept
        self._n_samples = n_samples

        # ── Compute Expected Calibration Error (ECE) ──────────────────────────
        self._calibration_error = self._compute_ece(logits, labels, n_bins=10)

        result = CalibrationResult(
            slope=self._slope,
            intercept=self._intercept,
            calibration_error=self._calibration_error,
            n_samples=self._n_samples,
        )

        logger.info(
            f"[calibration] Platt scaling fitted: slope={self._slope:.4f}, "
            f"intercept={self._intercept:.4f}, ECE={self._calibration_error:.4f}, "
            f"n_samples={self._n_samples}"
        )

        return result

    def calibrate(self, raw_logit: float) -> float:
        """
        Transform a raw logit into a calibrated probability.

        Uses: P(y=1|x) = 1 / (1 + exp(-(slope * x + intercept)))
        Output is clamped to [0, 1].

        Args:
            raw_logit: The raw classifier logit value.

        Returns:
            Calibrated probability in [0, 1].

        Raises:
            RuntimeError: If the calibrator has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "PlattCalibrator has not been fitted. Call fit() first."
            )

        z = self._slope * raw_logit + self._intercept
        # Sigmoid with numerical stability
        prob = self._sigmoid(z)
        # Clamp to [0, 1] for safety (sigmoid already does this, but explicit)
        return float(np.clip(prob, 0.0, 1.0))

    def save(self, path: Optional[Path] = None) -> None:
        """
        Persist calibration parameters to JSON.

        Args:
            path: Override path. Defaults to data/models/calibration_params.json.
        """
        if not self.is_fitted:
            logger.warning("[calibration] Cannot save — calibrator not fitted.")
            return

        save_path = path or CALIBRATION_PARAMS_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "slope": self._slope,
            "intercept": self._intercept,
            "calibration_error": self._calibration_error,
            "n_samples": self._n_samples,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        logger.info(f"[calibration] Parameters saved → {save_path}")

    def load(self, path: Optional[Path] = None) -> bool:
        """
        Load calibration parameters from JSON.

        Args:
            path: Override path. Defaults to data/models/calibration_params.json.

        Returns:
            True if loaded successfully, False otherwise.
        """
        load_path = path or CALIBRATION_PARAMS_PATH

        if not load_path.exists():
            logger.debug(
                f"[calibration] No calibration file at {load_path} — "
                f"calibrator remains unfitted."
            )
            return False

        try:
            with open(load_path, "r", encoding="utf-8") as f:
                params = json.load(f)

            self._slope = float(params["slope"])
            self._intercept = float(params["intercept"])
            self._calibration_error = float(params.get("calibration_error", 0.0))
            self._n_samples = int(params.get("n_samples", 0))

            # Validate monotonicity constraint on load
            if self._slope < 0:
                logger.warning(
                    f"[calibration] Loaded slope is negative ({self._slope}). "
                    f"Forcing positive for monotonicity."
                )
                self._slope = abs(self._slope)
                self._intercept = -self._intercept

            logger.info(
                f"[calibration] Parameters loaded from {load_path}: "
                f"slope={self._slope:.4f}, intercept={self._intercept:.4f}"
            )
            return True

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning(
                f"[calibration] Failed to load calibration from {load_path}: {e}"
            )
            self._slope = None
            self._intercept = None
            return False

    def _compute_ece(
        self, logits: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error with equal-width bins.

        ECE = sum over bins of (|bin_size| / N) * |accuracy(bin) - confidence(bin)|

        Args:
            logits: Raw logits from the classifier.
            labels: True binary labels.
            n_bins: Number of bins (default 10).

        Returns:
            ECE value in [0, 1].
        """
        # Get calibrated probabilities for all logits
        probs = np.array([self._sigmoid(self._slope * x + self._intercept) for x in logits])
        n_total = len(probs)

        if n_total == 0:
            return 0.0

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            if i == n_bins - 1:
                # Last bin includes upper boundary
                in_bin = (probs >= bin_lower) & (probs <= bin_upper)
            else:
                in_bin = (probs >= bin_lower) & (probs < bin_upper)

            bin_size = in_bin.sum()
            if bin_size == 0:
                continue

            bin_accuracy = labels[in_bin].mean()
            bin_confidence = probs[in_bin].mean()

            ece += (bin_size / n_total) * abs(bin_accuracy - bin_confidence)

        return float(ece)

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable sigmoid function."""
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            return exp_z / (1.0 + exp_z)

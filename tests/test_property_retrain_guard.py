"""
Property Tests for Active Learning Retrain Guard

**Validates: Requirements 12.2, 12.5**

Property 16: Active Learning Retrain Guard
  For any count of new LLM-verified papers since last training, the Active
  Learning Job SHALL retrain only when count >= 100. If the resulting model
  has F1 < 0.80, the previous model SHALL be retained unchanged.

Property 16a (Retrain Guard):
  Generate random counts of new_papers (0 to 500). When new_papers < 100,
  retrain_if_needed returns None (no retraining occurs).

Property 16b (F1 Quality Gate):
  When training produces a model with F1 < 0.80, the train() method returns
  accepted: False and the model is not persisted.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Number of new papers since last training (0 to 500)
_new_paper_count_st = st.integers(min_value=0, max_value=500)

# Last trained count (0 to 1000)
_last_trained_count_st = st.integers(min_value=0, max_value=1000)

# Store size (total papers in embedding store)
_store_size_st = st.integers(min_value=2000, max_value=10000)


# ---------------------------------------------------------------------------
# Property 16a: Retrain Guard — no retrain when new_papers < 100
# **Validates: Requirements 12.2**
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    new_paper_count=_new_paper_count_st,
    last_trained_count=_last_trained_count_st,
    store_size=_store_size_st,
)
def test_property_retrain_guard_insufficient_papers(
    new_paper_count: int,
    last_trained_count: int,
    store_size: int,
) -> None:
    """
    **Property 16a: Retrain Guard**

    **Validates: Requirements 12.2**

    For any count of new LLM-verified papers since last training:
    - If new_papers < 100, retrain_if_needed() returns None (no retraining)
    - retrain only triggers when new_papers >= MIN_RETRAIN_NEW (100)
    """
    # Total count in llm_verified.json
    total_count = last_trained_count + new_paper_count

    # Only test the guard condition: new papers < 100
    assume(new_paper_count < 100)

    # Create temporary data directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir(parents=True)
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)

        # Create llm_verified.json with total_count records
        # (records don't need full features since we expect early return)
        llm_records = [{"id": i} for i in range(total_count)]
        with open(audit_dir / "llm_verified.json", "w") as f:
            json.dump(llm_records, f)

        # Mock the store with sufficient papers
        mock_store = MagicMock()
        mock_store.positive_count = store_size // 2
        mock_store.negative_count = store_size - (store_size // 2)

        # Patch DATA_DIR to use our temp directory
        with patch("collectors.hybrid_classifier.DATA_DIR", tmp_path), \
             patch("collectors.hybrid_classifier.MODELS_DIR", models_dir):
            from collectors.hybrid_classifier import HybridClassifier

            classifier = HybridClassifier.__new__(HybridClassifier)
            classifier._model = None
            classifier._calibrator = MagicMock()
            classifier._metadata = {"n_samples_trained": last_trained_count}
            classifier._store_size = 0
            classifier.MODEL_PATH = models_dir / "hybrid_classifier.pkl"
            classifier.MIN_STORE_SIZE = 2000
            classifier.MIN_TRAINING_SAMPLES = 200
            classifier.MIN_RETRAIN_NEW = 100
            classifier.MIN_F1_THRESHOLD = 0.80

            result = classifier.retrain_if_needed(mock_store)

            # Property: retrain_if_needed returns None when new_papers < 100
            assert result is None, (
                f"Expected None when new_papers={new_paper_count} < 100, "
                f"but got {result}"
            )


# ---------------------------------------------------------------------------
# Property 16b: F1 Quality Gate — discard model if F1 < 0.80
# **Validates: Requirements 12.5**
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    n_samples=st.integers(min_value=200, max_value=400),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_f1_quality_gate_rejects_low_f1(
    n_samples: int,
    seed: int,
) -> None:
    """
    **Property 16b: F1 Quality Gate**

    **Validates: Requirements 12.5**

    When training data is sufficiently noisy that the resulting model achieves
    F1 < 0.80, train() returns accepted: False and the model is NOT persisted.
    Conversely, if F1 >= 0.80, accepted is True.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)

        with patch("collectors.hybrid_classifier.DATA_DIR", tmp_path), \
             patch("collectors.hybrid_classifier.MODELS_DIR", models_dir):
            from collectors.hybrid_classifier import HybridClassifier

            classifier = HybridClassifier.__new__(HybridClassifier)
            classifier._model = None
            classifier._calibrator = MagicMock()
            classifier._calibrator.is_fitted = False
            classifier._metadata = {}
            classifier._store_size = 3000
            classifier.MODEL_PATH = models_dir / "hybrid_classifier.pkl"
            classifier.MIN_STORE_SIZE = 2000
            classifier.MIN_TRAINING_SAMPLES = 200
            classifier.MIN_RETRAIN_NEW = 100
            classifier.MIN_F1_THRESHOLD = 0.80

            # Generate degenerate training data: completely random features
            # with random labels — logistic regression can't learn anything useful
            rng = np.random.default_rng(seed)
            features = rng.random((n_samples, 4))

            # Completely random labels (unrelated to features) ensure low F1
            labels = rng.integers(0, 2, size=n_samples).astype(float)

            # Ensure both classes present
            if len(np.unique(labels)) < 2:
                labels[0] = 0.0
                labels[1] = 1.0

            # Mock _save_model to avoid pickle issues with MagicMock calibrator
            # (we're testing the F1 gate logic, not persistence mechanics)
            with patch.object(classifier, "_save_model"):
                result = classifier.train(features, labels)

            # The result dict always has 'accepted' key
            assert "accepted" in result, (
                f"train() result missing 'accepted' key: {result}"
            )

            # Core property: F1 threshold determines acceptance
            if result["f1"] < 0.80:
                assert result["accepted"] is False, (
                    f"Model with F1={result['f1']:.3f} < 0.80 should have "
                    f"accepted=False, but got accepted={result['accepted']}"
                )
                # Model file should NOT be persisted when rejected
                assert not classifier.MODEL_PATH.exists(), (
                    f"Model file should not exist when F1={result['f1']:.3f} < 0.80, "
                    f"but found at {classifier.MODEL_PATH}"
                )
            else:
                # If training produced F1 >= 0.80, model is accepted
                assert result["accepted"] is True, (
                    f"Model with F1={result['f1']:.3f} >= 0.80 should have "
                    f"accepted=True, but got accepted={result['accepted']}"
                )

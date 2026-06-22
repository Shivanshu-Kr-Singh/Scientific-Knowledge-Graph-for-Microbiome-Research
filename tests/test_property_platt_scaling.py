"""
Property Tests for Platt Scaling Calibration

**Validates: Requirements 10.3, 11.2**

Property 15: Platt Scaling Monotonicity and Range
  For any fitted PlattCalibrator and any pair of logits (logit_a, logit_b)
  where logit_a < logit_b:
  - calibrated(logit_a) <= calibrated(logit_b)  (monotonicity)
  - 0.0 <= calibrated(logit_a) <= 1.0            (range)
  - 0.0 <= calibrated(logit_b) <= 1.0            (range)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from collectors.calibration import PlattCalibrator, CalibrationResult


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Training logits: 200+ values in [-5, 5]
_training_logits_st = st.lists(
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    min_size=200,
    max_size=300,
)

# Binary labels for training
_training_labels_st = st.lists(
    st.integers(min_value=0, max_value=1),
    min_size=200,
    max_size=300,
)

# Pairs of logits for testing monotonicity (reasonable range)
_logit_pair_st = st.tuples(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)


# ---------------------------------------------------------------------------
# Property 15: Platt Scaling Monotonicity and Range
# **Validates: Requirements 10.3, 11.2**
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    training_logits=_training_logits_st,
    training_labels=_training_labels_st,
    logit_pair=_logit_pair_st,
)
def test_property_platt_scaling_monotonicity_and_range(
    training_logits: list[float],
    training_labels: list[int],
    logit_pair: tuple[float, float],
) -> None:
    """
    **Property 15: Platt Scaling Monotonicity and Range**

    **Validates: Requirements 10.3, 11.2**

    For any fitted PlattCalibrator:
      - If logit_a < logit_b, then calibrated(logit_a) <= calibrated(logit_b) (monotonicity)
      - All calibrated outputs are in [0, 1] (range)
    """
    # Ensure training logits and labels have the same size
    min_len = min(len(training_logits), len(training_labels))
    training_logits = training_logits[:min_len]
    training_labels = training_labels[:min_len]

    # Need at least 200 samples (the calibrator's minimum)
    assume(min_len >= 200)

    # Need both classes present for fitting to succeed
    assume(0 in training_labels and 1 in training_labels)

    # Ensure logit_a < logit_b
    logit_a, logit_b = logit_pair
    assume(logit_a < logit_b)

    # Fit the calibrator
    calibrator = PlattCalibrator()
    logits_arr = np.array(training_logits, dtype=np.float64)
    labels_arr = np.array(training_labels, dtype=np.float64)

    result = calibrator.fit(logits_arr, labels_arr)

    # The fit should succeed given our constraints
    assume(result is not None)
    assert calibrator.is_fitted

    # Calibrate both logits
    cal_a = calibrator.calibrate(logit_a)
    cal_b = calibrator.calibrate(logit_b)

    # Property: Monotonicity — calibrated(logit_a) <= calibrated(logit_b)
    assert cal_a <= cal_b, (
        f"Monotonicity violated: calibrate({logit_a}) = {cal_a} > "
        f"calibrate({logit_b}) = {cal_b}"
    )

    # Property: Range — all outputs in [0, 1]
    assert 0.0 <= cal_a <= 1.0, (
        f"Range violated: calibrate({logit_a}) = {cal_a}, expected in [0, 1]"
    )
    assert 0.0 <= cal_b <= 1.0, (
        f"Range violated: calibrate({logit_b}) = {cal_b}, expected in [0, 1]"
    )

"""
Property Tests for Drift Monitor

**Validates: Requirements 14.1, 14.4**

Property 18: Drift Monitor Sampling Guarantee
  For any population of automated decisions from the past month, the Drift
  Monitor SHALL select a sample of size max(ceil(population_size * 0.01), 10),
  ensuring at least 10 papers are always selected when the population has
  ≥10 papers.

  Specifically:
  - If population == 0 → sample_size == 0
  - If 0 < population < 10 → sample_size == population (sample all)
  - If population >= 10 → sample_size == max(ceil(population * 0.01), 10)
  - Sample size is always ≥ 10 when population ≥ 10
"""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.drift_monitor import DriftMonitor


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Population sizes from 0 to 5000
_population_st = st.integers(min_value=0, max_value=5000)


def _make_fake_decisions(n: int) -> list[dict]:
    """Generate a list of N fake automated decision records."""
    decisions = []
    for i in range(n):
        decisions.append({
            "doi": f"10.1000/fake-{i}",
            "pmid": str(10000000 + i),
            "title": f"Fake Paper {i}",
            "stage": "stage2_rules",
            "decision": "keep" if i % 2 == 0 else "reject",
        })
    return decisions


# ---------------------------------------------------------------------------
# Property 18: Drift Monitor Sampling Guarantee
# **Validates: Requirements 14.1, 14.4**
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=10000)
@given(population_size=_population_st)
def test_property_drift_monitor_sampling_guarantee(
    tmp_path_factory,
    population_size: int,
) -> None:
    """
    **Property 18: Drift Monitor Sampling Guarantee**

    **Validates: Requirements 14.1, 14.4**

    For any population size:
    - population == 0 → sample_size == 0
    - 0 < population < 10 → sample_size == population
    - population >= 10 → sample_size == max(ceil(population * 0.01), 10)
    - sample_size is always >= 10 when population >= 10
    """
    fake_decisions = _make_fake_decisions(population_size)

    monitor = DriftMonitor()
    # Use a temporary directory for audit output
    audit_dir = tmp_path_factory.mktemp("drift_audit")
    monitor._audit_dir = audit_dir

    # Mock _load_automated_decisions to return our generated fake data
    with patch.object(monitor, "_load_automated_decisions", return_value=fake_decisions):
        result = monitor.run()

    sample_size = result["sample_size"]

    if population_size == 0:
        # No papers → no sample
        assert sample_size == 0, (
            f"Expected sample_size=0 for population=0, got {sample_size}"
        )
    elif population_size < 10:
        # Fewer than MIN_SAMPLE_SIZE → sample all
        assert sample_size == population_size, (
            f"Expected sample_size={population_size} for population < 10, "
            f"got {sample_size}"
        )
    else:
        # population >= 10 → max(ceil(population * 0.01), 10)
        expected = max(ceil(population_size * 0.01), 10)
        assert sample_size == expected, (
            f"Expected sample_size={expected} for population={population_size}, "
            f"got {sample_size}"
        )

        # Additionally verify: sample_size is always >= 10
        assert sample_size >= 10, (
            f"Sample size must be >= 10 when population >= 10, "
            f"got {sample_size} for population={population_size}"
        )


@settings(max_examples=50, deadline=10000)
@given(population_size=st.integers(min_value=10, max_value=5000))
def test_property_drift_monitor_minimum_10_guarantee(
    tmp_path_factory,
    population_size: int,
) -> None:
    """
    **Property 18: Drift Monitor Sampling Guarantee (Minimum 10 invariant)**

    **Validates: Requirements 14.4**

    For any population >= 10, the sample size SHALL always be at least 10.
    This is the critical safety guarantee ensuring meaningful drift review.
    """
    fake_decisions = _make_fake_decisions(population_size)

    monitor = DriftMonitor()
    audit_dir = tmp_path_factory.mktemp("drift_min10")
    monitor._audit_dir = audit_dir

    with patch.object(monitor, "_load_automated_decisions", return_value=fake_decisions):
        result = monitor.run()

    sample_size = result["sample_size"]

    # The key guarantee: at least 10 papers when population >= 10
    assert sample_size >= 10, (
        f"VIOLATED: sample_size={sample_size} < 10 for population={population_size}. "
        f"Drift monitor must always sample at least 10 papers when population >= 10."
    )

    # Sample can never exceed population
    assert sample_size <= population_size, (
        f"sample_size={sample_size} exceeds population={population_size}"
    )

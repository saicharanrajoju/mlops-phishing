"""Tests for the KS + PSI drift monitor."""

import numpy as np
import pandas as pd

from phishsentinel.monitoring.drift_monitor import (
    generate_drift_report,
    population_stability_index,
)


def _frame(seed: int, shift: float = 0.0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(shift, 1, n),
            "f2": rng.normal(shift, 1, n),
            "Result": rng.integers(0, 2, n),
        }
    )


def test_no_drift_on_same_distribution():
    ref = _frame(seed=1)
    cur = _frame(seed=2)  # same distribution, different sample
    report = generate_drift_report(ref, cur)
    assert report["dataset_drift"] is False
    assert report["n_features"] == 2  # target column excluded


def test_drift_detected_on_shifted_distribution():
    ref = _frame(seed=1, shift=0.0)
    cur = _frame(seed=2, shift=3.0)  # large mean shift -> drift
    report = generate_drift_report(ref, cur)
    assert report["dataset_drift"] is True
    assert report["n_drifted"] >= 1


def test_psi_zero_for_identical_arrays():
    x = np.linspace(0, 1, 1000)
    assert population_stability_index(x, x) == 0.0


def test_psi_positive_for_shifted_arrays():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(2, 1, 1000)
    assert population_stability_index(ref, cur) > 0.2

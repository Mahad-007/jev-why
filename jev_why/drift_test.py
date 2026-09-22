from __future__ import annotations

import numpy as np
import pytest

from jev_why.drift import calibration_z, cusum, drift_report, population_stability_index


def test_an_unchanged_distribution_does_not_alarm() -> None:
    rng = np.random.default_rng(0)
    base, live = rng.beta(2, 5, 2000), rng.beta(2, 5, 800)
    report = drift_report(base, live, permutations=300)
    assert not report.prediction_drift
    assert report.psi < 0.1


def test_a_moved_distribution_alarms_on_every_statistic() -> None:
    rng = np.random.default_rng(0)
    base, live = rng.beta(2, 5, 2000), rng.beta(5, 2, 800)
    report = drift_report(base, live, permutations=300)
    assert report.prediction_drift
    assert report.psi > 0.25
    assert report.js_distance > 0.3
    assert "moved materially" in report.summary()


def test_prediction_drift_is_not_claimed_to_be_calibration_drift() -> None:
    """The distinction the whole module is built around. Without outcomes, a
    quiet result means the inputs look similar, not that the probabilities are
    still trustworthy."""
    rng = np.random.default_rng(1)
    report = drift_report(rng.beta(2, 5, 1000), rng.beta(2, 5, 500), permutations=100)
    assert report.calibration_z is None
    assert "not that the probabilities are still trustworthy" in report.summary()


def test_calibration_z_needs_only_an_aggregate_count() -> None:
    """What makes this deployable: no per-row label join, just how many cases
    turned out positive."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 500)
    honest = int(p.sum())
    assert abs(calibration_z(p, honest)) < 2
    assert calibration_z(p, honest + 60) > 3
    assert calibration_z(p, honest - 60) < -3


def test_calibration_drift_is_reported_in_plain_words() -> None:
    rng = np.random.default_rng(3)
    base, live = rng.uniform(0, 1, 1000), rng.uniform(0, 1, 500)
    report = drift_report(base, live, observed_positives=int(live.sum()) + 80, permutations=50)
    assert report.calibration_drift
    assert "more positives than the probabilities implied" in report.summary()
    assert "Re-fit the threshold" in report.summary()


def test_a_degenerate_distribution_does_not_divide_by_zero() -> None:
    assert calibration_z([0.0, 0.0, 0.0], 0) == 0.0


def test_psi_is_zero_for_identical_histograms() -> None:
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_cusum_accumulates_a_drift_too_small_to_see_per_window() -> None:
    """Each window alone is unremarkable; twenty of them are not."""
    small = [0.8] * 40
    assert all(abs(z) < 3 for z in small), "no single window would alarm"
    assert cusum(small, k=0.5, h=5.0).alarmed


def test_cusum_ignores_noise_around_zero() -> None:
    assert not cusum([0.4, -0.5, 0.2, -0.3] * 25, k=0.5, h=5.0).alarmed


def test_cusum_detects_drift_in_both_directions() -> None:
    assert cusum([-1.0] * 20, k=0.5, h=5.0).alarmed
    assert cusum([1.0] * 20, k=0.5, h=5.0).alarmed

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from jev_why import calibration as c


def _overconfident(n: int = 4000, sharpen: float = 1.8, seed: int = 0) -> tuple[Any, Any, Any]:
    """A model whose ranking is right but whose probabilities are too extreme.
    This is precisely the shape of the public complaint about Jev."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    logit = np.log(true_p / (1 - true_p))
    return 1 / (1 + np.exp(-logit * sharpen)), y, true_p


def test_a_perfectly_calibrated_model_scores_near_zero_ece() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.02, 0.98, 20_000)
    y = (rng.uniform(size=20_000) < p).astype(int)
    assert c.ece(p, y) < 0.02
    assert c.calibration_slope_intercept(p, y)[0] == pytest.approx(1.0, abs=0.1)


def test_overconfidence_shows_up_as_a_slope_below_one() -> None:
    p, y, _ = _overconfident()
    slope, _ = c.calibration_slope_intercept(p, y)
    assert slope < 0.8
    assert "overconfident" in c.report(p, y).summary()


def test_underconfidence_shows_up_as_a_slope_above_one() -> None:
    p, y, _ = _overconfident(sharpen=0.5)
    assert c.calibration_slope_intercept(p, y)[0] > 1.2


def test_brier_decomposition_separates_ranking_from_calibration() -> None:
    """The whole reason this is the flagship metric: it splits apart exactly
    the two things people disagree about."""
    p, y, _ = _overconfident()
    parts = c.brier_decomposition(p, y)
    assert parts.resolution > parts.reliability
    assert parts.reads_as() == "strong ranking, weaker probabilities"
    assert parts.total == pytest.approx(
        parts.reliability - parts.resolution + parts.uncertainty, abs=0.02
    )


def test_auroc_is_untouched_by_recalibration() -> None:
    """Ranking quality is invariant to any monotone squeeze, which is why a bad
    ECE with a good AUROC calls for a calibrator, not a different model."""
    p, y, _ = _overconfident()
    calibrated = c.fit_isotonic(p, y)(p)
    assert c.auroc(p, y) == pytest.approx(c.auroc(calibrated, y), abs=0.01)


def test_auroc_of_a_constant_predictor_is_one_half() -> None:
    assert c.auroc([0.5] * 100, [0, 1] * 50) == pytest.approx(0.5)


def test_isotonic_beats_platt_on_a_shape_a_logistic_cannot_fit() -> None:
    p, y, _ = _overconfident()
    baseline = c.ece(p, y)
    assert c.ece(c.fit_platt(p, y)(p), y) < baseline
    assert c.ece(c.fit_isotonic(p, y)(p), y) < baseline


def test_in_sample_isotonic_looks_perfect_and_should_not_be_trusted() -> None:
    """Fitting and scoring on the same data makes isotonic calibration look
    flawless by construction. Held-out data is the only honest measurement."""
    p, y, _ = _overconfident(n=600)
    assert c.ece(c.fit_isotonic(p, y)(p), y) < 1e-6

    split = 300
    fitted = c.fit_isotonic(p[:split], y[:split])
    assert c.ece(fitted(p[split:]), y[split:]) > 1e-6


def test_quantile_bins_are_the_default_because_equal_width_leaves_bins_empty() -> None:
    rng = np.random.default_rng(5)
    skewed = rng.beta(1, 20, 2000)
    y = (rng.uniform(size=2000) < skewed).astype(int)
    uniform = c.calibration_curve(skewed, y, strategy="uniform")
    quantile = c.calibration_curve(skewed, y, strategy="quantile")
    assert len(uniform.non_empty) < len(uniform.bins)
    assert len(quantile.non_empty) == len(quantile.bins)


def test_adaptive_ece_does_not_depend_on_one_arbitrary_bin_count() -> None:
    p, y, _ = _overconfident()
    assert c.adaptive_ece(p, y) == pytest.approx(c.ece(p, y), abs=0.03)


def test_wilson_intervals_stay_inside_zero_and_one() -> None:
    assert c.wilson_interval(0, 10)[0] == 0.0
    assert c.wilson_interval(10, 10)[1] == 1.0


def test_noul_confidence_is_margin_from_your_threshold_not_from_a_half() -> None:
    """The API returns no confidence for a noul, and real deployments act on a
    tuned threshold rather than 0.5."""
    assert c.noul_confidence(0.5) == pytest.approx(0.0)
    assert c.noul_confidence(1.0) == pytest.approx(1.0)
    assert c.noul_confidence(0.9, threshold=0.9) == pytest.approx(0.0)
    assert c.noul_confidence(0.95, threshold=0.9) > 0


def test_mismatched_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="probabilities and"):
        c.ece([0.1, 0.2], [1])
    with pytest.raises(ValueError, match="no observations"):
        c.ece([], [])

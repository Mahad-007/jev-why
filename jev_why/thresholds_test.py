from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from jev_why.thresholds import (
    clopper_pearson_lower,
    coverage_curve,
    selective_thresholds,
    threshold_for_precision,
)


def _calibrated(n: int = 2000, seed: int = 1) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    return p, (rng.uniform(size=n) < p).astype(int)


def test_clopper_pearson_matches_published_values() -> None:
    assert clopper_pearson_lower(9, 10) == pytest.approx(0.606, abs=0.005)
    assert clopper_pearson_lower(90, 100) == pytest.approx(0.836, abs=0.005)


def test_the_bound_tightens_as_evidence_accumulates() -> None:
    """Why the bound, not the point estimate: nine out of ten is not the same
    evidence as nine hundred out of a thousand, though both read as 0.9."""
    same_rate = [clopper_pearson_lower(int(0.9 * n), n) for n in (10, 100, 1000, 10_000)]
    assert same_rate == sorted(same_rate)
    assert same_rate[0] < 0.7 < same_rate[-1]


def test_a_bound_based_threshold_is_never_looser_than_a_point_based_one() -> None:
    p, y = _calibrated()
    strict = threshold_for_precision(p, y, target=0.9, bound="clopper_pearson")
    loose = threshold_for_precision(p, y, target=0.9, bound="point")
    assert strict.threshold >= loose.threshold
    assert strict.coverage <= loose.coverage


def test_the_chosen_threshold_actually_delivers_the_target() -> None:
    p, y = _calibrated()
    result = threshold_for_precision(p, y, target=0.9)
    assert result.achieved
    assert result.precision_lower_bound >= 0.9
    assert (y[p >= result.threshold].mean()) >= 0.85


def test_among_qualifying_thresholds_the_most_automation_wins() -> None:
    p, y = _calibrated()
    result = threshold_for_precision(p, y, target=0.8)
    assert result.coverage > threshold_for_precision(p, y, target=0.95).coverage


def test_an_unreachable_target_says_so_instead_of_pretending() -> None:
    p, y = _calibrated(n=200)
    result = threshold_for_precision(p, y, target=0.999)
    assert not result.achieved
    assert "no threshold reaches" in result.explain()


def test_a_minimum_coverage_constraint_is_respected() -> None:
    p, y = _calibrated()
    result = threshold_for_precision(p, y, target=0.7, min_coverage=0.5)
    assert not result.achieved or result.coverage >= 0.5


def test_selective_thresholds_leave_an_abstain_band() -> None:
    """The shape of a real deployment: automate the confident ends, review the
    middle."""
    p, y = _calibrated()
    result = selective_thresholds(p, y, target_precision=0.9, target_npv=0.9)
    assert result.reject_below < result.accept_above
    assert 0 < result.abstain_rate < 1
    assert "review" in result.explain()


def test_the_sweep_is_bounded_so_large_logs_stay_interactive() -> None:
    p, y = _calibrated(n=50_000)
    result = threshold_for_precision(p, y, target=0.9, max_candidates=64)
    assert result.achieved


def test_coverage_falls_as_the_threshold_rises() -> None:
    p, y = _calibrated()
    curve = coverage_curve(p, y, steps=20)
    coverages = [point.coverage for point in curve]
    assert coverages == sorted(coverages, reverse=True)

"""Has the thing you calibrated against changed?

Pure.

The honest framing first, because packages that blur it are selling a proxy:
**prediction drift needs no labels; calibration drift needs outcomes.** You
cannot detect that probabilities have stopped meaning what they said without
observing something about what actually happened.

What makes calibration monitoring fail in practice is not the statistics, it is
the per-item label join nobody wants to build. So the recommended monitor here
needs only an aggregate count of how many cases turned out positive -- a number
most teams already have on a dashboard.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

import numpy as np
from numpy.typing import NDArray

EPS = 1e-9

Floats = Sequence[float] | NDArray[np.float64]


def quantile_bins(baseline: Floats, bins: int = 10) -> NDArray[np.float64]:
    """Bin edges from the baseline's own quantiles, so each bin carries equal
    baseline mass and the statistic is not an artifact of bin placement."""
    edges = np.quantile(np.asarray(baseline, dtype=np.float64), np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.unique(edges)


def _histogram(values: Floats, edges: NDArray[np.float64]) -> NDArray[np.float64]:
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=edges)
    total = counts.sum()
    return counts / total if total else counts.astype(np.float64)


def jensen_shannon_distance(p: Floats, q: Floats) -> float:
    """Square root of the JS divergence: a true metric, bounded in [0, 1]."""
    a = np.asarray(p, dtype=np.float64)
    b = np.asarray(q, dtype=np.float64)
    m = (a + b) / 2

    def kl(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / np.maximum(y[mask], EPS))))

    return float(np.sqrt(max(0.0, 0.5 * kl(a, m) + 0.5 * kl(b, m))))


def population_stability_index(p: Floats, q: Floats) -> float:
    """PSI, reported alongside JS because risk and compliance teams already know
    its bands (<0.1 stable, 0.1-0.25 watch, >0.25 investigate).

    It is not the headline, because it has no null distribution and those bands
    are convention rather than inference.
    """
    a = np.maximum(np.asarray(p, dtype=np.float64), EPS)
    b = np.maximum(np.asarray(q, dtype=np.float64), EPS)
    return float(np.sum((b - a) * np.log(b / a)))


def calibration_z(probabilities: Floats, observed_positives: int) -> float:
    """How surprising the outcome count is, if the probabilities were honest.

    Under good calibration the number of positives is Poisson-binomial with
    mean sum(p) and variance sum(p(1-p)). This is the statistic worth putting
    on a dashboard: it needs one aggregate count rather than a per-row join,
    and it speaks plainly -- "you predicted 412 urgent tickets, 561 actually
    were, z = +6.8".
    """
    p = np.asarray(probabilities, dtype=np.float64)
    expected = float(p.sum())
    variance = float(np.sum(p * (1 - p)))
    if variance <= 0:
        return 0.0
    return float((observed_positives - expected) / np.sqrt(variance))


@dataclass(frozen=True)
class CusumResult:
    alarm_index: int | None
    peak: float
    positive: tuple[float, ...]
    negative: tuple[float, ...]

    @property
    def alarmed(self) -> bool:
        return self.alarm_index is not None


def cusum(values: Floats, *, k: float = 0.5, h: float = 5.0) -> CusumResult:
    """Two-sided CUSUM over a stream of z-scores.

    A slow drift produces per-window scores too small to notice individually
    and impossible to miss once accumulated. Watching each window alone is how
    a gradual decay goes unreported for a quarter.
    """
    high = low = 0.0
    highs: list[float] = []
    lows: list[float] = []
    alarm: int | None = None
    for i, z in enumerate(values):
        high = max(0.0, high + z - k)
        low = min(0.0, low + z + k)
        highs.append(high)
        lows.append(low)
        if alarm is None and (high > h or low < -h):
            alarm = i
    peak = max(max(highs, default=0.0), abs(min(lows, default=0.0)))
    return CusumResult(alarm, peak, tuple(highs), tuple(lows))


@dataclass(frozen=True)
class DriftReport:
    n_baseline: int
    n_live: int
    js_distance: float
    psi: float
    p_value: float | None
    calibration_z: float | None
    permutations: int

    @property
    def prediction_drift(self) -> bool:
        return self.p_value is not None and self.p_value <= 0.05

    @property
    def calibration_drift(self) -> bool:
        return self.calibration_z is not None and abs(self.calibration_z) > 3.0

    def summary(self) -> str:
        lines = [
            f"distribution: JS {self.js_distance:.3f}, PSI {self.psi:.3f}"
            + (f", p={self.p_value:.3f}" if self.p_value is not None else "")
        ]
        if self.psi > 0.25:
            lines.append("PSI above 0.25: inputs have moved materially.")
        if self.calibration_z is not None:
            direction = "more" if self.calibration_z > 0 else "fewer"
            lines.append(
                f"outcomes: z={self.calibration_z:+.2f} "
                f"({direction} positives than the probabilities implied)"
            )
            if self.calibration_drift:
                lines.append("Calibration has drifted. Re-fit the threshold before trusting it.")
        elif not self.prediction_drift:
            lines.append(
                "No label information supplied, so this says the inputs look "
                "similar -- not that the probabilities are still trustworthy."
            )
        return " ".join(lines)


def drift_report(
    baseline: Floats,
    live: Floats,
    *,
    observed_positives: int | None = None,
    bins: int = 10,
    permutations: int = 2000,
    seed: int = 0,
) -> DriftReport:
    base = np.asarray(baseline, dtype=np.float64)
    current = np.asarray(live, dtype=np.float64)
    edges = quantile_bins(base, bins)

    base_hist = _histogram(base, edges)
    live_hist = _histogram(current, edges)
    observed = jensen_shannon_distance(base_hist, live_hist)

    p_value: float | None = None
    if permutations > 0 and base.size and current.size:
        rng = Random(seed)
        pooled = np.concatenate([base, current])
        n_live = current.size
        at_least = 0
        for _ in range(permutations):
            index = list(range(pooled.size))
            rng.shuffle(index)
            shuffled = pooled[index]
            candidate = jensen_shannon_distance(
                _histogram(shuffled[n_live:], edges), _histogram(shuffled[:n_live], edges)
            )
            at_least += candidate >= observed
        p_value = (1 + at_least) / (1 + permutations)

    z = calibration_z(current, observed_positives) if observed_positives is not None else None

    return DriftReport(
        n_baseline=int(base.size),
        n_live=int(current.size),
        js_distance=observed,
        psi=population_stability_index(base_hist, live_hist),
        p_value=p_value,
        calibration_z=z,
        permutations=permutations,
    )


__all__ = [
    "CusumResult",
    "DriftReport",
    "calibration_z",
    "cusum",
    "drift_report",
    "jensen_shannon_distance",
    "population_stability_index",
    "quantile_bins",
]

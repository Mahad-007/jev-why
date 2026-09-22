"""Turning a calibrated probability into a decision you can defend.

Pure.

Two things here that a metrics-only package cannot give you. First, the
threshold is chosen so the *lower confidence bound* on precision clears the
target, not the point estimate -- picking on the point estimate systematically
selects the threshold where you got lucky on a handful of samples. Second,
selective classification: a high threshold to auto-accept, a low one to
auto-reject, and an abstain band in between that goes to a human or a slower
model. That is the shape of every real deployment of a model like this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, lgamma, log, log1p
from typing import Literal

import numpy as np

from jev_why.calibration import Floats, Labels, wilson_interval

Bound = Literal["clopper_pearson", "wilson", "point"]


def _log_beta(a: float, b: float) -> float:
    return float(lgamma(a) + lgamma(b) - lgamma(a + b))


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b), by the Lentz continued fraction.

    Written out rather than pulled from scipy because scipy is a heavy
    dependency for one function, and this is the only special function the
    package needs. Converges in tens of iterations, so the threshold sweep
    stays interactive.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # The continued fraction converges fastest on the side where x is small.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    front = exp(a * log(x) + b * log1p(-x) - _log_beta(a, b)) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))

        d = 1.0 + numerator * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = 1e-30 if abs(c) < 1e-30 else c

        step = c * d
        f *= step
        if abs(1.0 - step) < 1e-12:
            break
    return float(min(1.0, max(0.0, front * (f - 1.0))))


def clopper_pearson_lower(successes: int, n: int, alpha: float = 0.05) -> float:
    """Exact lower bound on a binomial proportion.

    Conservative by construction, which is the point: a threshold picked on an
    optimistic point estimate is a threshold that fails in production. The
    bound is the alpha-quantile of Beta(k, n-k+1), found by bisection on the
    incomplete beta.
    """
    if n == 0 or successes <= 0:
        return 0.0
    if successes >= n:
        return float(alpha ** (1.0 / n))

    a, b = float(successes), float(n - successes + 1)
    lo, hi = 0.0, successes / n
    for _ in range(80):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return float(lo)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision_point: float
    precision_lower_bound: float
    recall: float
    coverage: float
    n_above: int
    achieved: bool

    def explain(self) -> str:
        if not self.achieved:
            return (
                "no threshold reaches the target precision with this much data. "
                f"The best available is {self.precision_point:.3f} at "
                f"{self.threshold:.3f}, covering {self.coverage:.0%}."
            )
        return (
            f"act automatically above {self.threshold:.3f}: precision at least "
            f"{self.precision_lower_bound:.3f} on {self.n_above} of "
            f"{round(self.n_above / max(self.coverage, 1e-9))} decisions "
            f"({self.coverage:.0%} coverage), sending the rest for review."
        )


def _lower_bound(successes: int, n: int, alpha: float, bound: Bound) -> float:
    if n == 0:
        return 0.0
    if bound == "point":
        return successes / n
    if bound == "wilson":
        return wilson_interval(successes, n, z=1.96)[0]
    return clopper_pearson_lower(successes, n, alpha)


def threshold_for_precision(
    p: Sequence[float],
    y: Sequence[int],
    *,
    target: float = 0.9,
    alpha: float = 0.05,
    bound: Bound = "clopper_pearson",
    min_coverage: float = 0.0,
    max_candidates: int = 256,
) -> ThresholdResult:
    probs = np.asarray(p, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    total_positive = float(labels.sum())

    # One descending pass gives the positive count above every cut, so the
    # sweep is O(n log n) rather than O(n^2).
    order = np.argsort(-probs)
    sorted_probs, sorted_labels = probs[order], labels[order]
    cumulative = np.cumsum(sorted_labels)
    # Only the last index of each run of equal probabilities is a real cut.
    cuts = np.flatnonzero(np.diff(sorted_probs) < 0)
    cuts = np.append(cuts, probs.size - 1)
    if cuts.size > max_candidates:
        cuts = cuts[np.linspace(0, cuts.size - 1, max_candidates).astype(int)]

    best: ThresholdResult | None = None
    fallback: ThresholdResult | None = None
    for cut in cuts:
        n_above = int(cut) + 1
        candidate = float(sorted_probs[cut])
        successes = int(cumulative[cut])
        coverage = n_above / probs.size
        result = ThresholdResult(
            threshold=candidate,
            precision_point=successes / n_above,
            precision_lower_bound=_lower_bound(successes, n_above, alpha, bound),
            recall=successes / total_positive if total_positive else 0.0,
            coverage=coverage,
            n_above=n_above,
            achieved=False,
        )
        if fallback is None or result.precision_point > fallback.precision_point:
            fallback = result
        # Among thresholds that clear the bar, prefer the one that keeps the
        # most work automated.
        qualifies = result.precision_lower_bound >= target and coverage >= min_coverage
        if qualifies and (best is None or result.coverage > best.coverage):
            best = ThresholdResult(**{**result.__dict__, "achieved": True})

    if best is not None:
        return best
    assert fallback is not None
    return fallback


@dataclass(frozen=True)
class SelectiveResult:
    accept_above: float
    reject_below: float
    abstain_rate: float
    accept_precision: float
    reject_npv: float

    def explain(self) -> str:
        return (
            f"auto-accept above {self.accept_above:.3f}, auto-reject below "
            f"{self.reject_below:.3f}, send {self.abstain_rate:.0%} for review."
        )


def selective_thresholds(
    p: Sequence[float],
    y: Sequence[int],
    *,
    target_precision: float = 0.95,
    target_npv: float = 0.95,
    alpha: float = 0.05,
) -> SelectiveResult:
    probs = np.asarray(p, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)

    accept = threshold_for_precision(p, y, target=target_precision, alpha=alpha)
    flipped = threshold_for_precision(
        (1.0 - probs).tolist(),
        (1.0 - labels).astype(int).tolist(),
        target=target_npv,
        alpha=alpha,
    )
    reject_below = 1.0 - flipped.threshold

    abstain = float(((probs < accept.threshold) & (probs > reject_below)).mean())
    return SelectiveResult(
        accept_above=accept.threshold,
        reject_below=reject_below,
        abstain_rate=abstain,
        accept_precision=accept.precision_lower_bound,
        reject_npv=flipped.precision_lower_bound,
    )


@dataclass(frozen=True)
class CoveragePoint:
    threshold: float
    coverage: float
    precision: float


def coverage_curve(p: Floats, y: Labels, *, steps: int = 50) -> tuple[CoveragePoint, ...]:
    probs = np.asarray(p, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    out: list[CoveragePoint] = []
    for candidate in np.linspace(probs.min(), probs.max(), steps):
        mask = probs >= candidate
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(CoveragePoint(float(candidate), n / probs.size, float(labels[mask].sum()) / n))
    return tuple(out)


__all__ = [
    "Bound",
    "CoveragePoint",
    "SelectiveResult",
    "ThresholdResult",
    "clopper_pearson_lower",
    "coverage_curve",
    "selective_thresholds",
    "threshold_for_precision",
]

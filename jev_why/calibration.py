"""Does Jev's confidence mean what it says -- on your data?

Pure: arrays in, dataclasses out.

The public argument about Jev is that its ranking is strong while its
probabilities are disputed. In Murphy's decomposition of the Brier score that
is a precise, testable claim: high resolution with poor reliability. So the
Brier decomposition is the flagship here rather than a bare ECE, because it
separates exactly the two things people disagree about, and the calibration
slope turns the answer into one number anyone can act on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12
BinStrategy = Literal["quantile", "uniform"]

# Callers naturally hold numpy arrays, and a decision log read from disk is a
# list. Accept both rather than making every caller convert.
Floats = Sequence[float] | NDArray[np.float64]
Labels = Sequence[int] | NDArray[Any]


def _as_arrays(p: Floats, y: Labels) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    probs = np.asarray(p, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if probs.shape != labels.shape:
        raise ValueError(f"got {probs.size} probabilities and {labels.size} labels")
    if probs.size == 0:
        raise ValueError("no observations")
    return probs, labels


def _bin_edges(p: NDArray[np.float64], bins: int, strategy: BinStrategy) -> NDArray[np.float64]:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, bins + 1)
    # Equal-mass bins by default. Equal-width binning leaves high-probability
    # bins empty on a skewed noul distribution, which makes ECE a function of
    # the bin count as much as of the model.
    quantiles = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
    quantiles[0], quantiles[-1] = 0.0, 1.0
    return np.unique(quantiles)


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class CalibrationCurve:
    bins: tuple[Bin, ...]
    strategy: BinStrategy

    @property
    def non_empty(self) -> tuple[Bin, ...]:
        return tuple(b for b in self.bins if b.count > 0)


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Behaves at 0 and 1, where the normal
    approximation produces bounds outside [0, 1]."""
    if n == 0:
        return 0.0, 1.0
    phat = successes / n
    denominator = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def calibration_curve(
    p: Floats, y: Labels, *, bins: int = 10, strategy: BinStrategy = "quantile"
) -> CalibrationCurve:
    probs, labels = _as_arrays(p, y)
    edges = _bin_edges(probs, bins, strategy)
    out: list[Bin] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < len(edges) - 2 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        if count == 0:
            out.append(Bin(float(lo), float(hi), 0, 0.0, 0.0, 0.0, 1.0))
            continue
        successes = float(labels[mask].sum())
        low, high = wilson_interval(successes, count)
        out.append(
            Bin(
                float(lo), float(hi), count, float(probs[mask].mean()), successes / count, low, high
            )
        )
    return CalibrationCurve(tuple(out), strategy)


def ece(p: Floats, y: Labels, *, bins: int = 10, strategy: BinStrategy = "quantile") -> float:
    curve = calibration_curve(p, y, bins=bins, strategy=strategy)
    total = sum(b.count for b in curve.bins)
    if not total:
        return 0.0
    return sum(b.count / total * abs(b.observed - b.mean_predicted) for b in curve.non_empty)


def mce(p: Floats, y: Labels, *, bins: int = 10, strategy: BinStrategy = "quantile") -> float:
    curve = calibration_curve(p, y, bins=bins, strategy=strategy)
    gaps = [abs(b.observed - b.mean_predicted) for b in curve.non_empty]
    return max(gaps) if gaps else 0.0


def adaptive_ece(p: Floats, y: Labels, bin_counts: Sequence[int] = (5, 10, 15, 20)) -> float:
    """ECE averaged over several bin counts, so the headline number is not an
    artifact of one arbitrary choice."""
    return float(np.mean([ece(p, y, bins=b) for b in bin_counts]))


def brier(p: Floats, y: Labels) -> float:
    probs, labels = _as_arrays(p, y)
    return float(np.mean((probs - labels) ** 2))


@dataclass(frozen=True)
class BrierParts:
    reliability: float
    """How far the probabilities are from the truth. Lower is better. This is
    the part people dispute."""

    resolution: float
    """How much the predictions separate outcomes. Higher is better. This is
    the part people agree is strong."""

    uncertainty: float
    total: float

    def reads_as(self) -> str:
        if self.resolution > self.reliability * 3:
            return "strong ranking, weaker probabilities"
        if self.reliability > self.resolution:
            return "probabilities are the dominant source of error"
        return "ranking and calibration contribute comparably"


def brier_decomposition(p: Floats, y: Labels, *, bins: int = 10) -> BrierParts:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty."""
    probs, labels = _as_arrays(p, y)
    curve = calibration_curve(p, y, bins=bins)
    n = probs.size
    base = float(labels.mean())

    reliability = sum(b.count / n * (b.mean_predicted - b.observed) ** 2 for b in curve.non_empty)
    resolution = sum(b.count / n * (b.observed - base) ** 2 for b in curve.non_empty)
    uncertainty = base * (1 - base)
    return BrierParts(reliability, resolution, uncertainty, brier(p, y))


def log_loss(p: Floats, y: Labels, *, eps: float = 1e-12) -> float:
    probs, labels = _as_arrays(p, y)
    clipped = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def auroc(p: Floats, y: Labels) -> float:
    """Ranking quality, invariant to any monotone recalibration. If AUROC is
    high while ECE is bad, the fix is a calibrator, not a different model."""
    probs, labels = _as_arrays(p, y)
    positives, negatives = labels == 1, labels == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, probs.size + 1)
    # Average ranks within ties so a constant predictor scores 0.5, not 1.0.
    _, inverse, counts = np.unique(probs, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_slope_intercept(p: Floats, y: Labels) -> tuple[float, float]:
    """Logistic regression of the outcome on the log-odds of the prediction.

    Slope below 1 means overconfident, above 1 underconfident, exactly 1 means
    the probabilities carry the right amount of certainty. One interpretable
    number, with an established meaning in clinical prediction.
    """
    probs, labels = _as_arrays(p, y)
    x = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / (1 - np.clip(probs, 1e-6, 1 - 1e-6)))
    beta = np.zeros(2)
    design = np.column_stack([np.ones_like(x), x])
    for _ in range(100):
        eta = design @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-9)
        hessian = design.T @ (design * w[:, None]) + np.eye(2) * 1e-9
        gradient = design.T @ (labels - mu)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[1]), float(beta[0])


Calibrator = Callable[[Floats], NDArray[np.float64]]


def fit_platt(p: Floats, y: Labels) -> Calibrator:
    slope, intercept = calibration_slope_intercept(p, y)

    def apply(values: Floats) -> NDArray[np.float64]:
        v = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.asarray(1 / (1 + np.exp(-(intercept + slope * np.log(v / (1 - v))))))

    return apply


def fit_isotonic(p: Floats, y: Labels) -> Calibrator:
    """Pool-adjacent-violators. Non-parametric, so it corrects shapes a logistic
    fit cannot, at the cost of needing more data."""
    probs, labels = _as_arrays(p, y)
    order = np.argsort(probs)
    xs, ys = probs[order], labels[order]

    values = list(ys)
    weights = [1.0] * len(ys)
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        total = weights[i] + weights[i + 1]
        merged = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total
        values[i : i + 2] = [merged]
        weights[i : i + 2] = [total]
        i = max(i - 1, 0)

    fitted: list[float] = []
    for value, weight in zip(values, weights, strict=True):
        fitted.extend([value] * int(weight))
    knots_x, knots_y = xs, np.asarray(fitted, dtype=np.float64)

    def apply(new: Floats) -> NDArray[np.float64]:
        return np.asarray(np.interp(np.asarray(new, dtype=np.float64), knots_x, knots_y))

    return apply


def noul_confidence(p: float, *, threshold: float = 0.5) -> float:
    """Confidence for a noul, which the API does not return.

    The documented form is |2p-1|, which is margin from 0.5. Real deployments
    act on a tuned threshold rather than 0.5, so this generalises it: margin
    from *your* decision boundary, scaled so it runs 0 at the boundary to 1 at
    certainty.
    """
    return float(abs(p - threshold) / max(threshold, 1 - threshold))


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    ece: float
    adaptive_ece: float
    mce: float
    brier: float
    parts: BrierParts
    log_loss: float
    auroc: float
    slope: float
    intercept: float
    base_rate: float
    curve: CalibrationCurve

    distribution: tuple[int, ...] = ()
    """Counts over uniform bins of the predicted probability.

    Deliberately not the calibration bins: those are equal-mass, so their
    counts are identical by construction and a panel drawn from them would
    show a flat row of bars carrying no information. What a reader needs to
    interpret an ECE is where the predictions actually sit.
    """

    def summary(self) -> str:
        direction = "overconfident" if self.slope < 1 else "underconfident"
        return (
            f"{self.n} decisions. AUROC {self.auroc:.3f}, ECE {self.ece:.3f}, "
            f"Brier {self.brier:.3f} (reliability {self.parts.reliability:.4f}, "
            f"resolution {self.parts.resolution:.4f}). Calibration slope "
            f"{self.slope:.2f}, so {direction}. {self.parts.reads_as()}."
        )


def prediction_histogram(p: Floats, *, bins: int = 20) -> tuple[int, ...]:
    """Counts of predictions in uniform bins over [0, 1]."""
    counts, _ = np.histogram(np.asarray(p, dtype=np.float64), bins=bins, range=(0.0, 1.0))
    return tuple(int(c) for c in counts)


def report(p: Floats, y: Labels, *, bins: int = 10) -> CalibrationReport:
    probs, labels = _as_arrays(p, y)
    slope, intercept = calibration_slope_intercept(p, y)
    return CalibrationReport(
        n=int(probs.size),
        ece=ece(p, y, bins=bins),
        adaptive_ece=adaptive_ece(p, y),
        mce=mce(p, y, bins=bins),
        brier=brier(p, y),
        parts=brier_decomposition(p, y, bins=bins),
        log_loss=log_loss(p, y),
        auroc=auroc(p, y),
        slope=slope,
        intercept=intercept,
        base_rate=float(labels.mean()),
        curve=calibration_curve(p, y, bins=bins),
        distribution=prediction_histogram(p),
    )


__all__ = [
    "Bin",
    "BrierParts",
    "CalibrationCurve",
    "CalibrationReport",
    "Calibrator",
    "adaptive_ece",
    "auroc",
    "brier",
    "brier_decomposition",
    "calibration_curve",
    "calibration_slope_intercept",
    "ece",
    "fit_isotonic",
    "fit_platt",
    "log_loss",
    "mce",
    "noul_confidence",
    "report",
    "wilson_interval",
]

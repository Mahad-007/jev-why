"""Turning a matrix of measured values into per-span attributions.

Pure: numpy in, numpy out. Every estimator here is tested against a synthetic
set function whose Shapley values are known in closed form, so correctness is
established with no API key and no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from jev_why.coalitions import EMPTY, CoalitionPlan, kernel_weight

EPS = 1e-12


@dataclass(frozen=True)
class EstimateResult:
    phi: NDArray[np.float64]
    necessity: NDArray[np.float64]
    sufficiency: NDArray[np.float64]
    efficiency_gap: float
    stderr: NDArray[np.float64] | None = None
    dropped: int = 0
    """Coalitions whose call failed. KernelSHAP refits on what it has; a
    partially failed run should be visibly degraded, never quietly wrong."""


class Estimator(Protocol):
    @property
    def name(self) -> str: ...

    def estimate(self, plan: CoalitionPlan, values: NDArray[np.float64]) -> EstimateResult: ...


def _endpoints(plan: CoalitionPlan, values: NDArray[np.float64]) -> tuple[float, float]:
    full = frozenset(range(plan.n_spans))
    return float(values[plan.index_of(full)]), float(values[plan.index_of(EMPTY)])


def _efficiency_gap(phi: NDArray[np.float64], v_full: float, v_empty: float) -> float:
    """How far the attributions are from summing to what they should.

    Computed over the finite entries only. A single failed call would otherwise
    turn the whole diagnostic into a NaN, and a NaN printed as a percentage is
    a number that looks measured and is not.
    """
    total = v_full - v_empty
    finite = phi[np.isfinite(phi)]
    if not finite.size or not np.isfinite(total):
        return float("nan")
    return float(abs(finite.sum() - total) / max(abs(total), EPS))


@dataclass(frozen=True)
class OcclusionEstimator:
    """Necessity and sufficiency, averaged.

    Necessity is what the span is worth in last position; sufficiency is what it
    is worth in first. Their mean approximates the Shapley value, and reporting
    them separately is informative in its own right: a span with high
    sufficiency and near-zero necessity is redundant evidence, which is a fact
    about the document worth surfacing.
    """

    name: str = "occlusion"

    def estimate(self, plan: CoalitionPlan, values: NDArray[np.float64]) -> EstimateResult:
        n = plan.n_spans
        v_full, v_empty = _endpoints(plan, values)
        full = frozenset(range(n))

        necessity = np.zeros(n, dtype=np.float64)
        sufficiency = np.zeros(n, dtype=np.float64)
        have_loo = have_loi = False

        for i in range(n):
            loo = full - {i}
            if loo in plan.coalitions:
                necessity[i] = v_full - float(values[plan.index_of(loo)])
                have_loo = True
            single = frozenset({i})
            if single in plan.coalitions:
                sufficiency[i] = float(values[plan.index_of(single)]) - v_empty
                have_loi = True

        if have_loo and have_loi:
            phi = (necessity + sufficiency) / 2.0
        elif have_loo:
            phi = necessity.copy()
        else:
            phi = sufficiency.copy()

        return EstimateResult(phi, necessity, sufficiency, _efficiency_gap(phi, v_full, v_empty))


@dataclass(frozen=True)
class KernelShapEstimator:
    """Weighted least squares over sampled coalitions, with Shapley's
    efficiency axiom imposed as a hard constraint rather than hoped for.

    The constraint is applied by elimination: fix phi[n-1] to whatever makes the
    values sum to v(All) - v(0), regress on the remaining n-1 free parameters,
    then recover it. That guarantees efficiency exactly instead of penalising
    deviations from it.
    """

    name: str = "shapley"
    ridge: float = 1e-8

    def estimate(self, plan: CoalitionPlan, values: NDArray[np.float64]) -> EstimateResult:
        n = plan.n_spans
        v_full, v_empty = _endpoints(plan, values)
        total = v_full - v_empty

        rows: list[NDArray[np.float64]] = []
        targets: list[float] = []
        weights: list[float] = []
        dropped = 0
        for idx, coalition in enumerate(plan.coalitions):
            size = len(coalition)
            if size == 0 or size == n:
                continue
            value = values[idx]
            if not np.isfinite(value):
                dropped += 1
                continue
            indicator = np.zeros(n, dtype=np.float64)
            indicator[list(coalition)] = 1.0
            rows.append(indicator)
            targets.append(float(value) - v_empty)
            weights.append(kernel_weight(n, size))

        if len(rows) < n:
            # Not enough information to identify n parameters; fall back rather
            # than return a confidently wrong fit.
            fallback = OcclusionEstimator().estimate(plan, values)
            return EstimateResult(
                fallback.phi,
                fallback.necessity,
                fallback.sufficiency,
                fallback.efficiency_gap,
                None,
                dropped,
            )

        z = np.vstack(rows)
        y = np.asarray(targets, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()

        # Eliminate the last coefficient using sum(phi) == total.
        x = z[:, : n - 1] - z[:, n - 1 : n]
        y_adj = y - z[:, n - 1] * total

        wx = x * w[:, None]
        gram = x.T @ wx
        gram += np.eye(n - 1) * (self.ridge * max(np.trace(gram), EPS) / max(n - 1, 1))
        rhs = wx.T @ y_adj

        try:
            free = np.linalg.solve(gram, rhs)
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            free, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
            gram_inv = np.linalg.pinv(gram)

        phi = np.empty(n, dtype=np.float64)
        phi[: n - 1] = free
        phi[n - 1] = total - free.sum()

        residual = y_adj - x @ free
        dof = max(len(y_adj) - (n - 1), 1)
        sigma2 = float((w * residual**2).sum() * len(y_adj) / dof)
        stderr_free = np.sqrt(np.maximum(np.diag(gram_inv) * sigma2, 0.0))
        stderr = np.empty(n, dtype=np.float64)
        stderr[: n - 1] = stderr_free
        stderr[n - 1] = float(np.sqrt(np.sum(stderr_free**2)))

        occlusion = OcclusionEstimator().estimate(plan, values)
        return EstimateResult(
            phi,
            occlusion.necessity,
            occlusion.sufficiency,
            _efficiency_gap(phi, v_full, v_empty),
            stderr,
            dropped,
        )


@dataclass(frozen=True)
class PermutationShapEstimator:
    """ApproShapley. The reference implementation, not the recommended mode.

    Correct by construction and slow: it exists so KernelSHAP has something
    independent to be checked against.
    """

    name: str = "permutation"

    def estimate(self, plan: CoalitionPlan, values: NDArray[np.float64]) -> EstimateResult:
        n = plan.n_spans
        permutations = plan.meta.get("permutations", ())
        if not permutations:
            raise ValueError("permutation estimator needs a permutation plan")

        lookup = {c: float(values[i]) for i, c in enumerate(plan.coalitions)}
        phi = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)

        for perm in permutations:
            running: set[int] = set()
            previous = lookup[EMPTY]
            for item in perm:
                running.add(item)
                current = lookup[frozenset(running)]
                phi[item] += current - previous
                counts[item] += 1
                previous = current

        phi = np.divide(phi, np.maximum(counts, 1.0))
        v_full, v_empty = _endpoints(plan, values)
        occlusion = OcclusionEstimator().estimate(plan, values)
        return EstimateResult(
            phi, occlusion.necessity, occlusion.sufficiency, _efficiency_gap(phi, v_full, v_empty)
        )


def resolve_estimator(name: str) -> Estimator:
    match name:
        case "occlusion":
            return OcclusionEstimator()
        case "shapley" | "kernelshap":
            return KernelShapEstimator()
        case "permutation":
            return PermutationShapEstimator()
        case _:
            raise ValueError(f"unknown estimator {name!r}")


__all__ = [
    "EstimateResult",
    "Estimator",
    "KernelShapEstimator",
    "OcclusionEstimator",
    "PermutationShapEstimator",
    "resolve_estimator",
]

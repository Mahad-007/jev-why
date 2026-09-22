"""Planning which masked variants to evaluate.

Pure: given a span count and a budget, produce the exact set of coalitions to
call. No network, no randomness beyond a seeded Random.

Separating the plan from its execution is what makes budget pre-flight exact,
runs resumable, and the whole estimator layer testable against a synthetic
oracle with no API key.

The nesting property: `kernelshap_plan` emits the occlusion coalitions first, in
the same order. So upgrading a cached occlusion run to Shapley pays only for the
difference. Accuracy becomes an incremental purchase rather than a fresh one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import comb
from random import Random
from typing import Any

from jev_why.types import Coalition


@dataclass(frozen=True)
class CoalitionPlan:
    n_spans: int
    coalitions: tuple[Coalition, ...]
    estimator: str
    weights: tuple[float, ...] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_calls(self) -> int:
        return len(self.coalitions)

    def index_of(self, coalition: Coalition) -> int:
        return self.coalitions.index(coalition)


def _full(n: int) -> Coalition:
    return frozenset(range(n))


EMPTY: Coalition = frozenset()


def kernel_weight(n: int, size: int) -> float:
    """The Shapley kernel. Largest at |S| = 1 and |S| = n-1, which is exactly
    where the leave-one-in and leave-one-out coalitions live -- the cheap
    estimator samples the most informative strata by construction."""
    if size <= 0 or size >= n:
        return float("inf")
    return (n - 1) / (comb(n, size) * size * (n - size))


def occlusion_plan(n: int, *, mode: str = "both") -> CoalitionPlan:
    """Leave-one-out, leave-one-in, or both.

    `both` costs 2n+2 calls and is the default, because leave-one-out alone has
    a failure mode that matters: if two spans each independently justify the
    answer, removing either changes nothing, both score zero, and the
    explanation reports that nothing in the document mattered. Leave-one-in
    catches exactly that case.

    Averaging the two is not a heuristic. Shapley averages a span's marginal
    contribution over every position in a random permutation; leave-one-out is
    that contribution in last position and leave-one-in is it in first. Their
    mean is a two-point quadrature of the same integral.
    """
    coalitions: list[Coalition] = [_full(n), EMPTY]
    if mode in ("loo", "both"):
        coalitions += [_full(n) - {i} for i in range(n)]
    if mode in ("loi", "both"):
        coalitions += [frozenset({i}) for i in range(n)]
    return CoalitionPlan(n, _dedupe(coalitions), "occlusion", meta={"mode": mode})


def kernelshap_plan(n: int, *, budget_calls: int | None = None, seed: int = 0) -> CoalitionPlan:
    """Coalitions for KernelSHAP, richest strata first.

    1. The endpoints, then the full occlusion set (strata |S| = 1 and n-1),
       which carry the largest kernel weight and are enumerated exactly.
    2. Further paired strata (k, n-k) enumerated exhaustively while they fit.
    3. The residual budget spent on antithetic pairs: a sampled coalition and
       its complement together. Complement pairing is the largest variance
       reduction available here and it costs nothing.
    """
    budget = budget_calls if budget_calls is not None else 8 * n + 2
    budget = max(budget, 2 * n + 2)
    rng = Random(seed)

    coalitions: list[Coalition] = list(occlusion_plan(n, mode="both").coalitions)
    exhausted = {1, n - 1}

    k = 2
    while k <= n - k and len(coalitions) < budget:
        pair_cost = comb(n, k) + (comb(n, n - k) if n - k != k else 0)
        if len(coalitions) + pair_cost > budget:
            break
        for subset in _all_subsets(n, k):
            coalitions.append(subset)
        if n - k != k:
            for subset in _all_subsets(n, n - k):
                coalitions.append(subset)
        exhausted |= {k, n - k}
        k += 1

    sizes = [s for s in range(1, n) if s not in exhausted]
    if sizes:
        weights = [kernel_weight(n, s) for s in sizes]
        total = sum(weights)
        probs = [w / total for w in weights]
        seen = set(coalitions)
        guard = 0
        while len(coalitions) < budget and guard < budget * 50:
            guard += 1
            size = _weighted_pick(rng, sizes, probs)
            subset = frozenset(rng.sample(range(n), size))
            complement = _full(n) - subset
            for candidate in (subset, complement):
                if candidate not in seen and len(coalitions) < budget:
                    coalitions.append(candidate)
                    seen.add(candidate)

    final = _dedupe(coalitions)
    weights_out = tuple(kernel_weight(n, len(c)) for c in final)
    return CoalitionPlan(
        n,
        final,
        "shapley",
        weights_out,
        meta={"budget": budget, "seed": seed, "exhausted": sorted(exhausted)},
    )


def permutation_plan(n: int, *, m_permutations: int = 20, seed: int = 0) -> CoalitionPlan:
    """ApproShapley: walk random permutations from empty to full.

    Strictly more expensive than KernelSHAP for equal accuracy, so this is not
    the recommended accurate mode. It ships because it is trivially, obviously
    correct, and it is the reference the KernelSHAP solver is tested against on
    set functions whose Shapley values are known in closed form. Two
    independent estimators agreeing on a known game is the strongest
    correctness evidence available without a network.
    """
    rng = Random(seed)
    coalitions: list[Coalition] = [_full(n), EMPTY]
    permutations: list[tuple[int, ...]] = []
    for _ in range(m_permutations):
        order = list(range(n))
        rng.shuffle(order)
        for perm in (tuple(order), tuple(reversed(order))):
            permutations.append(perm)
            running: set[int] = set()
            for item in perm:
                running.add(item)
                coalitions.append(frozenset(running))
    return CoalitionPlan(
        n,
        _dedupe(coalitions),
        "permutation",
        meta={"permutations": tuple(permutations), "seed": seed},
    )


def _all_subsets(n: int, k: int) -> list[Coalition]:
    from itertools import combinations

    return [frozenset(c) for c in combinations(range(n), k)]


def _weighted_pick(rng: Random, items: list[int], probs: list[float]) -> int:
    r = rng.random()
    acc = 0.0
    for item, p in zip(items, probs, strict=True):
        acc += p
        if r <= acc:
            return item
    return items[-1]


def _dedupe(coalitions: list[Coalition]) -> tuple[Coalition, ...]:
    """Preserve first-seen order. Order is the contract that makes the
    occlusion plan a prefix of the Shapley plan."""
    seen: set[Coalition] = set()
    out: list[Coalition] = []
    for c in coalitions:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


__all__ = [
    "EMPTY",
    "CoalitionPlan",
    "kernel_weight",
    "kernelshap_plan",
    "occlusion_plan",
    "permutation_plan",
]

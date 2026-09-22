"""Does the explanation describe what the model actually did?

Pure: planning and scoring only. The calls are made by the orchestration layer.

An unfaithful explanation is worse than no explanation, because it is
convincing. The metrics here are the standard erasure-based ones, and the part
that matters most is the control: an attribution is only credible if it beats a
random ordering of the same spans at the same budget. Reporting
comprehensiveness alone says nothing, because deleting any 30% of a document
moves the answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

from jev_why.types import Coalition

DEFAULT_SCHEDULE: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.5)
"""Deletion budgets, as fractions of the span count.

The schedule stops at half the document on purpose. All the discriminating
power is at small k: once most of the spans are gone, a random ordering has
removed the real cause too, so every method converges and the high end only
dilutes the average. Extending the schedule upward makes a faithful
explanation look less distinguishable than it is."""
DEFAULT_RANDOM_TRIALS = 10


@dataclass(frozen=True)
class FaithfulnessPlan:
    """Which coalitions to evaluate, and how to read the results back."""

    n_spans: int
    coalitions: tuple[Coalition, ...]
    ks: tuple[int, ...]
    comprehensiveness: tuple[int, ...]
    sufficiency: tuple[int, ...]
    random_comprehensiveness: tuple[tuple[int, ...], ...]
    random_sufficiency: tuple[tuple[int, ...], ...]
    counter_evidence: int
    full: int
    empty: int

    @property
    def n_calls(self) -> int:
        return len(self.coalitions)


@dataclass(frozen=True)
class CurvePoint:
    k: int
    fraction: float
    comprehensiveness: float
    sufficiency: float
    random_comprehensiveness: float


@dataclass(frozen=True)
class FaithfulnessReport:
    comprehensiveness: float
    """Mean drop when the top-ranked spans are removed. Higher is better."""

    sufficiency: float
    """Mean drop when only the top-ranked spans are kept. Lower is better: a
    faithful subset reproduces the decision on its own."""

    random_comprehensiveness: float
    lift: float
    """Comprehensiveness above the random control. This is the honest headline;
    the raw figure on its own is not interpretable."""

    p_value: float
    counter_evidence_holds: bool
    """Removing the spans scored most negative should push the value UP. An
    explanation that gets the sign backwards is worse than useless in any
    setting where someone acts on it."""

    curve: tuple[CurvePoint, ...]
    random_trials: int
    cross_masked: bool = False

    @property
    def credible(self) -> bool:
        return self.lift > 0 and self.p_value <= 0.1 and self.counter_evidence_holds

    def verdict(self) -> str:
        if not self.counter_evidence_holds:
            return "not faithful: removing negatively-scored spans did not raise the value"
        if self.lift <= 0:
            return "not faithful: no better than deleting random spans"
        if self.p_value > 0.1:
            return f"inconclusive: lift {self.lift:+.3f}, p={self.p_value:.2f}"
        return f"faithful: {self.lift:+.3f} above random, p={self.p_value:.2f}"


def _ks(n: int, schedule: Sequence[float]) -> tuple[int, ...]:
    seen: list[int] = []
    for fraction in schedule:
        k = max(1, min(n - 1, round(fraction * n)))
        if k not in seen:
            seen.append(k)
    return tuple(seen)


def faithfulness_plan(
    ranking: Sequence[int],
    n_spans: int,
    *,
    schedule: Sequence[float] = DEFAULT_SCHEDULE,
    random_trials: int = DEFAULT_RANDOM_TRIALS,
    negative_ranking: Sequence[int] | None = None,
    seed: int = 0,
) -> FaithfulnessPlan:
    rng = Random(seed)
    full = frozenset(range(n_spans))
    ordered: list[Coalition] = []
    index: dict[Coalition, int] = {}

    def add(coalition: Coalition) -> int:
        if coalition not in index:
            index[coalition] = len(ordered)
            ordered.append(coalition)
        return index[coalition]

    full_idx = add(full)
    empty_idx = add(frozenset())

    ks = _ks(n_spans, schedule)
    comprehensiveness = tuple(add(full - set(ranking[:k])) for k in ks)
    sufficiency = tuple(add(frozenset(ranking[:k])) for k in ks)

    random_comp: list[tuple[int, ...]] = []
    random_suff: list[tuple[int, ...]] = []
    for _ in range(random_trials):
        shuffled = list(range(n_spans))
        rng.shuffle(shuffled)
        random_comp.append(tuple(add(full - set(shuffled[:k])) for k in ks))
        random_suff.append(tuple(add(frozenset(shuffled[:k])) for k in ks))

    negatives = list(negative_ranking or [])[: max(1, n_spans // 5)]
    counter = add(full - set(negatives)) if negatives else full_idx

    return FaithfulnessPlan(
        n_spans=n_spans,
        coalitions=tuple(ordered),
        ks=ks,
        comprehensiveness=comprehensiveness,
        sufficiency=sufficiency,
        random_comprehensiveness=tuple(random_comp),
        random_sufficiency=tuple(random_suff),
        counter_evidence=counter,
        full=full_idx,
        empty=empty_idx,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_faithfulness(
    plan: FaithfulnessPlan, values: Sequence[float], *, cross_masked: bool = False
) -> FaithfulnessReport:
    baseline = values[plan.full]

    comp = [baseline - values[i] for i in plan.comprehensiveness]
    suff = [baseline - values[i] for i in plan.sufficiency]
    random_comp_per_trial = [
        _mean([baseline - values[i] for i in trial]) for trial in plan.random_comprehensiveness
    ]

    aopc_comp = _mean(comp)
    aopc_random = _mean(random_comp_per_trial)

    # Permutation p-value. With R trials the floor is 1/(R+1), so a report that
    # quotes p must also quote how many trials produced it.
    at_least_as_good = sum(1 for r in random_comp_per_trial if r >= aopc_comp)
    p_value = (1 + at_least_as_good) / (1 + len(random_comp_per_trial))

    curve = tuple(
        CurvePoint(
            k=k,
            fraction=k / plan.n_spans,
            comprehensiveness=comp[i],
            sufficiency=suff[i],
            random_comprehensiveness=_mean(
                [baseline - values[trial[i]] for trial in plan.random_comprehensiveness]
            ),
        )
        for i, k in enumerate(plan.ks)
    )

    counter_value = values[plan.counter_evidence]
    counter_holds = plan.counter_evidence == plan.full or counter_value >= baseline

    return FaithfulnessReport(
        comprehensiveness=aopc_comp,
        sufficiency=_mean(suff),
        random_comprehensiveness=aopc_random,
        lift=aopc_comp - aopc_random,
        p_value=p_value,
        counter_evidence_holds=counter_holds,
        curve=curve,
        random_trials=len(random_comp_per_trial),
        cross_masked=cross_masked,
    )


__all__ = [
    "DEFAULT_RANDOM_TRIALS",
    "DEFAULT_SCHEDULE",
    "CurvePoint",
    "FaithfulnessPlan",
    "FaithfulnessReport",
    "faithfulness_plan",
    "score_faithfulness",
]

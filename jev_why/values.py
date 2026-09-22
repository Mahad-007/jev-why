"""Turning an Answer into numbers an estimator can work on.

Pure. This module decides what "the model changed its mind by this much" means,
which is the question every attribution method quietly assumes someone else has
answered.

Two quantities come out of every question:

  primary   a signed scalar, defined per coalition, that Shapley and occlusion
            estimate over. Signed because direction is the most useful single
            bit: "this sentence pushed it toward billing" is actionable,
            "this sentence was influential" is not.

  magnitude an unsigned pairwise distance between two answer distributions,
            used to rank spans by total influence regardless of direction. It
            catches a span that reshuffles probability mass without moving the
            top answer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jev_why.types import Answer, Link, QuestionType

EPS = 1e-4
SATURATION_HIGH = 0.95
SATURATION_LOW = 0.05


@dataclass(frozen=True, slots=True)
class ValueSpec:
    """How to reduce one question's answer to one number."""

    question: str
    qtype: QuestionType
    link: Link = "prob"
    target: str | None = None
    """For a choice, the class being explained. Defaults to the baseline
    argmax, but any class can be explained -- "why not technical?" is a
    legitimate question."""

    contrast: tuple[str, str] | None = None
    """For a choice, explain log(p[a]/p[b]) instead. The cleanest contrastive
    statement and the one that maps to a pairwise decision."""

    n_levels: int = 0


def logit(p: float) -> float:
    q = min(max(p, EPS), 1.0 - EPS)
    return math.log(q / (1.0 - q))


def is_saturated(p: float) -> bool:
    """True when a probability sits so close to 0 or 1 that differences in
    probability space compress into noise.

    At p=0.995 no span can move the probability by more than 0.005, so the
    ranking collapses. Log-odds is additive in evidence and keeps resolving.
    """
    return p > SATURATION_HIGH or p < SATURATION_LOW


def choose_link(baseline_p: float, requested: str) -> Link:
    if requested in ("prob", "logit"):
        return requested  # type: ignore[return-value]
    return "logit" if is_saturated(baseline_p) else "prob"


def expected_level(probabilities: Mapping[int, float]) -> float:
    """E = sum(i * p_i) over ordered levels.

    Recomputed from the distribution rather than taken from the returned scalar
    because E is *linear* in the answer distribution, so Shapley values over E
    decompose exactly into Shapley values over each per-level probability.
    Nothing else on the menu has that property.
    """
    total = sum(probabilities.values())
    if total <= 0:
        return 0.0
    return sum(level * p for level, p in probabilities.items()) / total


def primary_value(answer: Answer, spec: ValueSpec) -> float:
    """The scalar the estimator attributes over."""
    match answer.qtype:
        case "noul":
            p = answer.noul if answer.noul is not None else 0.0
            return logit(p) if spec.link == "logit" else p

        case "choice":
            probs = answer.probabilities or {}
            if spec.contrast is not None:
                a, b = spec.contrast
                return logit(float(probs.get(a, 0.0))) - logit(float(probs.get(b, 0.0)))
            target = spec.target or answer.choice
            p = float(probs.get(target, 0.0)) if target else 0.0
            return logit(p) if spec.link == "logit" else p

        case "score":
            if answer.probabilities:
                levels = {int(k): float(v) for k, v in answer.probabilities.items()}
                return expected_level(levels)
            return float(answer.score or 0.0)

    raise ValueError(f"unknown question type {answer.qtype!r}")


def distribution(answer: Answer) -> tuple[float, ...]:
    """The answer as a probability vector, for pairwise distances."""
    match answer.qtype:
        case "noul":
            p = float(answer.noul or 0.0)
            return (1.0 - p, p)
        case "choice":
            probs = answer.probabilities or {}
            return tuple(float(probs[k]) for k in sorted(probs))
        case "score":
            probs = answer.probabilities or {}
            return tuple(float(probs[k]) for k in sorted(probs, key=lambda x: int(x)))
    raise ValueError(f"unknown question type {answer.qtype!r}")


def _normalise(v: Sequence[float]) -> tuple[float, ...]:
    total = sum(v)
    if total <= 0:
        return tuple(1.0 / len(v) for _ in v) if v else ()
    return tuple(x / total for x in v)


def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits, so the result sits in [0, 1].

    The right magnitude for a choice, where classes are unordered and any
    reshuffling of mass counts equally.
    """
    if len(p) != len(q) or not p:
        return 0.0
    pn, qn = _normalise(p), _normalise(q)

    def kl(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(a, b, strict=True) if x > 0 and y > 0)

    m = tuple((x + y) / 2 for x, y in zip(pn, qn, strict=True))
    return max(0.0, 0.5 * kl(pn, m) + 0.5 * kl(qn, m))


def wasserstein1(p: Sequence[float], q: Sequence[float]) -> float:
    """Earth mover's distance over an ordinal support, normalised to [0, 1].

    The right magnitude for a score, and the reason JSD is wrong there: score
    levels are ordered, so a shift from "calm" to "annoyed" must count for less
    than a shift from "calm" to "angry". JSD is blind to that and would rank
    a one-level nudge equal to a three-level swing.
    """
    if len(p) != len(q) or len(p) < 2:
        return 0.0
    pn, qn = _normalise(p), _normalise(q)
    cumulative = 0.0
    total = 0.0
    for x, y in zip(pn[:-1], qn[:-1], strict=True):
        cumulative += x - y
        total += abs(cumulative)
    return total / (len(p) - 1)


def magnitude(baseline: Answer, masked: Answer, qtype: QuestionType) -> float:
    """Direction-free influence of whatever changed between two answers."""
    match qtype:
        case "noul":
            return abs(float(baseline.noul or 0.0) - float(masked.noul or 0.0))
        case "choice":
            return jensen_shannon(distribution(baseline), distribution(masked))
        case "score":
            return wasserstein1(distribution(baseline), distribution(masked))
    raise ValueError(f"unknown question type {qtype!r}")


__all__ = [
    "EPS",
    "ValueSpec",
    "choose_link",
    "distribution",
    "expected_level",
    "is_saturated",
    "jensen_shannon",
    "logit",
    "magnitude",
    "primary_value",
    "wasserstein1",
]

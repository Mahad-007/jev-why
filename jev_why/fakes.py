"""Test doubles, shipped as part of the package rather than hidden in a test
directory, because they are useful to anyone extending an estimator.

SyntheticOracle is the important one. It is a set function whose Shapley values
are known exactly, which is what lets the whole estimator layer be verified
without spending a call or holding an API key.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from random import Random

import numpy as np
from numpy.typing import NDArray

from jev_why.coalitions import CoalitionPlan
from jev_why.types import Answer, JevResponse, Usage


@dataclass
class SyntheticOracle:
    """A set function built from additive terms and unanimity games.

    v(S) = base + sum(additive[i] for i in S) + sum(w for T, w in interactions if T subset of S)

    The Shapley value of a unanimity game on T with weight w gives w/|T| to each
    member of T and nothing to anyone else. Combined with linearity, that makes
    the exact Shapley vector of this whole construction a closed form -- which
    is the point. An estimator that cannot recover it is broken, and one that
    can has been checked against ground truth rather than against itself.
    """

    n: int
    additive: Mapping[int, float] = field(default_factory=dict)
    interactions: Mapping[frozenset[int], float] = field(default_factory=dict)
    base: float = 0.0
    noise: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)
        self.calls = 0

    def value(self, coalition: AbstractSet[int]) -> float:
        self.calls += 1
        total = self.base
        for i in coalition:
            total += self.additive.get(i, 0.0)
        for members, weight in self.interactions.items():
            if members <= set(coalition):
                total += weight
        if self.noise:
            total += self._rng.gauss(0.0, self.noise)
        return total

    def true_shapley(self) -> NDArray[np.float64]:
        phi = np.zeros(self.n, dtype=np.float64)
        for i, weight in self.additive.items():
            phi[i] += weight
        for members, weight in self.interactions.items():
            share = weight / len(members)
            for i in members:
                phi[i] += share
        return phi

    def evaluate(self, plan: CoalitionPlan) -> NDArray[np.float64]:
        """Value every coalition in a plan, in plan order."""
        return np.array([self.value(c) for c in plan.coalitions], dtype=np.float64)


@dataclass
class FakeClock:
    """A clock that only moves when told to, so rate limiting and backoff can be
    tested to the millisecond without sleeping."""

    now: float = 0.0

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeJevClient:
    """A client that answers from a scripted function instead of the network.

    `responder` receives the rendered state and returns a mapping of question
    name to Answer, so a test can make the answer depend on the state in
    whatever way the test needs.
    """

    responder: object
    model: str = "jev-fake-1.0"
    fail_on: frozenset[int] = frozenset()
    """Call ordinals that should raise, for exercising partial-failure paths."""

    def __post_init__(self) -> None:
        self.calls = 0

    async def system_one(
        self, state: object, questions: Mapping[str, object], *, model: str | None = None
    ) -> JevResponse:
        ordinal = self.calls
        self.calls += 1
        if ordinal in self.fail_on:
            raise RuntimeError(f"scripted failure on call {ordinal}")
        answers = self.responder(state, questions)  # type: ignore[operator]
        tokens = len(str(state)) // 4
        return JevResponse(self.model, answers, Usage(tokens, 0))


def noul_responder(
    weights: Mapping[str, float], *, question: str = "q", base: float = 0.0
) -> object:
    """A responder whose probability rises with each keyword present in the
    state. Gives a text-level oracle with a known answer for wiring tests."""

    def respond(state: object, questions: Mapping[str, object]) -> dict[str, Answer]:
        text = str(state)
        total = base + sum(w for token, w in weights.items() if token in text)
        p = 1.0 / (1.0 + np.exp(-total))
        return {question: Answer(qtype="noul", noul=float(p))}

    return respond


__all__ = ["FakeClock", "FakeJevClient", "SyntheticOracle", "noul_responder"]

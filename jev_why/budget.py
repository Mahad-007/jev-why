"""Estimating and enforcing what a run will cost.

Pure. No clock, no network. The guard is handed usage figures by the executor
rather than reading them itself, which is what makes overspend behaviour
testable without spending anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

PRICE_PER_MTOK_USD = 0.042
"""Input pricing as published in September 2026. Output tokens are free, which
is the whole reason this technique is affordable."""

CHARS_PER_TOKEN = 3.6
JSON_OVERHEAD = 1.08
SAFETY_FACTOR = 1.15
"""Estimation without the vendor's tokenizer is a guess, so it guesses high.
The guard self-calibrates from the input_tokens every response reports, and
after one real run the estimate is close."""


class BudgetExceeded(RuntimeError):
    """A run would cost, or has cost, more than the ceiling allows."""

    def __init__(self, message: str, *, spent_usd: float, ceiling_usd: float) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.ceiling_usd = ceiling_usd


def estimate_state_tokens(state: Any) -> int:
    if isinstance(state, str):
        return max(1, round(len(state) / CHARS_PER_TOKEN))
    text = repr(state)
    return max(1, round(len(text) / CHARS_PER_TOKEN * JSON_OVERHEAD))


def estimate_question_tokens(questions: Mapping[str, Any]) -> int:
    return max(1, round(len(repr(questions)) / CHARS_PER_TOKEN * JSON_OVERHEAD))


def estimate_cost(tokens: int, *, price_per_mtok: float = PRICE_PER_MTOK_USD) -> float:
    return tokens * price_per_mtok / 1_000_000


@dataclass(frozen=True)
class CostEstimate:
    calls: int
    state_tokens: int
    question_tokens: int
    mean_kept_fraction: float
    total_tokens: int
    usd: float

    def explain(self) -> str:
        return (
            f"{self.calls} calls, about {self.total_tokens:,} input tokens, roughly ${self.usd:.4f}"
        )


def estimate_plan_cost(
    *,
    calls: int,
    state_tokens: int,
    question_tokens: int,
    mean_kept_fraction: float = 1.0,
    calibration: float = 1.0,
) -> CostEstimate:
    """What a coalition plan will cost.

    Note where the question tokens sit: they are paid on *every* call, not once.
    Sharing one coalition sample across a panel of K questions saves
    (K*S + Q) / (S + Q), not an unbounded factor -- and for many small questions
    against a short state the saving inverts. The library reports the real
    number rather than assuming the flattering one.
    """
    per_call = state_tokens * mean_kept_fraction + question_tokens
    total = round(calls * per_call * SAFETY_FACTOR * calibration)
    return CostEstimate(
        calls, state_tokens, question_tokens, mean_kept_fraction, total, estimate_cost(total)
    )


def amortisation_factor(state_tokens: int, question_tokens: int, n_questions: int) -> float:
    """How much sharing one coalition sample across the panel actually saves.

    Returns the ratio of running each question separately to running them
    together. Above 1 is a saving; at or below 1 the panel is dominated by
    question text and there is nothing to gain.
    """
    if n_questions <= 0 or state_tokens + question_tokens <= 0:
        return 1.0
    separate = n_questions * state_tokens + question_tokens
    shared = state_tokens + question_tokens
    return separate / shared


@dataclass(frozen=True)
class Budget:
    max_usd: float | None = 0.25
    max_calls: int | None = None
    on_exceed: Literal["raise", "warn", "truncate"] = "raise"

    @classmethod
    def of(cls, value: Budget | float | None) -> Budget:
        if value is None:
            return cls(max_usd=None)
        if isinstance(value, Budget):
            return value
        return cls(max_usd=float(value))


@dataclass
class BudgetGuard:
    """Tracks estimated and actual spend.

    Pre-flight is only an estimate, so enforcement also happens at runtime: the
    executor reports each response's real token count and the guard trips as
    soon as the ceiling is crossed. A trip returns what has already been paid
    for rather than discarding it -- abandoning 900 successful calls because of
    a stop at call 901 would be its own kind of waste.
    """

    budget: Budget = field(default_factory=Budget)
    calibration: float = 1.0
    spent_tokens: int = 0
    calls: int = 0
    _estimated_tokens: int = 0

    def preflight(
        self,
        *,
        calls: int,
        state_tokens: int,
        question_tokens: int,
        mean_kept_fraction: float = 1.0,
    ) -> CostEstimate:
        estimate = estimate_plan_cost(
            calls=calls,
            state_tokens=state_tokens,
            question_tokens=question_tokens,
            mean_kept_fraction=mean_kept_fraction,
            calibration=self.calibration,
        )
        if self.budget.max_calls is not None and calls > self.budget.max_calls:
            raise BudgetExceeded(
                f"plan needs {calls} calls, ceiling is {self.budget.max_calls}",
                spent_usd=0.0,
                ceiling_usd=self.budget.max_usd or 0.0,
            )
        if (
            self.budget.max_usd is not None
            and estimate.usd > self.budget.max_usd
            and self.budget.on_exceed == "raise"
        ):
            raise BudgetExceeded(
                f"plan would cost about ${estimate.usd:.4f}, ceiling is "
                f"${self.budget.max_usd:.4f}. Raise budget=, reduce max_spans, "
                f"or use a coarser chunker.",
                spent_usd=estimate.usd,
                ceiling_usd=self.budget.max_usd,
            )
        self._estimated_tokens = estimate.total_tokens
        return estimate

    def affordable_calls(
        self, *, state_tokens: int, question_tokens: int, mean_kept_fraction: float = 1.0
    ) -> int:
        """How many calls fit under the ceiling. Used by on_exceed='truncate' to
        shrink the plan rather than refuse it."""
        if self.budget.max_usd is None:
            return 1_000_000
        per_call = (
            (state_tokens * mean_kept_fraction + question_tokens) * SAFETY_FACTOR * self.calibration
        )
        cost_per_call = estimate_cost(max(1, round(per_call)))
        return max(0, int(self.budget.max_usd // cost_per_call)) if cost_per_call else 0

    def record(self, input_tokens: int) -> None:
        self.spent_tokens += input_tokens
        self.calls += 1

    def recalibrate(self) -> None:
        """Correct the estimator from observed usage, so the next pre-flight in
        this process is grounded rather than guessed."""
        if self.calls and self._estimated_tokens:
            observed_per_call = self.spent_tokens / self.calls
            estimated_per_call = self._estimated_tokens / max(self.calls, 1)
            if estimated_per_call > 0:
                self.calibration = max(0.25, min(4.0, observed_per_call / estimated_per_call))

    @property
    def spent_usd(self) -> float:
        return estimate_cost(self.spent_tokens)

    def check(self) -> None:
        if self.budget.max_usd is None or self.budget.on_exceed != "raise":
            return
        if self.spent_usd > self.budget.max_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} of a ${self.budget.max_usd:.4f} ceiling "
                f"after {self.calls} calls",
                spent_usd=self.spent_usd,
                ceiling_usd=self.budget.max_usd,
            )


__all__ = [
    "PRICE_PER_MTOK_USD",
    "Budget",
    "BudgetExceeded",
    "BudgetGuard",
    "CostEstimate",
    "amortisation_factor",
    "estimate_cost",
    "estimate_plan_cost",
    "estimate_question_tokens",
    "estimate_state_tokens",
]

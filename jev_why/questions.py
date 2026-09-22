"""Question specifications and their validation.

Pure. These mirror the three Jev primitives and enforce the documented API
limits locally, so a malformed panel fails before it costs a call rather than
after N of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

from jev_why.types import QuestionType

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
CONTEXT_LIMIT_TOKENS = 64_000


class QuestionError(ValueError):
    """A question spec violates a documented API limit."""


@dataclass(frozen=True, slots=True)
class Noul:
    """A yes/no question, answered as a probability."""

    instructions: str
    criteria: Mapping[str, str] | None = None

    qtype: ClassVar[QuestionType] = "noul"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            payload["criteria"] = dict(self.criteria)
        return payload

    def validate(self, name: str) -> None:
        _require_instructions(name, self.instructions)
        if self.criteria is not None and set(self.criteria) - {"true", "false"}:
            raise QuestionError(
                f"question {name!r}: noul criteria may only contain 'true' and 'false', "
                f"got {sorted(self.criteria)}"
            )


@dataclass(frozen=True, slots=True)
class Choice:
    """Pick one option from a fixed set."""

    instructions: str
    criteria: Mapping[str, str | None]

    qtype: ClassVar[QuestionType] = "choice"

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }

    def validate(self, name: str) -> None:
        _require_instructions(name, self.instructions)
        if not self.criteria:
            raise QuestionError(f"question {name!r}: choice needs at least one option")
        if len(self.criteria) < 2:
            raise QuestionError(
                f"question {name!r}: choice needs at least 2 options to be a decision, "
                f"got {len(self.criteria)}"
            )
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise QuestionError(
                f"question {name!r}: choice allows at most {MAX_CHOICE_OPTIONS} options, "
                f"got {len(self.criteria)}"
            )


@dataclass(frozen=True, slots=True)
class Score:
    """Rate against ordered levels. The levels are ordinal, and jev-why relies on
    that ordering: distance between levels is meaningful."""

    instructions: str
    criteria: Sequence[str] = ()

    qtype: ClassVar[QuestionType] = "score"

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }

    def validate(self, name: str) -> None:
        _require_instructions(name, self.instructions)
        n = len(self.criteria)
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise QuestionError(
                f"question {name!r}: score needs between {MIN_SCORE_LEVELS} and "
                f"{MAX_SCORE_LEVELS} levels, got {n}"
            )


QuestionSpec = Noul | Choice | Score


def _require_instructions(name: str, instructions: str) -> None:
    if not instructions or not instructions.strip():
        raise QuestionError(f"question {name!r}: instructions must not be empty")


def validate_panel(questions: Mapping[str, QuestionSpec]) -> None:
    """Check a whole panel before spending anything on it."""
    if not questions:
        raise QuestionError("at least one question is required")
    for name, spec in questions.items():
        if not name or not name.strip():
            raise QuestionError("question names must not be empty")
        spec.validate(name)


def panel_payload(questions: Mapping[str, QuestionSpec]) -> dict[str, Any]:
    """Serialise a panel in sorted-key order.

    Sorting is what lets the response cache hit across runs that declared the
    same questions in a different order. That assumes question order does not
    affect answers, which is an assumption about the API rather than a fact
    about it -- `jev-why doctor` measures it, and if order turns out to matter
    the cache key has to include order and this sort has to go.
    """
    return {name: questions[name].to_payload() for name in sorted(questions)}


__all__ = [
    "CONTEXT_LIMIT_TOKENS", "Choice", "MAX_CHOICE_OPTIONS", "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS", "Noul", "QuestionError", "QuestionSpec", "Score",
    "panel_payload", "validate_panel",
]

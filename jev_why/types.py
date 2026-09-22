"""Core value types.

Pure: no network, no filesystem, no asyncio. Everything here is a frozen
dataclass or a plain alias, so an Explanation can be archived to JSON and read
back years later as an audit artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# A Jev `state` is whatever you are asking about.
State = str | dict[str, Any] | list[Any]

# The indices of the spans a masked variant KEEPS. Using "keep" rather than
# "mask" throughout avoids a class of off-by-one-inversion bugs: a coalition is
# a subset in the set-function sense, and set functions are defined on what is
# present.
Coalition = frozenset[int]

QuestionType = Literal["noul", "choice", "score"]
Link = Literal["prob", "logit"]


class SpanKind(str, Enum):
    SENTENCE = "sentence"
    LINE = "line"
    PARAGRAPH = "paragraph"
    JSON_LEAF = "json_leaf"
    HUNK = "hunk"
    GROUP = "group"


class MaskMode(str, Enum):
    """How a span is removed when it is not in the coalition.

    REDACT replaces the span with a neutral placeholder, keeping offsets, schema
    shape and array indices intact. DELETE removes it outright.

    REDACT is the default for text and JSON. DELETE is correct for diffs, where a
    subset of hunks is itself a valid diff and redacting a hunk body would
    produce a malformed patch. That is a per-medium judgement, not an
    inconsistency: the goal in both cases is to stay on the manifold of inputs
    the model could plausibly see.
    """

    REDACT = "redact"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class Span:
    """One unit of the state that can be independently masked."""

    index: int
    label: str
    text: str
    kind: SpanKind
    start: int | None = None
    end: int | None = None
    tokens: int = 0

    def preview(self, width: int = 60) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 1] + "…"


@dataclass(frozen=True, slots=True)
class Attribution:
    """What one span did to one question."""

    span: Span
    phi: float
    """Primary signed effect. Positive means the span pushed the decision toward
    the baseline answer."""

    necessity: float
    """v(All) - v(All without this span). High when removing the span changes the
    answer -- but zero for a span that is merely redundant with another."""

    sufficiency: float
    """v(this span alone) - v(nothing). High when the span alone reproduces the
    answer. This is what catches redundant evidence that necessity misses."""

    magnitude: float
    """Class-agnostic total influence. Jensen-Shannon for choice, Wasserstein-1
    for score, |phi| for noul."""

    stderr: float | None = None
    significant: bool = True
    """False when |phi| falls below the measured noise floor, in which case this
    row is reporting sampling noise rather than evidence."""


@dataclass(frozen=True, slots=True)
class QuestionExplanation:
    """Attributions for a single question, plus the diagnostics needed to judge
    whether to trust them."""

    question: str
    qtype: QuestionType
    baseline: float
    """v(All): the value on the unmodified state."""

    empty: float
    """v(0): the value with every span redacted. Not an empty state -- an empty
    state may be rejected outright, and the null reference needs to be a real
    input."""

    link: Link
    target: str | None
    attributions: tuple[Attribution, ...]
    efficiency_gap: float
    """|sum(phi) - (baseline - empty)| normalised. Occlusion carries no
    efficiency guarantee; a large gap means interactions dominate and the
    cheap estimator is out of its depth."""

    noise_sigma: float
    estimator: str

    def top(
        self, k: int = 5, *, signed: bool = False, include_insignificant: bool = False
    ) -> tuple[Attribution, ...]:
        rows = self.attributions
        if not include_insignificant:
            rows = tuple(a for a in rows if a.significant)
        key = (lambda a: a.phi) if signed else (lambda a: abs(a.phi))
        return tuple(sorted(rows, key=key, reverse=True)[:k])

    def ranking(self) -> tuple[int, ...]:
        """Span indices ordered by descending |phi|. The input to every
        faithfulness metric."""
        return tuple(
            a.span.index for a in sorted(self.attributions, key=lambda a: abs(a.phi), reverse=True)
        )


@dataclass(frozen=True, slots=True)
class Spend:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class Explanation:
    """The result of explaining one state against a panel of questions.

    All questions share one coalition sample, so cross-question comparisons
    ("this sentence raises urgency and lowers frustration") are valid rather
    than an artifact of two different random draws.
    """

    spans: tuple[Span, ...]
    questions: Mapping[str, QuestionExplanation]
    spend: Spend
    model_version: str
    mask_mode: MaskMode
    seed: int
    artifact_score: float | None = None
    """How truncated or incoherent the masked variants looked to Jev itself.
    High means the attributions are partly measuring the mask rather than the
    missing content."""

    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __getitem__(self, question: str) -> QuestionExplanation:
        return self.questions[question]

    def __iter__(self) -> Any:
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Answer:
    """One question's answer, normalised across the three primitives."""

    qtype: QuestionType
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: Mapping[Any, float] | None = None
    """Keyed by option name for a choice, by level index for a score."""

    confidence: float | None = None
    """Absent for a noul: the API returns only the raw probability, so
    jev-why derives noul confidence rather than reading it."""

    legend: Mapping[Any, Any] | None = None


@dataclass(frozen=True, slots=True)
class JevResponse:
    model: str
    answers: Mapping[str, Answer]
    usage: Usage


@dataclass(frozen=True, slots=True)
class JevRequest:
    state: State
    questions: Mapping[str, Any]
    model: str
    coalition: Coalition | None = None
    """Which coalition produced this request, for mapping responses back to rows
    of the value matrix. None for probes that sit outside the plan."""

    bypass_cache: bool = False
    """The noise probe measures repeat variance, so it must not be served from a
    deterministic cache or it would measure zero by construction."""


__all__ = [
    "Answer",
    "Attribution",
    "Coalition",
    "Explanation",
    "JevRequest",
    "JevResponse",
    "Link",
    "MaskMode",
    "QuestionExplanation",
    "QuestionType",
    "Span",
    "SpanKind",
    "Spend",
    "State",
    "Usage",
]

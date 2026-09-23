"""Measure the assumptions this library rests on, against the live API.

I/O.

Three of jev-why's design decisions depend on how Jev actually behaves, and
none of them are documented. Rather than assert them in a README, measure them
for about a cent and print what came back:

  determinism        occlusion compares probabilities across calls. If repeated
                     identical calls disagree, every small attribution is noise
                     wearing a number.

  question order     the response cache sorts questions by name so two runs
                     that declared the same panel differently share entries.
                     That is only sound if order does not change answers.

  question count     the whole cost argument is that one masked state answers
                     every question at once. If answering twenty questions
                     together shifts any one of them relative to asking it
                     alone, the amortisation is unsound and panel attributions
                     are not comparable to production single-question calls.
                     This is the load-bearing one.

  mask artifact      how much a redacted state reads as damaged rather than
                     merely shorter.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from jev_why.attribution import ARTIFACT_INSTRUCTIONS, ARTIFACT_QUESTION
from jev_why.chunking import TEXT_PLACEHOLDER
from jev_why.client import JevClient
from jev_why.questions import Noul, QuestionSpec, ordered_payload, panel_payload
from jev_why.types import Answer
from jev_why.values import ValueSpec, primary_value

PROBE_STATE = (
    "A customer writes: I was charged twice for the same annual plan on the third "
    "of the month, I have emailed twice about it, and nobody has replied. I would "
    "like the duplicate refunded today. Separately, the mobile app crashes when I "
    "open the receipts tab on an older phone."
)


@dataclass
class Finding:
    name: str
    passed: bool
    detail: str
    value: float | None = None

    def line(self) -> str:
        mark = "ok  " if self.passed else "WARN"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class DoctorReport:
    findings: list[Finding] = field(default_factory=list)
    calls: int = 0
    cost_usd: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(f.passed for f in self.findings)

    def render(self) -> str:
        lines = [f.line() for f in self.findings]
        lines.append("")
        lines.append(f"{self.calls} calls, about ${self.cost_usd:.4f}")
        if not self.all_passed:
            lines.append(
                "A WARN does not mean the library is broken. It means an assumption "
                "it makes does not hold on your account, and the README should say so "
                "rather than the default being trusted."
            )
        return "\n".join(lines)


def _panel() -> dict[str, QuestionSpec]:
    return {
        "is_urgent": Noul(instructions="Is the customer asking for something urgent?"),
        "is_billing": Noul(instructions="Is this about billing rather than a technical fault?"),
        "is_repeat": Noul(instructions="Has the customer contacted support before about this?"),
    }


def _filler(n: int) -> dict[str, QuestionSpec]:
    return {
        f"filler_{i:02d}": Noul(instructions=f"Does the text mention topic number {i}?")
        for i in range(n)
    }


async def _ask(
    client: JevClient,
    state: Any,
    questions: Mapping[str, QuestionSpec],
    model: str,
    *,
    keep_order: bool = False,
) -> tuple[Mapping[str, Answer], int]:
    # The normal path sorts questions by name so the cache hits across
    # declaration orders. keep_order bypasses that, because sorting would
    # erase the very thing the order-sensitivity probe measures.
    payload = ordered_payload(questions) if keep_order else panel_payload(questions)
    response = await client.system_one(state, payload, model=model)
    return response.answers, response.usage.input_tokens


def _noul(answers: Mapping[str, Answer], name: str) -> float:
    spec = ValueSpec(name, "noul")
    return primary_value(answers[name], spec)


async def run_doctor(
    client: JevClient,
    *,
    model: str = "jev-latest",
    repeats: int = 5,
    interference_questions: int = 19,
) -> DoctorReport:
    report = DoctorReport()
    panel = _panel()
    tokens = 0

    # 1. Determinism.
    repeated: list[dict[str, float]] = []
    for _ in range(repeats):
        answers, used = await _ask(client, PROBE_STATE, panel, model)
        tokens += used
        report.calls += 1
        repeated.append({name: _noul(answers, name) for name in panel})

    sigmas = {
        name: statistics.stdev([r[name] for r in repeated]) if repeats > 1 else 0.0
        for name in panel
    }
    worst = max(sigmas.values()) if sigmas else 0.0
    report.findings.append(
        Finding(
            "determinism",
            worst < 0.01,
            f"worst repeat spread over {repeats} identical calls is {worst:.5f}. "
            + (
                f"Attributions below about {3 * worst:.4f} are noise and jev-why suppresses them."
                if worst > 0
                else "Identical calls agree exactly."
            ),
            worst,
        )
    )

    # 2. Question order.
    # keep_order, because the normal path sorts by name and would erase the
    # very thing this probe measures.
    reversed_panel = {name: panel[name] for name in reversed(list(panel))}
    answers, used = await _ask(client, PROBE_STATE, reversed_panel, model, keep_order=True)
    tokens += used
    report.calls += 1
    order_shift = max(abs(_noul(answers, name) - repeated[0][name]) for name in panel)
    report.findings.append(
        Finding(
            "question order",
            order_shift <= max(3 * worst, 0.01),
            f"reversing the declaration order moved answers by at most {order_shift:.5f}. "
            + (
                "The cache key may sort questions by name."
                if order_shift <= max(3 * worst, 0.01)
                else "The cache key must include question order; sorting is unsound here."
            ),
            order_shift,
        )
    )

    # 3. Question-count interference -- the load-bearing assumption.
    alone_name = "is_urgent"
    alone, used = await _ask(client, PROBE_STATE, {alone_name: panel[alone_name]}, model)
    tokens += used
    report.calls += 1
    crowded_panel: dict[str, QuestionSpec] = {**panel, **_filler(interference_questions)}
    crowded, used = await _ask(client, PROBE_STATE, crowded_panel, model)
    tokens += used
    report.calls += 1
    interference = abs(_noul(alone, alone_name) - _noul(crowded, alone_name))
    report.findings.append(
        Finding(
            "question count",
            interference <= max(3 * worst, 0.01),
            f"asking one question alone versus alongside {len(crowded_panel) - 1} others "
            f"moved it by {interference:.5f}. "
            + (
                "Questions are independent, so one coalition sample can serve the whole panel."
                if interference <= max(3 * worst, 0.01)
                else "Questions interfere. Panel attributions are NOT comparable to "
                "single-question production calls, and the amortisation claim does "
                "not hold on this account."
            ),
            interference,
        )
    )

    # 4. Mask artifact.
    sentences = [s.strip() for s in PROBE_STATE.split(". ") if s.strip()]
    redacted = " ".join([sentences[0], TEXT_PLACEHOLDER, TEXT_PLACEHOLDER, *sentences[3:]])
    probe: dict[str, QuestionSpec] = {ARTIFACT_QUESTION: Noul(instructions=ARTIFACT_INSTRUCTIONS)}
    intact_answers, used = await _ask(client, PROBE_STATE, probe, model)
    tokens += used
    report.calls += 1
    masked_answers, used = await _ask(client, redacted, probe, model)
    tokens += used
    report.calls += 1
    intact = _noul(intact_answers, ARTIFACT_QUESTION)
    masked = _noul(masked_answers, ARTIFACT_QUESTION)
    report.findings.append(
        Finding(
            "mask artifact",
            masked - intact < 0.5,
            f"redacting two sentences raised 'looks damaged' from {intact:.3f} to "
            f"{masked:.3f}. "
            + (
                "Redaction is close enough to the manifold to attribute against."
                if masked - intact < 0.5
                else "Redaction is conspicuous, so attributions partly measure the mask. "
                "Prefer mask='delete' and read the artifact score on every run."
            ),
            masked - intact,
        )
    )

    from jev_why.budget import estimate_cost

    report.cost_usd = estimate_cost(tokens)
    return report


__all__ = ["PROBE_STATE", "DoctorReport", "Finding", "run_doctor"]

"""The doctor's own logic, against scripted clients.

These tests do not measure Jev. They check that the diagnostic reports what it
found, including when what it found is bad news.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jev_why.doctor import run_doctor
from jev_why.types import Answer, JevResponse, Usage


class ScriptedProbe:
    def __init__(
        self,
        *,
        jitter: float = 0.0,
        order_sensitive: bool = False,
        interference: float = 0.0,
        artifact: float = 0.05,
        mask_is_conspicuous: bool = False,
    ) -> None:
        self.jitter = jitter
        self.order_sensitive = order_sensitive
        self.interference = interference
        self.artifact = artifact
        self.mask_is_conspicuous = mask_is_conspicuous
        self.calls = 0

    async def system_one(
        self, state: Any, questions: Mapping[str, Any], *, model: str | None = None
    ) -> JevResponse:
        names = list(questions)
        answers: dict[str, Answer] = {}
        for i, name in enumerate(names):
            value = 0.70
            if self.jitter:
                value += self.jitter * ((self.calls % 3) - 1)
            if self.order_sensitive:
                value += 0.05 * i
            if self.interference and len(names) > 5:
                value += self.interference
            if name.startswith("__jev_why_artifact__"):
                bump = 0.9 if self.mask_is_conspicuous else 0.04
                value = self.artifact + (bump if "[…]" in str(state) else 0.0)
            answers[name] = Answer(qtype="noul", noul=min(1.0, max(0.0, value)))
        self.calls += 1
        return JevResponse("jev-1.13.0", answers, Usage(len(str(state)) // 4, 0))


async def test_a_clean_account_passes_every_check() -> None:
    report = await run_doctor(ScriptedProbe(), repeats=3)
    assert report.all_passed, report.render()
    assert report.calls > 0
    assert report.cost_usd > 0


async def test_a_wobbling_model_fails_determinism_and_quantifies_it() -> None:
    """If repeated identical calls disagree, small attributions are noise
    wearing a number."""
    report = await run_doctor(ScriptedProbe(jitter=0.05), repeats=5)
    finding = next(f for f in report.findings if f.name == "determinism")
    assert not finding.passed
    assert finding.value is not None and finding.value > 0.01
    assert "suppresses" in finding.detail


async def test_order_sensitivity_condemns_the_sorted_cache_key() -> None:
    report = await run_doctor(ScriptedProbe(order_sensitive=True), repeats=3)
    finding = next(f for f in report.findings if f.name == "question order")
    assert not finding.passed
    assert "must include question order" in finding.detail


async def test_question_interference_invalidates_the_amortisation_claim() -> None:
    """The load-bearing assumption. If it fails, the cost argument fails with
    it and the report has to say so plainly."""
    report = await run_doctor(ScriptedProbe(interference=0.2), repeats=3)
    finding = next(f for f in report.findings if f.name == "question count")
    assert not finding.passed
    assert "NOT comparable" in finding.detail
    assert "amortisation claim does not hold" in finding.detail


async def test_a_conspicuous_mask_recommends_deletion_instead() -> None:
    report = await run_doctor(ScriptedProbe(mask_is_conspicuous=True), repeats=3)
    finding = next(f for f in report.findings if f.name == "mask artifact")
    assert not finding.passed
    assert "mask='delete'" in finding.detail


async def test_the_report_explains_what_a_warning_means() -> None:
    rendered = await run_doctor(ScriptedProbe(jitter=0.05), repeats=3)
    assert "does not mean the library is broken" in rendered.render()

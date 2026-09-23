"""Orchestration: chunk, plan, execute, estimate.

This is the only module that wires the pure core to the network, and the only
one most users call.

Two probes ride along with every run because they cost almost nothing and
without them the numbers cannot be interpreted:

  noise floor     Jev is described as deterministic but that is not documented,
                  and if repeated identical calls vary then attributions below
                  that variance are noise presented as insight. So measure it:
                  repeat the unmodified state a few times, take the spread, and
                  flag anything smaller.

  mask artifact   occlusion perturbs the input off the manifold of things the
                  model normally sees, so part of what is measured is the mask
                  rather than the missing content. An extra question -- "does
                  this input look truncated or redacted?" -- rides on every
                  call for about 2% more tokens and measures exactly that.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from jev_why.budget import (
    Budget,
    BudgetGuard,
    amortisation_factor,
    estimate_question_tokens,
    estimate_state_tokens,
)
from jev_why.cache import Cache, resolve_cache
from jev_why.chunking import Chunker, Segmentation, resolve_chunker, with_mask_mode
from jev_why.client import JevClient, make_client
from jev_why.coalitions import EMPTY, CoalitionPlan, kernelshap_plan, occlusion_plan
from jev_why.estimators import resolve_estimator
from jev_why.executor import AsyncExecutor, ExecutorConfig, Progress
from jev_why.faithfulness import (
    DEFAULT_RANDOM_TRIALS,
    DEFAULT_SCHEDULE,
    FaithfulnessReport,
    faithfulness_plan,
    score_faithfulness,
)
from jev_why.questions import CONTEXT_LIMIT_TOKENS, QuestionSpec, panel_payload, validate_panel
from jev_why.types import (
    Attribution,
    Explanation,
    JevRequest,
    JevResponse,
    MaskMode,
    QuestionExplanation,
    Span,
    Spend,
    State,
)
from jev_why.values import ValueSpec, choose_link, magnitude, primary_value

ARTIFACT_QUESTION = "__jev_why_artifact__"
ARTIFACT_INSTRUCTIONS = (
    "Does this input appear truncated, redacted, or incoherent, as if pieces of it were removed?"
)
SIGNIFICANCE_SIGMA = 3.0
DEFAULT_NOISE_PROBES = 3


class ContextTooLarge(ValueError):
    pass


class EndpointsUnavailable(RuntimeError):
    """Neither the unmodified state nor its fully redacted counterpart could be
    scored, so there is nothing to measure against."""


@dataclass(frozen=True)
class _Prepared:
    segmentation: Segmentation
    plan: CoalitionPlan
    payload: dict[str, Any]
    model: str
    warnings: tuple[str, ...]


def _mean_kept_fraction(plan: CoalitionPlan) -> float:
    if not plan.coalitions or plan.n_spans == 0:
        return 1.0
    return sum(len(c) for c in plan.coalitions) / (plan.n_calls * plan.n_spans)


def _prepare(
    state: State,
    questions: Mapping[str, QuestionSpec],
    *,
    chunker: Chunker | str,
    method: str,
    mask: MaskMode | str,
    model: str,
    artifact_probe: bool,
    shapley_budget: int | None,
    seed: int,
) -> _Prepared:
    validate_panel(questions)
    warnings: list[str] = []

    resolved = resolve_chunker(chunker, state)
    segmentation = resolved.segment(state)
    n = len(segmentation.spans)
    if n < 2:
        raise ValueError(
            f"the state split into {n} span(s), so there is nothing to compare. "
            "Use a finer chunker, or lower min_chars."
        )

    if isinstance(mask, str):
        mask = MaskMode(mask)
    if mask is not segmentation.mask_mode:
        warnings.append(
            f"requested mask={mask.value} but the chunker is configured for "
            f"{segmentation.mask_mode.value}; the chunker wins."
        )

    payload = panel_payload(questions)
    if artifact_probe:
        payload[ARTIFACT_QUESTION] = {"type": "noul", "instructions": ARTIFACT_INSTRUCTIONS}

    state_tokens = estimate_state_tokens(state)
    question_tokens = estimate_question_tokens(payload)
    if state_tokens + question_tokens > CONTEXT_LIMIT_TOKENS:
        raise ContextTooLarge(
            f"state plus questions is about {state_tokens + question_tokens:,} tokens, "
            f"over the {CONTEXT_LIMIT_TOKENS:,} limit. Shorten the state or the panel."
        )

    factor = amortisation_factor(state_tokens, question_tokens, len(questions))
    if len(questions) > 1 and factor < 1.5:
        warnings.append(
            f"sharing one coalition sample across this panel saves only {factor:.1f}x, "
            "because the question specs are large relative to the state. They are "
            "input tokens on every call."
        )

    plan = (
        occlusion_plan(n)
        if method == "occlusion"
        else kernelshap_plan(n, budget_calls=shapley_budget, seed=seed)
    )
    return _Prepared(segmentation, plan, payload, model, tuple(warnings))


def _requests(prepared: _Prepared, *, noise_probes: int) -> tuple[list[JevRequest], int]:
    requests = [
        JevRequest(
            state=prepared.segmentation.render(coalition),
            questions=prepared.payload,
            model=prepared.model,
            coalition=coalition,
        )
        for coalition in prepared.plan.coalitions
    ]
    full = frozenset(range(prepared.plan.n_spans))
    probes = [
        # bypass_cache, or a deterministic cache would make repeat variance
        # measure exactly zero by construction.
        JevRequest(
            state=prepared.segmentation.render(full),
            questions=prepared.payload,
            model=prepared.model,
            bypass_cache=True,
        )
        for _ in range(max(0, noise_probes))
    ]
    return requests + probes, len(requests)


def _noise_sigma(probes: Sequence[JevResponse | None], question: str, spec: ValueSpec) -> float:
    values = [
        primary_value(r.answers[question], spec)
        for r in probes
        if r is not None and question in r.answers
    ]
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _artifact_score(responses: Sequence[JevResponse | None]) -> float | None:
    scores = [
        float(r.answers[ARTIFACT_QUESTION].noul or 0.0)
        for r in responses
        if r is not None and ARTIFACT_QUESTION in r.answers
    ]
    return float(sum(scores) / len(scores)) if scores else None


def _build_question_explanation(
    *,
    question: str,
    spec: QuestionSpec,
    spans: Sequence[Span],
    plan: CoalitionPlan,
    responses: Sequence[JevResponse | None],
    probes: Sequence[JevResponse | None],
    link_mode: str,
    estimator_name: str,
    target: str | None,
) -> QuestionExplanation:
    full = frozenset(range(plan.n_spans))
    baseline_response = responses[plan.index_of(full)]
    if baseline_response is None or question not in baseline_response.answers:
        raise RuntimeError(
            f"the unmasked call for {question!r} did not come back, so there is no "
            "baseline to compare against."
        )
    baseline_answer = baseline_response.answers[question]

    if baseline_answer.qtype == "noul":
        baseline_p = float(baseline_answer.noul or 0.0)
    elif baseline_answer.qtype == "choice":
        probs = baseline_answer.probabilities or {}
        baseline_p = float(probs.get(baseline_answer.choice, 0.0))
    else:
        baseline_p = 0.5
    link = choose_link(baseline_p, link_mode)

    value_spec = ValueSpec(question, baseline_answer.qtype, link, target)
    values = np.array(
        [
            primary_value(r.answers[question], value_spec)
            if r is not None and question in r.answers
            else np.nan
            for r in responses
        ],
        dtype=np.float64,
    )

    result = resolve_estimator(estimator_name).estimate(plan, values)
    sigma = _noise_sigma(probes, question, value_spec)

    attributions = []
    for span in spans:
        loo = full - {span.index}
        masked = responses[plan.index_of(loo)] if loo in plan.coalitions else None
        mag = (
            magnitude(baseline_answer, masked.answers[question], baseline_answer.qtype)
            if masked is not None and question in masked.answers
            else abs(float(result.phi[span.index]))
        )
        attributions.append(
            Attribution(
                span=span,
                phi=float(result.phi[span.index]),
                necessity=float(result.necessity[span.index]),
                sufficiency=float(result.sufficiency[span.index]),
                magnitude=float(mag),
                stderr=float(result.stderr[span.index]) if result.stderr is not None else None,
                significant=abs(float(result.phi[span.index])) > SIGNIFICANCE_SIGMA * sigma,
            )
        )

    return QuestionExplanation(
        question=question,
        qtype=baseline_answer.qtype,
        baseline=float(values[plan.index_of(full)]),
        empty=float(values[plan.index_of(EMPTY)]),
        link=link,
        target=target,
        attributions=tuple(attributions),
        efficiency_gap=result.efficiency_gap,
        noise_sigma=sigma,
        estimator=estimator_name,
    )


async def _faithfulness_for(
    *,
    question: str,
    explanation: QuestionExplanation,
    segmentation: Segmentation,
    payload: Mapping[str, Any],
    model: str,
    executor: AsyncExecutor,
    schedule: Sequence[float],
    random_trials: int,
    cross_mask: bool,
    seed: int,
) -> FaithfulnessReport:
    negatives = [
        a.span.index for a in sorted(explanation.attributions, key=lambda a: a.phi) if a.phi < 0
    ]
    plan = faithfulness_plan(
        explanation.ranking(),
        len(segmentation.spans),
        schedule=schedule,
        random_trials=random_trials,
        negative_ranking=negatives,
        seed=seed,
    )

    scoring = segmentation
    if cross_mask:
        other = MaskMode.DELETE if segmentation.mask_mode is MaskMode.REDACT else MaskMode.REDACT
        scoring = with_mask_mode(segmentation, other)

    outcome = await executor.run(
        [
            JevRequest(state=scoring.render(c), questions=payload, model=model, coalition=c)
            for c in plan.coalitions
        ]
    )

    value_spec = ValueSpec(question, explanation.qtype, explanation.link, explanation.target)
    values = [
        primary_value(r.answers[question], value_spec)
        if r is not None and question in r.answers
        else float("nan")
        for r in outcome.responses
    ]
    return score_faithfulness(plan, values, cross_masked=cross_mask)


async def explain_async(
    state: State,
    questions: Mapping[str, QuestionSpec],
    *,
    chunker: Chunker | str = "auto",
    method: str = "occlusion",
    budget: Budget | float | None = None,
    model: str | None = None,
    mask: MaskMode | str = MaskMode.REDACT,
    link: str = "auto",
    target: str | None = None,
    artifact_probe: bool = True,
    noise_probes: int = DEFAULT_NOISE_PROBES,
    shapley_budget: int | None = None,
    faithfulness: bool = False,
    faithfulness_schedule: Sequence[float] = DEFAULT_SCHEDULE,
    random_trials: int = DEFAULT_RANDOM_TRIALS,
    cross_mask: bool = True,
    cache: Cache | str | None = None,
    concurrency: int = 16,
    seed: int = 0,
    api_key: str | None = None,
    client: JevClient | None = None,
    progress: Callable[[Progress], None] | None = None,
) -> Explanation:
    # Build the client first: it owns the model identifier, and that identifier
    # goes into the cache key. Preparing the plan before knowing the provider
    # would key the cache on a name nothing actually answered to -- and the two
    # providers disagree about what a valid name even looks like.
    owned = client is None
    active = client or make_client(api_key=api_key, model=model)

    prepared = _prepare(
        state,
        questions,
        chunker=chunker,
        method=method,
        mask=mask,
        model=model or active.model,
        artifact_probe=artifact_probe,
        shapley_budget=shapley_budget,
        seed=seed,
    )
    requests, n_plan = _requests(prepared, noise_probes=noise_probes)

    guard = BudgetGuard(Budget.of(budget))
    estimate = guard.preflight(
        calls=len(requests),
        state_tokens=estimate_state_tokens(state),
        question_tokens=estimate_question_tokens(prepared.payload),
        mean_kept_fraction=_mean_kept_fraction(prepared.plan),
    )

    executor = AsyncExecutor(
        active,
        cache=resolve_cache(cache),
        budget=guard,
        config=ExecutorConfig(max_concurrency=concurrency, seed=seed),
    )
    try:
        # The two endpoint coalitions are load-bearing in a way the rest are
        # not: every attribution is measured against v(full) and v(empty), so
        # losing either invalidates the whole run. Buy them first. If they
        # cannot be had, this stops after two calls instead of discovering it
        # once the other forty have been paid for.
        endpoints = await executor.run(requests[:2])
        if any(r is None for r in endpoints.responses[:2]):
            reason = "; ".join(endpoints.failures.values()) or "no response"
            raise EndpointsUnavailable(
                "could not score the unmodified state and its fully redacted "
                f"counterpart, so there is no baseline to attribute against: {reason}. "
                "Nothing further was spent. If the provider is throttling, lower "
                "the rate and rerun -- cached answers are reused."
            )

        outcome = await executor.run(requests, progress=progress)
    finally:
        if owned and not faithfulness and hasattr(active, "aclose"):
            await active.aclose()

    plan_responses = outcome.responses[:n_plan]
    probe_responses = outcome.responses[n_plan:]

    warnings = list(prepared.warnings)
    if outcome.truncated:
        warnings.append(
            f"run stopped early after about ${outcome.spend.cost_usd:.4f}; "
            f"the estimate was ${estimate.usd:.4f}. Attributions are based on "
            "partial data."
        )
    if outcome.failures:
        warnings.append(f"{len(outcome.failures)} of {n_plan} calls failed")

    explanations = {}
    for name, spec in questions.items():
        question_explanation = _build_question_explanation(
            question=name,
            spec=spec,
            spans=prepared.segmentation.spans,
            plan=prepared.plan,
            responses=plan_responses,
            probes=probe_responses,
            link_mode=link,
            estimator_name=prepared.plan.estimator,
            target=target,
        )
        if question_explanation.efficiency_gap > 0.25 and method == "occlusion":
            warnings.append(
                f"{name}: spans interact strongly (efficiency gap "
                f"{question_explanation.efficiency_gap:.0%}), so the cheap estimator is "
                f"out of its depth. Re-run with method='shapley' -- with the same cache "
                f"it pays only for the difference."
            )
        if question_explanation.noise_sigma > 0 and not any(
            a.significant for a in question_explanation.attributions
        ):
            warnings.append(
                f"{name}: no span exceeded {SIGNIFICANCE_SIGMA:g} times the measured "
                f"noise floor ({question_explanation.noise_sigma:.4f}). Treat this "
                "explanation as inconclusive."
            )
        explanations[name] = question_explanation

    reports: dict[str, FaithfulnessReport] = {}
    if faithfulness:
        try:
            for name, question_explanation in explanations.items():
                reports[name] = await _faithfulness_for(
                    question=name,
                    explanation=question_explanation,
                    segmentation=prepared.segmentation,
                    payload=prepared.payload,
                    model=prepared.model,
                    executor=executor,
                    schedule=faithfulness_schedule,
                    random_trials=random_trials,
                    cross_mask=cross_mask,
                    seed=seed,
                )
                if not reports[name].credible:
                    warnings.append(f"{name}: {reports[name].verdict()}")
        finally:
            if owned and hasattr(active, "aclose"):
                await active.aclose()

    artifact = _artifact_score(plan_responses) if artifact_probe else None
    if artifact is not None and artifact > 0.5:
        warnings.append(
            f"masked variants looked truncated to Jev itself (artifact score "
            f"{artifact:.2f}), so these attributions partly measure the mask rather "
            "than the missing content."
        )

    return Explanation(
        spans=tuple(prepared.segmentation.spans),
        questions=explanations,
        spend=Spend(
            outcome.spend.calls,
            outcome.spend.cache_hits,
            outcome.spend.input_tokens,
            outcome.spend.cost_usd,
        ),
        model_version=outcome.model_version or prepared.model,
        mask_mode=prepared.segmentation.mask_mode,
        seed=seed,
        artifact_score=artifact,
        warnings=tuple(warnings),
        faithfulness=reports,
    )


def explain(state: State, questions: Mapping[str, QuestionSpec], **kwargs: Any) -> Explanation:
    """Synchronous front door. One implementation, two entrances."""
    return asyncio.run(explain_async(state, questions, **kwargs))


__all__ = [
    "ARTIFACT_QUESTION",
    "ContextTooLarge",
    "EndpointsUnavailable",
    "explain",
    "explain_async",
]

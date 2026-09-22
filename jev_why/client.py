"""The network boundary.

This module and executor.py are the only places that talk to the API. Nothing
here does any arithmetic that matters; the job is to build a request, make the
call, and normalise the response into the plain types the pure core works on.

Retries, Retry-After handling and per-call backoff are delegated to the SDK's
RetryPolicy rather than reimplemented. What jev-why adds on top is concurrency
discovery, which lives in executor.py.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from jev_why.types import Answer, JevResponse, State, Usage

OFFLINE_ENV = "JEV_WHY_OFFLINE"
API_KEY_ENV = "TYPESAFE_API_KEY"
MODEL_ENV = "JEV_WHY_MODEL"
DEFAULT_MODEL = "jev-latest"


class OfflineError(RuntimeError):
    """A real client was requested while the offline guard is set.

    The guard exists so a test that forgets to inject a fake fails loudly
    instead of quietly spending someone's quota.
    """


class MissingApiKey(RuntimeError):
    pass


class ModelVersionDrift(RuntimeError):
    """Responses in one run came from two different model versions.

    Attribution compares probabilities across calls. If half the coalitions were
    scored by one version and half by another after a mid-run rollout, every
    delta is contaminated and the explanation is wrong in a way nothing else
    would reveal. That is why jev-latest is the wrong default for this use case.
    """


class JevClient(Protocol):
    async def system_one(
        self, state: State, questions: Mapping[str, Any], *, model: str | None = None
    ) -> JevResponse: ...


def _convert_answer(raw: Any) -> Answer:
    kind = getattr(raw, "type", None)
    if kind == "noul":
        return Answer(qtype="noul", noul=float(raw.noul))
    if kind == "choice":
        return Answer(
            qtype="choice",
            choice=raw.choice,
            probabilities=dict(raw.probabilities or {}),
            confidence=float(raw.confidence) if raw.confidence is not None else None,
        )
    if kind == "score":
        return Answer(
            qtype="score",
            score=float(raw.score),
            probabilities={int(k): float(v) for k, v in (raw.probabilities or {}).items()},
            confidence=float(raw.confidence) if raw.confidence is not None else None,
            legend=dict(raw.legend or {}),
        )
    raise ValueError(f"unrecognised answer type {kind!r}")


def convert_response(raw: Any) -> JevResponse:
    usage = getattr(raw, "usage", None)
    return JevResponse(
        model=raw.model,
        answers={name: _convert_answer(a) for name, a in raw.answers.items()},
        usage=Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        ),
    )


def _build_questions(payload: Mapping[str, Any]) -> dict[str, Any]:
    import typesafe_sdk as ts

    built: dict[str, Any] = {}
    for name, spec in payload.items():
        match spec["type"]:
            case "noul":
                built[name] = ts.Noul(**spec)
            case "choice":
                built[name] = ts.Choice(**spec)
            case "score":
                built[name] = ts.Score(**spec)
            case other:
                raise ValueError(f"question {name!r}: unknown type {other!r}")
    return built


@dataclass
class TypeSafeJevClient:
    """Thin wrapper over the official async SDK client."""

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    timeout_s: float = 30.0
    max_retries: int = 5

    def __post_init__(self) -> None:
        if os.environ.get(OFFLINE_ENV):
            raise OfflineError(
                f"{OFFLINE_ENV} is set, so no live client will be built. "
                "Inject a fake client or a populated cache instead."
            )
        key = self.api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingApiKey(
                f"set {API_KEY_ENV} or pass api_key=. jev-why never writes the key "
                "to disk or into the response cache."
            )
        import typesafe_sdk as ts

        self._client = ts.AsyncTypeSafeClient(
            api_key=key,
            timeout=self.timeout_s,
            # Let the SDK own per-call retry: it already honours Retry-After and
            # jitters its backoff. Duplicating that here would just make the two
            # layers fight over the same 429.
            retry=ts.RetryPolicy(max_retries=self.max_retries, respect_retry_after=True),
        )

    async def system_one(
        self, state: State, questions: Mapping[str, Any], *, model: str | None = None
    ) -> JevResponse:
        raw = await self._client.system_one(
            state=state, questions=_build_questions(questions), model=model or self.model
        )
        return convert_response(raw)

    async def aclose(self) -> None:
        await self._client.aclose()


def resolve_model(model: str | None) -> str:
    return model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def check_model_consistency(seen: str | None, incoming: str, *, strict: bool = True) -> str:
    if seen is None:
        return incoming
    if seen != incoming and strict:
        raise ModelVersionDrift(
            f"this run started on {seen} and is now being answered by {incoming}. "
            f"Probabilities are not comparable across versions, so the attributions "
            f"would be meaningless. Pin a concrete version with {MODEL_ENV} and rerun."
        )
    return seen


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MODEL",
    "MODEL_ENV",
    "OFFLINE_ENV",
    "JevClient",
    "MissingApiKey",
    "ModelVersionDrift",
    "OfflineError",
    "TypeSafeJevClient",
    "check_model_consistency",
    "convert_response",
    "resolve_model",
]

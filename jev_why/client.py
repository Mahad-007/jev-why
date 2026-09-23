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
JEVAI_KEY_ENV = "JEV_API_KEY"
PROVIDER_ENV = "JEV_WHY_PROVIDER"
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
    @property
    def model(self) -> str:
        """The concrete model this client asks for.

        Providers disagree about what a model identifier looks like -- TypeSafe
        takes `jev-latest`, jevai.org rejects it and wants `typesafe-ai/jev` --
        so the client owns the name and callers read it rather than guessing.
        It also goes into the cache key, which is why an empty default would be
        wrong: two providers must not share entries.
        """

    async def system_one(
        self, state: State, questions: Mapping[str, Any], *, model: str | None = None
    ) -> JevResponse: ...

    async def aclose(self) -> None: ...


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


# --------------------------------------------------------------------------
# jevai.org
# --------------------------------------------------------------------------

JEVAI_BASE_URL = "https://www.jevai.org"
JEVAI_PATH = "/api/v1/decisions"
JEVAI_BODY_LIMIT = 32 * 1024
JEVAI_MODEL = "typesafe-ai/jev"


class JevRateLimitError(RuntimeError):
    """The provider is throttling.

    Named so the executor's throttle detection recognises it. jevai.org signals
    this with HTTP 200 and code -1 rather than a 429, so it has to be raised
    from the envelope rather than the status line.
    """


class JevApiError(RuntimeError):
    pass


class BodyTooLarge(ValueError):
    pass


@dataclass
class JevAiClient:
    """Client for jevai.org, which speaks the same question and answer shapes
    as TypeSafe's own API behind a different envelope.

    Three differences drive the code below:

      envelope   every reply is {code, message, data} with HTTP 200, including
                 errors and throttling, so success cannot be read off the
                 status code.

      no usage   the response carries no token count, so spend is estimated
                 rather than measured and the budget guard cannot self-
                 calibrate. Anything reporting cost through this provider has
                 to say it is an estimate.

      32 KiB     a hard body limit, checked before sending so an oversized
                 state fails locally instead of costing a round trip.
    """

    api_key: str | None = None
    model: str = JEVAI_MODEL
    base_url: str = JEVAI_BASE_URL
    timeout_s: float = 60.0
    transport: Any = None
    """An injected transport, for tests. Supplying one bypasses the offline
    guard and the key requirement, because a mock transport cannot reach the
    network however the environment is configured."""

    def __post_init__(self) -> None:
        if self.transport is not None:
            import httpx2

            self._key = self.api_key or "injected"
            self._client = httpx2.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_s,
                transport=self.transport,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            return
        if os.environ.get(OFFLINE_ENV):
            raise OfflineError(
                f"{OFFLINE_ENV} is set, so no live client will be built. "
                "Inject a fake client or a populated cache instead."
            )
        key = self.api_key or os.environ.get(JEVAI_KEY_ENV) or os.environ.get(API_KEY_ENV)
        if not key:
            raise MissingApiKey(
                f"set {JEVAI_KEY_ENV} or pass api_key=. jev-why never writes the key "
                "to disk or into the response cache."
            )
        import httpx2

        self._key = key
        self._client = httpx2.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )

    async def system_one(
        self, state: State, questions: Mapping[str, Any], *, model: str | None = None
    ) -> JevResponse:
        import json

        body: dict[str, Any] = {"state": state, "questions": dict(questions)}
        chosen = model or self.model
        if chosen and chosen not in ("jev-latest", ""):
            body["model"] = chosen

        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        if len(encoded) > JEVAI_BODY_LIMIT:
            raise BodyTooLarge(
                f"request body is {len(encoded):,} bytes, over the "
                f"{JEVAI_BODY_LIMIT:,} byte limit at {self.base_url}. Shorten the "
                "state, or chunk more coarsely so each masked variant is smaller."
            )

        response = await self._client.post(JEVAI_PATH, content=encoded)
        if response.status_code in (429, 529):
            raise JevRateLimitError(f"{response.status_code} from {self.base_url}")
        if response.status_code == 401:
            raise MissingApiKey(f"{self.base_url} rejected the key (401)")

        payload = response.json()
        if payload.get("code") != 0:
            message = str(payload.get("message", "unknown error"))
            if "too many requests" in message.lower() or "rate limit" in message.lower():
                raise JevRateLimitError(message)
            raise JevApiError(f"{message} (http {response.status_code})")

        data = payload.get("data") or {}
        answers = {
            name: _convert_answer(_Wrap(raw)) for name, raw in (data.get("answers") or {}).items()
        }
        # No usage in the envelope, so estimate from what was actually sent.
        return JevResponse(
            model=str(data.get("model") or chosen or JEVAI_MODEL),
            answers=answers,
            usage=Usage(input_tokens=max(1, len(encoded) // 4), output_tokens=0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class _Wrap:
    """Adapt a plain dict answer to the attribute access _convert_answer wants,
    so both providers share one conversion path."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._raw = raw

    def __getattr__(self, name: str) -> Any:
        return self._raw.get(name)


def make_client(
    *, api_key: str | None = None, model: str | None = None, provider: str | None = None
) -> JevClient:
    """Pick a provider.

    Chosen explicitly with JEV_WHY_PROVIDER, otherwise inferred from the key:
    a `jev_` prefix is a jevai.org key, anything else is TypeSafe's own.
    """
    key = api_key or os.environ.get(JEVAI_KEY_ENV) or os.environ.get(API_KEY_ENV) or ""
    choice = (provider or os.environ.get(PROVIDER_ENV) or "").lower()
    if not choice:
        choice = "jevai" if key.startswith("jev_") else "typesafe"

    if choice == "jevai":
        return JevAiClient(api_key=api_key or key or None, model=model or JEVAI_MODEL)
    if choice == "typesafe":
        return TypeSafeJevClient(api_key=api_key, model=resolve_model(model))
    raise ValueError(f"unknown provider {choice!r}; use 'jevai' or 'typesafe'")

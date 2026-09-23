"""Provider behaviour, against a mock transport. No network, no key."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from jev_why.client import (
    JEVAI_BODY_LIMIT,
    BodyTooLarge,
    JevAiClient,
    JevApiError,
    JevRateLimitError,
    MissingApiKey,
    ModelVersionDrift,
    OfflineError,
    check_model_consistency,
    make_client,
)

ANSWERS = {
    "is_injection": {"type": "noul", "noul": 0.99},
    "topic": {
        "type": "choice",
        "choice": "security",
        "confidence": 1,
        "probabilities": {"security": 1, "billing": 0},
    },
    "severity": {
        "type": "score",
        "score": 1.82,
        "confidence": 0.72,
        "legend": {"0": "none", "1": "low", "2": "high"},
        "probabilities": {"0": 0.01, "1": 0.16, "2": 0.83},
    },
}


def _client(handler: Any, **kwargs: Any) -> JevAiClient:
    return JevAiClient(api_key="jev_test", transport=httpx2.MockTransport(handler), **kwargs)


def _ok(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json={"code": 0, "message": "ok", "data": {"answers": ANSWERS}})


async def test_all_three_answer_shapes_are_converted() -> None:
    client = _client(_ok)
    response = await client.system_one("x", {"q": {"type": "noul"}})

    assert response.answers["is_injection"].noul == 0.99
    assert response.answers["topic"].choice == "security"
    assert response.answers["topic"].confidence == 1.0
    # Score levels are keyed by index, which the expected-level calculation needs.
    assert list(response.answers["severity"].probabilities or {}) == [0, 1, 2]
    assert response.answers["severity"].score == 1.82


async def test_throttling_arrives_as_http_200_and_must_still_be_detected() -> None:
    """jevai.org signals a throttle with code -1 and a 200, so reading the
    status line alone would treat it as a successful answer."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "code": -1,
                "message": "Too many requests. Please try again later.",
                "data": None,
            },
        )

    with pytest.raises(JevRateLimitError, match="Too many requests"):
        await _client(handler).system_one("x", {"q": {"type": "noul"}})


async def test_an_edge_429_is_also_a_throttle() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, text="rate limited")

    with pytest.raises(JevRateLimitError):
        await _client(handler).system_one("x", {"q": {"type": "noul"}})


async def test_the_throttle_name_is_what_the_executor_looks_for() -> None:
    """The executor lowers concurrency by matching the exception name, so this
    coupling deserves a test rather than a comment."""
    from jev_why.executor import _is_throttle

    assert _is_throttle(JevRateLimitError("x"))
    assert not _is_throttle(JevApiError("x"))


async def test_other_envelope_errors_are_not_mistaken_for_throttling() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json={"code": -1, "message": "model must be a Jev identifier.", "data": None}
        )

    with pytest.raises(JevApiError, match="Jev identifier"):
        await _client(handler).system_one("x", {"q": {"type": "noul"}})


async def test_an_oversized_body_fails_locally_rather_than_costing_a_call() -> None:
    calls = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return _ok(request)

    with pytest.raises(BodyTooLarge, match="32,768 byte limit"):
        await _client(handler).system_one("x" * (JEVAI_BODY_LIMIT + 1000), {"q": {"type": "noul"}})
    assert calls == [], "nothing may be sent once the body is known to be too large"


async def test_jev_latest_is_not_sent_because_the_provider_rejects_it() -> None:
    """jevai.org wants an identifier like typesafe-ai/jev and 400s on
    jev-latest, so the alias must not be forwarded."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return _ok(request)

    await _client(handler).system_one("x", {"q": {"type": "noul"}}, model="jev-latest")
    assert "model" not in seen[0]

    await _client(handler).system_one("x", {"q": {"type": "noul"}}, model="typesafe-ai/jev")
    assert seen[1]["model"] == "typesafe-ai/jev"


async def test_usage_is_estimated_because_the_provider_reports_none() -> None:
    """Anything quoting a cost through this provider is quoting an estimate,
    and the budget guard cannot self-calibrate against it."""
    response = await _client(_ok).system_one("a fairly long state string", {"q": {"type": "noul"}})
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens == 0


async def test_a_rejected_key_says_so_plainly() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"detail": "nope"})

    with pytest.raises(MissingApiKey, match="401"):
        await _client(handler).system_one("x", {"q": {"type": "noul"}})


def test_the_provider_is_inferred_from_the_key_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JEV_WHY_OFFLINE", raising=False)
    assert isinstance(make_client(api_key="jev_abc"), JevAiClient)


def test_an_explicit_provider_overrides_the_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JEV_WHY_OFFLINE", raising=False)
    with pytest.raises(ValueError, match="unknown provider"):
        make_client(api_key="jev_abc", provider="telepathy")


def test_the_offline_guard_blocks_both_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEV_WHY_OFFLINE", "1")
    with pytest.raises(OfflineError):
        JevAiClient(api_key="jev_abc")


def test_a_missing_key_names_the_variable_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEV_WHY_OFFLINE", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(MissingApiKey, match="JEV_API_KEY"):
        JevAiClient()


def test_a_mid_run_version_change_is_refused() -> None:
    assert check_model_consistency(None, "jev-1.13.0") == "jev-1.13.0"
    assert check_model_consistency("jev-1.13.0", "jev-1.13.0") == "jev-1.13.0"
    with pytest.raises(ModelVersionDrift, match="not comparable"):
        check_model_consistency("jev-1.13.0", "jev-1.14.0")


def test_version_drift_can_be_allowed_deliberately() -> None:
    assert check_model_consistency("a", "b", strict=False) == "a"

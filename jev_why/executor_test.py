from __future__ import annotations

import pytest

from jev_why.budget import Budget, BudgetGuard, estimate_cost
from jev_why.cache import MemoryCache
from jev_why.client import ModelVersionDrift
from jev_why.executor import AsyncExecutor, ExecutorConfig
from jev_why.types import Answer, JevRequest, JevResponse, Usage


class ScriptedClient:
    """Answers from a script, counting calls and failing on demand."""

    def __init__(
        self,
        *,
        fail_indices: set[int] | None = None,
        model: str = "jev-1.13.0",
        error: type[Exception] = RuntimeError,
        tokens: int = 100,
    ) -> None:
        self.calls = 0
        self.fail_indices = fail_indices or set()
        self.model = model
        self.error = error
        self.tokens = tokens
        self.seen: list[object] = []

    async def system_one(
        self, state: object, questions: object, *, model: str | None = None
    ) -> JevResponse:
        ordinal = self.calls
        self.calls += 1
        self.seen.append(state)
        if ordinal in self.fail_indices:
            raise self.error(f"scripted failure {ordinal}")
        return JevResponse(self.model, {"q": Answer(qtype="noul", noul=0.5)}, Usage(self.tokens, 0))


class RateLimitError(RuntimeError):
    pass


def _requests(n: int, model: str = "jev-latest") -> list[JevRequest]:
    return [
        JevRequest(state=f"state-{i}", questions={"q": {"type": "noul"}}, model=model)
        for i in range(n)
    ]


async def _noop_sleep(_seconds: float) -> None:
    return None


async def test_every_request_gets_a_response() -> None:
    client = ScriptedClient()
    result = await AsyncExecutor(client, sleep=_noop_sleep).run(_requests(10))
    assert client.calls == 10
    assert all(r is not None for r in result.responses)
    assert result.ok


async def test_cached_responses_are_never_re_requested() -> None:
    """The checkpoint property: a rerun pays only for what it has not done."""
    cache = MemoryCache()
    client = ScriptedClient()
    requests = _requests(6)
    await AsyncExecutor(client, cache=cache, sleep=_noop_sleep).run(requests)
    assert client.calls == 6

    second = ScriptedClient()
    result = await AsyncExecutor(second, cache=cache, sleep=_noop_sleep).run(requests)
    assert second.calls == 0
    assert result.spend.cache_hits == 6


async def test_a_partial_rerun_pays_only_for_the_difference() -> None:
    """What makes upgrading occlusion to Shapley an incremental purchase."""
    cache = MemoryCache()
    await AsyncExecutor(ScriptedClient(), cache=cache, sleep=_noop_sleep).run(_requests(4))
    client = ScriptedClient()
    await AsyncExecutor(client, cache=cache, sleep=_noop_sleep).run(_requests(10))
    assert client.calls == 6


async def test_failures_are_reported_per_request_not_raised() -> None:
    """One bad coalition must not discard the rest of a paid run."""
    client = ScriptedClient(fail_indices={0, 1, 2, 3})
    config = ExecutorConfig(max_attempts=1, max_concurrency=1)
    result = await AsyncExecutor(client, config=config, sleep=_noop_sleep).run(_requests(4))
    assert len(result.failures) == 4
    assert not result.ok


async def test_a_transient_failure_is_retried_and_succeeds() -> None:
    client = ScriptedClient(fail_indices={0})
    config = ExecutorConfig(max_attempts=3, max_concurrency=1)
    result = await AsyncExecutor(client, config=config, sleep=_noop_sleep).run(_requests(1))
    assert result.failures == {}
    assert result.responses[0] is not None
    assert client.calls == 2


async def test_a_budget_trip_keeps_what_was_already_paid_for() -> None:
    """Abandoning successful calls because of a stop on a later one would be
    its own kind of waste."""
    guard = BudgetGuard(Budget(max_usd=estimate_cost(250)))
    client = ScriptedClient(tokens=100)
    config = ExecutorConfig(max_concurrency=1)
    result = await AsyncExecutor(client, budget=guard, config=config, sleep=_noop_sleep).run(
        _requests(50)
    )
    assert result.truncated
    assert sum(1 for r in result.responses if r is not None) >= 2
    assert client.calls < 50, "the run must actually stop"


async def test_a_mid_run_model_rollout_fails_loudly() -> None:
    """Comparing probabilities across model versions produces contaminated
    deltas that nothing downstream could detect."""

    class RollingClient(ScriptedClient):
        async def system_one(
            self, state: object, questions: object, *, model: str | None = None
        ) -> JevResponse:
            version = "jev-1.13.0" if self.calls < 2 else "jev-1.14.0"
            self.calls += 1
            return JevResponse(version, {"q": Answer(qtype="noul", noul=0.5)}, Usage(10, 0))

    config = ExecutorConfig(max_concurrency=1, max_attempts=1)
    executor = AsyncExecutor(RollingClient(), config=config, sleep=_noop_sleep)
    with pytest.raises(ModelVersionDrift, match="not comparable across versions"):
        await executor.run(_requests(4))


async def test_throttling_reduces_concurrency() -> None:
    client = ScriptedClient(fail_indices={0, 1}, error=RateLimitError)
    executor = AsyncExecutor(
        client, config=ExecutorConfig(max_concurrency=16, max_attempts=3), sleep=_noop_sleep
    )
    await executor.run(_requests(3))
    assert executor._aimd.limit < 16


async def test_bypass_cache_requests_are_neither_read_nor_written() -> None:
    """The noise probe measures repeat variance, so a deterministic cache would
    make it measure zero by construction."""
    cache = MemoryCache()
    client = ScriptedClient()
    probes = [
        JevRequest(state="same", questions={"q": {}}, model="m", bypass_cache=True)
        for _ in range(3)
    ]
    await AsyncExecutor(client, cache=cache, sleep=_noop_sleep).run(probes)
    assert client.calls == 3
    assert cache.entries == {}


async def test_progress_is_reported_as_work_completes() -> None:
    seen: list[int] = []
    await AsyncExecutor(ScriptedClient(), sleep=_noop_sleep).run(
        _requests(5), progress=lambda p: seen.append(p.done)
    )
    assert len(seen) == 5
    assert max(seen) == 5


async def test_spend_reflects_the_tokens_the_api_reported() -> None:
    client = ScriptedClient(tokens=250)
    result = await AsyncExecutor(client, sleep=_noop_sleep).run(_requests(4))
    assert result.spend.input_tokens == 1000
    assert result.spend.cost_usd == pytest.approx(estimate_cost(1000))

"""Async fan-out over a coalition plan.

I/O. The pure core decides *what* to evaluate; this decides how fast and how
safely to actually do it.

Three guarantees matter here:

  checkpointing   every response is cached the moment it arrives, so an abort
                  at call 900 of 1000 keeps 900 of them
  partial results a budget trip or a run of failures returns what succeeded
                  along with what did not, rather than discarding paid work
  honest failure  a coalition that never got an answer comes back as NaN, and
                  the estimator reports how many rows it dropped
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

from jev_why.budget import BudgetExceeded, BudgetGuard
from jev_why.cache import Cache, CacheMiss, NullCache, cache_key
from jev_why.client import JevClient, ModelVersionDrift, check_model_consistency
from jev_why.ratelimit import AimdController, TokenBucket, backoff_delay
from jev_why.types import JevRequest, JevResponse, Spend


@dataclass(frozen=True)
class ExecutorConfig:
    max_concurrency: int = 16
    requests_per_minute: float | None = None
    max_attempts: int = 4
    adaptive: bool = True
    seed: int = 0
    backoff_base_s: float = 0.25
    backoff_cap_s: float = 20.0
    """How long to wait between retries, as base * 2**attempt capped at cap.

    The defaults suit a per-second rate limit, where the first retry should be
    quick. A provider that enforces a quota over an hour is a different animal:
    retrying in a quarter of a second just spends another unit of a quota that
    has already run out, so both numbers need raising together. Raising only
    the cap does nothing, since it takes eleven doublings to get there from
    0.25.

    A run against such a provider is survivable at all only because every
    successful response is cached the moment it arrives, so an interrupted run
    resumes without paying for anything twice.
    """


@dataclass(frozen=True)
class Progress:
    done: int
    total: int
    cache_hits: int
    spent_usd: float


@dataclass
class ExecutionResult:
    responses: list[JevResponse | None]
    failures: dict[int, str] = field(default_factory=dict)
    spend: Spend = field(default_factory=Spend)
    model_version: str | None = None
    truncated: bool = False
    """True when the run stopped early. The caller still gets everything that
    was paid for."""

    @property
    def ok(self) -> bool:
        return not self.failures and not self.truncated


@dataclass
class _RunState:
    """Mutable bookkeeping shared by the fan-out tasks."""

    done: int = 0
    hits: int = 0
    model: str | None = None
    stop: bool = False
    drift: ModelVersionDrift | None = None
    """Set when responses in one run came from two model versions. This is not
    a per-request failure -- it invalidates every delta in the run -- so it
    stops the fan-out and is re-raised to the caller as a single clear error
    rather than surfacing as a task-group exception group."""

    def should_stop(self) -> bool:
        """Read through a method rather than the attribute directly.

        Sibling tasks set `stop` concurrently, which a narrowing type checker
        cannot see -- and neither can a reader skimming for why the flag is
        checked twice.
        """
        return self.stop


class _AdaptiveGate:
    """Concurrency limiter whose ceiling can move while calls are in flight.

    asyncio.Semaphore is fixed at construction, which is no use when the point
    is to discover an undocumented rate limit rather than assume one.
    """

    def __init__(self, controller: AimdController) -> None:
        self._controller = controller
        self._in_flight = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._condition:
            while self._in_flight >= self._controller.limit:
                await self._condition.wait()
            self._in_flight += 1

    async def release(self) -> None:
        async with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()


class AsyncExecutor:
    def __init__(
        self,
        client: JevClient,
        *,
        cache: Cache | None = None,
        budget: BudgetGuard | None = None,
        config: ExecutorConfig | None = None,
        sleep: Callable[[float], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.client = client
        self.cache: Cache = cache or NullCache()
        self.budget = budget or BudgetGuard()
        self.config = config or ExecutorConfig()
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or (lambda: asyncio.get_event_loop().time())
        self._rng = Random(self.config.seed)
        self._aimd = AimdController(
            limit=self.config.max_concurrency, maximum=self.config.max_concurrency
        )
        self._bucket = (
            TokenBucket(
                rate=self.config.requests_per_minute / 60.0,
                capacity=max(1.0, self.config.max_concurrency),
            )
            if self.config.requests_per_minute
            else None
        )

    async def run(
        self,
        requests: Sequence[JevRequest],
        *,
        progress: Callable[[Progress], None] | None = None,
    ) -> ExecutionResult:
        responses: list[JevResponse | None] = [None] * len(requests)
        failures: dict[int, str] = {}
        gate = _AdaptiveGate(self._aimd)
        state = _RunState()

        async def one(index: int, request: JevRequest) -> None:
            if state.should_stop():
                return
            key = cache_key(model=request.model, state=request.state, questions=request.questions)

            if not request.bypass_cache:
                try:
                    cached = self.cache.get(key)
                except CacheMiss as miss:
                    failures[index] = str(miss)
                    return
                if cached is not None:
                    responses[index] = cached
                    state.hits += 1
                    state.done += 1
                    if not _track_model(state, cached.model):
                        return
                    _report(progress, state, len(requests), self.budget)
                    return

            for attempt in range(self.config.max_attempts):
                if state.should_stop():
                    return
                await self._pace()
                await gate.acquire()
                try:
                    response = await self.client.system_one(
                        request.state, request.questions, model=request.model
                    )
                except Exception as error:
                    if self.config.adaptive and _is_throttle(error):
                        self._aimd.on_throttle(self._clock())
                    if attempt == self.config.max_attempts - 1:
                        failures[index] = f"{type(error).__name__}: {error}"
                        state.done += 1
                        _report(progress, state, len(requests), self.budget)
                        return
                    await self._sleep(
                        backoff_delay(
                            attempt,
                            base=self.config.backoff_base_s,
                            cap=self.config.backoff_cap_s,
                            rng=self._rng,
                        )
                    )
                    continue
                finally:
                    await gate.release()

                if self.config.adaptive:
                    self._aimd.on_success(self._clock())

                # Cache first: the response is paid for whatever happens next.
                if not request.bypass_cache:
                    self.cache.put(key, response)
                responses[index] = response
                if not _track_model(state, response.model):
                    return
                self.budget.record(response.usage.input_tokens)
                state.done += 1
                _report(progress, state, len(requests), self.budget)
                try:
                    self.budget.check()
                except BudgetExceeded as exceeded:
                    state.stop = True
                    failures[index] = str(exceeded)
                return

        # TaskGroup, not gather: a budget trip or a model-version change has to
        # cancel the calls still in flight rather than let them keep spending.
        # It is 3.11+, which is what sets this package's floor.
        async with asyncio.TaskGroup() as group:
            for index, request in enumerate(requests):
                group.create_task(one(index, request))

        if state.drift is not None:
            raise state.drift

        self.budget.recalibrate()
        spend = Spend(
            calls=self.budget.calls,
            cache_hits=state.hits,
            input_tokens=self.budget.spent_tokens,
            cost_usd=self.budget.spent_usd,
        )
        return ExecutionResult(responses, failures, spend, state.model, truncated=state.stop)

    async def _pace(self) -> None:
        now = self._clock()
        wait = self._aimd.wait_for(now)
        if self._bucket is not None:
            wait = max(wait, self._bucket.try_acquire(1.0, now))
        if wait > 0:
            await self._sleep(wait)


def _track_model(state: _RunState, model: str) -> bool:
    """Record which version answered, stopping the run if it changed."""
    try:
        state.model = check_model_consistency(state.model, model)
    except ModelVersionDrift as drift:
        state.drift = drift
        state.stop = True
        return False
    return True


def _report(
    progress: Callable[[Progress], None] | None, state: _RunState, total: int, budget: BudgetGuard
) -> None:
    if progress is not None:
        with contextlib.suppress(Exception):
            progress(Progress(state.done, total, state.hits, budget.spent_usd))


def _is_throttle(error: Exception) -> bool:
    name = type(error).__name__.lower()
    return "ratelimit" in name or "overload" in name or "toomany" in name


__all__ = ["AsyncExecutor", "ExecutionResult", "ExecutorConfig", "Progress"]

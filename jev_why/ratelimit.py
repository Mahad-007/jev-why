"""Pacing, as pure state machines.

Pure: the clock is injected and the RNG is seeded, so every branch here is
testable to the millisecond without sleeping and without a network.

The SDK ships its own RetryPolicy, which handles per-call retries, Retry-After
and jittered backoff. This module deliberately does not duplicate that. What it
adds is the layer above: how many calls to have in flight at once against an
API whose rate limits are not documented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random


@dataclass
class TokenBucket:
    """Classic token bucket. `try_acquire` returns how long to wait, so the
    caller owns the sleeping and this stays pure."""

    rate: float
    capacity: float
    tokens: float = field(default=0.0)
    updated: float = 0.0

    def __post_init__(self) -> None:
        if self.tokens == 0.0:
            self.tokens = self.capacity

    def try_acquire(self, amount: float, now: float) -> float:
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = now
        if self.tokens >= amount:
            self.tokens -= amount
            return 0.0
        if self.rate <= 0:
            return float("inf")
        return (amount - self.tokens) / self.rate


def backoff_delay(
    attempt: int, *, base: float = 0.25, cap: float = 20.0, rng: Random | None = None
) -> float:
    """Full jitter: uniform over [0, min(cap, base * 2**attempt)].

    Full jitter rather than a decorrelated variant because this fan-out is
    bursty and synchronised. Every in-flight call hits the same throttle at the
    same moment, and retrying them in lockstep is exactly what turns one 429
    into a sustained one.
    """
    ceiling = min(cap, base * (2**attempt))
    return (rng or Random()).uniform(0.0, ceiling)


@dataclass
class AimdController:
    """Additive-increase, multiplicative-decrease concurrency.

    Jev's rate limits are not published, so rather than guess a number and
    either leave throughput on the table or hammer the API, start conservative
    and discover the ceiling: creep up while calls succeed, halve on a throttle,
    and hold through a cooldown so the probe does not immediately re-trip.
    """

    limit: int = 8
    minimum: int = 1
    maximum: int = 64
    probe_interval: int = 20
    cooldown_s: float = 2.0
    successes: int = 0
    cooling_until: float = 0.0

    def on_success(self, now: float = 0.0) -> int:
        if now < self.cooling_until:
            return self.limit
        self.successes += 1
        if self.successes >= self.probe_interval:
            self.successes = 0
            self.limit = min(self.maximum, self.limit + 1)
        return self.limit

    def on_throttle(self, now: float = 0.0, retry_after: float | None = None) -> int:
        self.successes = 0
        self.limit = max(self.minimum, self.limit // 2)
        self.cooling_until = now + (retry_after if retry_after is not None else self.cooldown_s)
        return self.limit

    def wait_for(self, now: float) -> float:
        return max(0.0, self.cooling_until - now)


__all__ = ["AimdController", "TokenBucket", "backoff_delay"]

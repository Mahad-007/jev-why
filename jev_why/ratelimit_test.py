from __future__ import annotations

from random import Random

from jev_why.ratelimit import AimdController, TokenBucket, backoff_delay


def test_bucket_grants_until_empty_then_quotes_a_wait() -> None:
    bucket = TokenBucket(rate=10.0, capacity=5.0)
    assert [bucket.try_acquire(1.0, 0.0) for _ in range(5)] == [0.0] * 5
    assert bucket.try_acquire(1.0, 0.0) == 0.1


def test_bucket_refills_with_elapsed_time() -> None:
    bucket = TokenBucket(rate=10.0, capacity=5.0)
    for _ in range(5):
        bucket.try_acquire(1.0, 0.0)
    assert bucket.try_acquire(1.0, 1.0) == 0.0


def test_bucket_never_refills_past_capacity() -> None:
    bucket = TokenBucket(rate=10.0, capacity=5.0)
    bucket.try_acquire(5.0, 0.0)
    bucket.try_acquire(0.0, 1_000.0)
    assert bucket.tokens == 5.0


def test_backoff_is_bounded_and_jittered() -> None:
    rng = Random(0)
    delays = [backoff_delay(4, base=0.25, cap=20.0, rng=rng) for _ in range(50)]
    assert all(0.0 <= d <= 4.0 for d in delays)
    assert len(set(delays)) > 1, "identical delays would resynchronise the fan-out"


def test_backoff_saturates_at_the_cap() -> None:
    rng = Random(0)
    assert all(backoff_delay(20, cap=5.0, rng=rng) <= 5.0 for _ in range(20))


def test_concurrency_creeps_up_while_calls_succeed() -> None:
    aimd = AimdController(limit=4, probe_interval=3)
    for _ in range(3):
        aimd.on_success()
    assert aimd.limit == 5


def test_a_throttle_halves_concurrency_immediately() -> None:
    aimd = AimdController(limit=16)
    assert aimd.on_throttle(now=0.0) == 8
    assert aimd.on_throttle(now=0.0) == 4


def test_concurrency_never_drops_below_one() -> None:
    aimd = AimdController(limit=1)
    assert aimd.on_throttle(now=0.0) == 1


def test_cooldown_stops_the_probe_from_immediately_re_tripping() -> None:
    aimd = AimdController(limit=16, probe_interval=1, cooldown_s=5.0)
    aimd.on_throttle(now=100.0)
    for _ in range(10):
        aimd.on_success(now=101.0)
    assert aimd.limit == 8, "still cooling, so no increase"
    aimd.on_success(now=106.0)
    assert aimd.limit == 9


def test_retry_after_overrides_the_default_cooldown() -> None:
    aimd = AimdController(limit=8)
    aimd.on_throttle(now=10.0, retry_after=30.0)
    assert aimd.wait_for(now=10.0) == 30.0

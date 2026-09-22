from __future__ import annotations

import os
from pathlib import Path

import pytest

from jev_why.cache import (
    CacheMiss,
    MemoryCache,
    NullCache,
    SqliteCache,
    cache_key,
    resolve_cache,
)
from jev_why.types import Answer, JevResponse, Usage


def _response(model: str = "jev-1.13.0") -> JevResponse:
    return JevResponse(
        model,
        {
            "urgent": Answer(qtype="noul", noul=0.91),
            "mood": Answer(qtype="score", score=1.4, probabilities={0: 0.2, 1: 0.2, 2: 0.6}),
        },
        Usage(412, 0),
    )


def test_key_is_stable_across_question_declaration_order() -> None:
    """What lets two runs that declared the same panel differently share
    entries."""
    a = cache_key(model="m", state="s", questions={"a": {"t": 1}, "b": {"t": 2}})
    b = cache_key(model="m", state="s", questions={"b": {"t": 2}, "a": {"t": 1}})
    assert a == b


def test_key_changes_with_state_questions_and_model() -> None:
    base = cache_key(model="m", state="s", questions={"a": {"t": 1}})
    assert base != cache_key(model="m", state="other", questions={"a": {"t": 1}})
    assert base != cache_key(model="other", state="s", questions={"a": {"t": 1}})
    assert base != cache_key(model="m", state="s", questions={"a": {"t": 2}})


def test_sqlite_round_trip_preserves_integer_score_levels(tmp_path: Path) -> None:
    """Score probabilities are keyed by level index. Losing the integer keys
    would break the expected-level calculation on every cached read."""
    with SqliteCache(tmp_path) as cache:
        key = cache_key(model="m", state="s", questions={"q": {}})
        cache.put(key, _response())
        got = cache.get(key)

    assert got is not None
    assert list(got.answers["mood"].probabilities or {}) == [0, 1, 2]
    assert got.answers["urgent"].noul == pytest.approx(0.91)
    assert got.usage.input_tokens == 412


def test_cache_directory_is_owner_only(tmp_path: Path) -> None:
    """Entries hold raw states, which for tickets or diffs may be sensitive."""
    with SqliteCache(tmp_path):
        assert oct(os.stat(tmp_path).st_mode)[-3:] == "700"


def test_offline_cache_raises_rather_than_silently_calling(tmp_path: Path) -> None:
    with (
        SqliteCache(tmp_path, offline=True) as cache,
        pytest.raises(CacheMiss, match="offline"),
    ):
        cache.get("absent")


def test_pruning_reports_how_much_it_removed(tmp_path: Path) -> None:
    with SqliteCache(tmp_path) as cache:
        for i in range(5):
            cache.put(f"k{i}", _response())
        assert cache.count() == 5
        assert cache.prune(all_entries=True) == 5
        assert cache.count() == 0


def test_cache_records_which_model_answered(tmp_path: Path) -> None:
    """Needed to spot a mid-run rollout, which would contaminate every delta."""
    with SqliteCache(tmp_path) as cache:
        cache.put("a", _response("jev-1.13.0"))
        cache.put("b", _response("jev-1.14.0"))
        assert cache.models() == ["jev-1.13.0", "jev-1.14.0"]


def test_null_cache_never_stores_anything() -> None:
    cache = NullCache()
    cache.put("k", _response())
    assert cache.get("k") is None


def test_memory_cache_tracks_its_hit_rate() -> None:
    cache = MemoryCache()
    cache.put("k", _response())
    cache.get("k")
    cache.get("missing")
    assert cache.stats.hit_rate == pytest.approx(0.5)


def test_resolve_cache_accepts_none_a_path_or_an_instance(tmp_path: Path) -> None:
    assert isinstance(resolve_cache(None), NullCache)

    resolved = resolve_cache(str(tmp_path))
    assert isinstance(resolved, SqliteCache)
    resolved.close()

    existing = MemoryCache()
    assert resolve_cache(existing) is existing

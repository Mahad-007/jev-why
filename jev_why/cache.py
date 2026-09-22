"""Content-addressed response cache.

This module touches the filesystem, so it sits outside the pure core.

The cache is not an optimisation, it is the checkpoint. Each response is
written the instant it arrives, so a crash, a Ctrl-C or a budget trip at call
900 of 1000 loses nothing and the rerun pays for the remaining 100. It is also
what lets an occlusion run be upgraded to Shapley for the price of the
difference, and what lets the test suite exercise the whole network path
without a key.

Privacy: entries hold raw states. For support tickets, diffs or documents that
means content on disk that may be sensitive. The default directory is created
with owner-only permissions, NullCache exists for when nothing should be
written at all, and `jev-why cache prune` removes what is there.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jev_why.types import Answer, JevResponse, Usage

SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = "~/.cache/jev-why"


class CacheMiss(KeyError):
    """Raised instead of calling when the cache is running offline."""


def cache_key(*, model: str, state: Any, questions: Mapping[str, Any]) -> str:
    """A stable digest of everything that can change an answer.

    Questions arrive already sorted by name (see questions.panel_payload), so
    two runs that declared the same panel in a different order share entries.
    That assumes question order does not affect answers; `jev-why doctor`
    measures whether it does, and if it does this key has to grow an order
    component.
    """
    payload = {
        "schema": SCHEMA_VERSION,
        "model": model,
        "state": state,
        "questions": {name: questions[name] for name in sorted(questions)},
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class Cache(Protocol):
    def get(self, key: str) -> JevResponse | None: ...
    def put(self, key: str, response: JevResponse) -> None: ...
    @property
    def stats(self) -> CacheStats: ...


def _encode(response: JevResponse) -> str:
    return json.dumps(
        {
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "answers": {
                name: {
                    "qtype": a.qtype,
                    "noul": a.noul,
                    "choice": a.choice,
                    "score": a.score,
                    "probabilities": {str(k): v for k, v in (a.probabilities or {}).items()}
                    or None,
                    "confidence": a.confidence,
                    "legend": {str(k): v for k, v in (a.legend or {}).items()} or None,
                }
                for name, a in response.answers.items()
            },
        },
        separators=(",", ":"),
    )


def _decode(blob: str) -> JevResponse:
    raw = json.loads(blob)
    answers = {}
    for name, a in raw["answers"].items():
        probs = a.get("probabilities")
        if probs and a["qtype"] == "score":
            probs = {int(k): v for k, v in probs.items()}
        answers[name] = Answer(
            qtype=a["qtype"],
            noul=a.get("noul"),
            choice=a.get("choice"),
            score=a.get("score"),
            probabilities=probs,
            confidence=a.get("confidence"),
            legend=a.get("legend"),
        )
    usage = raw.get("usage") or {}
    return JevResponse(
        raw["model"],
        answers,
        Usage(usage.get("input_tokens") or 0, usage.get("output_tokens") or 0),
    )


@dataclass
class MemoryCache:
    """In-process only. The default for tests."""

    entries: dict[str, JevResponse] = field(default_factory=dict)
    _stats: CacheStats = field(default_factory=CacheStats)

    def get(self, key: str) -> JevResponse | None:
        hit = self.entries.get(key)
        self._stats = CacheStats(
            self._stats.hits + (1 if hit else 0),
            self._stats.misses + (0 if hit else 1),
            self._stats.writes,
        )
        return hit

    def put(self, key: str, response: JevResponse) -> None:
        self.entries[key] = response
        self._stats = CacheStats(self._stats.hits, self._stats.misses, self._stats.writes + 1)

    @property
    def stats(self) -> CacheStats:
        return self._stats


@dataclass
class NullCache:
    """Stores nothing. For states that must not be written to disk."""

    _stats: CacheStats = field(default_factory=CacheStats)

    def get(self, key: str) -> JevResponse | None:
        self._stats = CacheStats(self._stats.hits, self._stats.misses + 1, self._stats.writes)
        return None

    def put(self, key: str, response: JevResponse) -> None:
        return None

    @property
    def stats(self) -> CacheStats:
        return self._stats


class SqliteCache:
    """SQLite rather than a directory of JSON files.

    Atomic writes across concurrent async writers and across processes, no
    inode blowup at a hundred thousand entries, and `cache stats` becomes a
    query rather than a directory walk.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_CACHE_DIR,
        *,
        offline: bool = False,
        strict_model: bool = True,
    ) -> None:
        directory = Path(path).expanduser()
        if directory.suffix in (".sqlite", ".db"):
            self.path, directory = directory, directory.parent
        else:
            self.path = directory / "responses.sqlite"
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)

        self.offline = offline
        self.strict_model = strict_model
        self._stats = CacheStats()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, model TEXT NOT NULL, body TEXT NOT NULL,"
            " input_tokens INTEGER NOT NULL DEFAULT 0,"
            " created_at REAL NOT NULL DEFAULT (julianday('now')))"
        )
        self._conn.commit()

    def get(self, key: str) -> JevResponse | None:
        with self._lock:
            row = self._conn.execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            self._stats = CacheStats(self._stats.hits, self._stats.misses + 1, self._stats.writes)
            if self.offline:
                raise CacheMiss(
                    f"{key[:12]}... is not cached and the cache is offline. "
                    "Run once with a key to record it, or pass offline=False."
                )
            return None
        self._stats = CacheStats(self._stats.hits + 1, self._stats.misses, self._stats.writes)
        return _decode(row[0])

    def put(self, key: str, response: JevResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, body, input_tokens) "
                "VALUES (?, ?, ?, ?)",
                (key, response.model, _encode(response), response.usage.input_tokens),
            )
            self._conn.commit()
        self._stats = CacheStats(self._stats.hits, self._stats.misses, self._stats.writes + 1)

    def prune(self, *, all_entries: bool = False, older_than_days: float | None = None) -> int:
        with self._lock:
            if all_entries:
                cursor = self._conn.execute("DELETE FROM responses")
            elif older_than_days is not None:
                cursor = self._conn.execute(
                    "DELETE FROM responses WHERE created_at < julianday('now') - ?",
                    (older_than_days,),
                )
            else:
                return 0
            self._conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def models(self) -> list[str]:
        with self._lock:
            return [
                r[0]
                for r in self._conn.execute("SELECT DISTINCT model FROM responses ORDER BY model")
            ]

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()

    def __enter__(self) -> SqliteCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # A long-lived process that forgets to close should not leak a file
        # handle per cache it opened.
        self.close()

    @property
    def stats(self) -> CacheStats:
        return self._stats


def resolve_cache(cache: Cache | str | None, *, offline: bool = False) -> Cache:
    if cache is None:
        return NullCache()
    if isinstance(cache, str):
        return SqliteCache(cache, offline=offline)
    return cache


__all__ = [
    "DEFAULT_CACHE_DIR",
    "Cache",
    "CacheMiss",
    "CacheStats",
    "MemoryCache",
    "NullCache",
    "SqliteCache",
    "cache_key",
    "resolve_cache",
]

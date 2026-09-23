"""Reading decision logs off disk.

Touches the filesystem, so it sits outside the pure core. Parsing is kept
separate from the statistics so calibration and drift stay testable on plain
arrays.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LogError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionLog:
    probabilities: tuple[float, ...]
    labels: tuple[int, ...]
    """Empty when the log carries no outcomes. Prediction drift can be measured
    without them; calibration cannot."""

    source: str = ""

    @property
    def has_labels(self) -> bool:
        return len(self.labels) == len(self.probabilities) and bool(self.labels)

    def __len__(self) -> int:
        return len(self.probabilities)


def dotted(record: Any, path: str) -> Any:
    """Look up "answers.is_urgent.noul" in a nested record.

    A decision log is usually the API response written straight to disk, so the
    interesting number is nested rather than top level.
    """
    node = record
    for part in path.split("."):
        if isinstance(node, dict):
            if part not in node:
                raise LogError(f"field {path!r} not found: no key {part!r}")
            node = node[part]
        elif isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError) as error:
                raise LogError(f"field {path!r}: bad index {part!r}") from error
        else:
            raise LogError(f"field {path!r}: {part!r} is not addressable")
    return node


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix in (".csv", ".tsv"):
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise LogError(f"{path}:{number}: {error}") from error


def read_log(
    path: str | Path,
    *,
    prob_field: str,
    label_field: str | None = None,
    positive: str = "1",
) -> DecisionLog:
    """Read a .jsonl, .csv or .tsv decision log."""
    source = Path(path)
    if not source.exists():
        raise LogError(f"{source} does not exist")

    probabilities: list[float] = []
    labels: list[int] = []
    for record in _rows(source):
        raw = dotted(record, prob_field)
        try:
            probabilities.append(float(raw))
        except (TypeError, ValueError) as error:
            raise LogError(f"{prob_field!r} is not a number: {raw!r}") from error
        if label_field is not None:
            value = dotted(record, label_field)
            labels.append(_as_label(value, positive))

    if not probabilities:
        raise LogError(f"{source} contained no usable rows")
    return DecisionLog(tuple(probabilities), tuple(labels), str(source))


def _as_label(value: Any, positive: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0.5)
    text = str(value).strip().lower()
    if text in ("true", "yes", "y", "1", positive.lower()):
        return 1
    if text in ("false", "no", "n", "0"):
        return 0
    raise LogError(f"cannot read {value!r} as a label; pass --positive to name the positive value")


def write_log(
    path: str | Path,
    probabilities: Sequence[float],
    labels: Sequence[int] | None = None,
    *,
    field: str = "p",
) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for i, probability in enumerate(probabilities):
            row: dict[str, Any] = {field: float(probability)}
            if labels is not None:
                row["label"] = int(labels[i])
            handle.write(json.dumps(row) + "\n")


__all__ = ["DecisionLog", "LogError", "dotted", "read_log", "write_log"]

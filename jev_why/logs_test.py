from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_why.logs import LogError, dotted, read_log, write_log


def _jsonl(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_a_nested_field_is_addressable_by_a_dotted_path() -> None:
    """A decision log is usually the API response written straight to disk, so
    the number that matters is nested."""
    record = {"answers": {"is_urgent": {"noul": 0.91}}}
    assert dotted(record, "answers.is_urgent.noul") == 0.91


def test_a_list_index_works_in_a_path() -> None:
    assert dotted({"items": [{"p": 0.3}, {"p": 0.7}]}, "items.1.p") == 0.7


def test_a_missing_field_names_itself() -> None:
    with pytest.raises(LogError, match=r"answers\.missing"):
        dotted({"answers": {}}, "answers.missing")


def test_jsonl_with_labels_round_trips(tmp_path: Path) -> None:
    path = _jsonl(tmp_path, [{"p": 0.8, "label": 1}, {"p": 0.2, "label": 0}])
    log = read_log(path, prob_field="p", label_field="label")
    assert log.probabilities == (0.8, 0.2)
    assert log.labels == (1, 0)
    assert log.has_labels


def test_a_log_without_labels_is_usable_for_drift_but_says_so(tmp_path: Path) -> None:
    path = _jsonl(tmp_path, [{"p": 0.8}, {"p": 0.2}])
    log = read_log(path, prob_field="p")
    assert not log.has_labels
    assert len(log) == 2


def test_csv_is_read_as_well_as_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("p,label\n0.9,yes\n0.1,no\n", encoding="utf-8")
    log = read_log(path, prob_field="p", label_field="label")
    assert log.labels == (1, 0)


def test_boolean_and_textual_labels_are_both_understood(tmp_path: Path) -> None:
    path = _jsonl(tmp_path, [{"p": 0.9, "y": True}, {"p": 0.1, "y": "false"}])
    assert read_log(path, prob_field="p", label_field="y").labels == (1, 0)


def test_an_unreadable_label_asks_for_the_positive_value(tmp_path: Path) -> None:
    path = _jsonl(tmp_path, [{"p": 0.9, "y": "escalated"}])
    with pytest.raises(LogError, match="--positive"):
        read_log(path, prob_field="p", label_field="y")
    assert read_log(path, prob_field="p", label_field="y", positive="escalated").labels == (1,)


def test_a_malformed_line_reports_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"p": 0.5}\nnot json\n', encoding="utf-8")
    with pytest.raises(LogError, match=":2:"):
        read_log(path, prob_field="p")


def test_an_empty_log_is_an_error_rather_than_an_empty_result(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(LogError, match="no usable rows"):
        read_log(path, prob_field="p")


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(LogError, match="does not exist"):
        read_log(tmp_path / "nope.jsonl", prob_field="p")


def test_write_then_read_is_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    write_log(path, [0.1, 0.9], [0, 1])
    log = read_log(path, prob_field="p", label_field="label")
    assert log.probabilities == (0.1, 0.9)
    assert log.labels == (0, 1)

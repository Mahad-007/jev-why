from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_why.cli import EXIT_CHECK_FAILED, EXIT_ERROR, EXIT_OK, main
from jev_why.logs import write_log


def _labelled_log(tmp_path: Path, n: int = 800) -> Path:
    import numpy as np

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < p).astype(int)
    path = tmp_path / "decisions.jsonl"
    write_log(path, p.tolist(), y.tolist())
    return path


def test_calibrate_reports_and_writes_a_diagram(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _labelled_log(tmp_path)
    plot = tmp_path / "reliability.svg"
    assert (
        main(
            [
                "calibrate",
                "--log",
                str(log),
                "--prob-field",
                "p",
                "--label-field",
                "label",
                "--plot",
                str(plot),
            ]
        )
        == EXIT_OK
    )
    assert "decisions" in capsys.readouterr().out
    assert plot.read_text().startswith("<svg")


def test_calibrate_fails_the_gate_when_ece_is_too_high(tmp_path: Path) -> None:
    """Exit codes carry meaning so a gate is a one-liner in CI."""
    log = _labelled_log(tmp_path)
    assert (
        main(
            [
                "calibrate",
                "--log",
                str(log),
                "--prob-field",
                "p",
                "--label-field",
                "label",
                "--max-ece",
                "0.0001",
            ]
        )
        == EXIT_CHECK_FAILED
    )


def test_calibrate_without_outcomes_is_a_usage_error(tmp_path: Path) -> None:
    path = tmp_path / "unlabelled.jsonl"
    write_log(path, [0.1, 0.9])
    with pytest.raises(SystemExit):
        main(["calibrate", "--log", str(path), "--prob-field", "p"])


def test_threshold_prints_an_actionable_sentence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _labelled_log(tmp_path)
    assert (
        main(
            [
                "threshold",
                "--log",
                str(log),
                "--prob-field",
                "p",
                "--label-field",
                "label",
                "--target-precision",
                "0.8",
            ]
        )
        == EXIT_OK
    )
    assert "act automatically above" in capsys.readouterr().out


def test_an_unreachable_target_exits_with_the_check_code(tmp_path: Path) -> None:
    log = _labelled_log(tmp_path, n=200)
    assert (
        main(
            [
                "threshold",
                "--log",
                str(log),
                "--prob-field",
                "p",
                "--label-field",
                "label",
                "--target-precision",
                "0.9999",
            ]
        )
        == EXIT_CHECK_FAILED
    )


def test_selective_mode_reports_an_abstain_band(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _labelled_log(tmp_path)
    main(
        [
            "threshold",
            "--log",
            str(log),
            "--prob-field",
            "p",
            "--label-field",
            "label",
            "--selective",
            "--target-precision",
            "0.85",
            "--target-npv",
            "0.85",
        ]
    )
    assert "send" in capsys.readouterr().out


def test_drift_between_identical_logs_is_quiet(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    for name in ("base", "live"):
        write_log(tmp_path / f"{name}.jsonl", rng.beta(2, 5, 600).tolist())
    assert (
        main(
            [
                "drift",
                "--baseline",
                str(tmp_path / "base.jsonl"),
                "--live",
                str(tmp_path / "live.jsonl"),
                "--prob-field",
                "p",
                "--permutations",
                "100",
            ]
        )
        == EXIT_OK
    )


def test_drift_on_a_moved_distribution_exits_with_the_check_code(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    write_log(tmp_path / "base.jsonl", rng.beta(2, 5, 600).tolist())
    write_log(tmp_path / "live.jsonl", rng.beta(5, 2, 600).tolist())
    assert (
        main(
            [
                "drift",
                "--baseline",
                str(tmp_path / "base.jsonl"),
                "--live",
                str(tmp_path / "live.jsonl"),
                "--prob-field",
                "p",
                "--permutations",
                "100",
            ]
        )
        == EXIT_CHECK_FAILED
    )


def test_cache_stats_reports_an_empty_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cache", "stats", "--path", str(tmp_path)]) == EXIT_OK
    assert "0 entries" in capsys.readouterr().out


def test_errors_are_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "calibrate",
            "--log",
            str(tmp_path / "absent.jsonl"),
            "--prob-field",
            "p",
            "--label-field",
            "y",
        ]
    )
    assert code == EXIT_ERROR
    assert "LogError" in capsys.readouterr().err


def test_question_files_are_parsed_into_specs(tmp_path: Path) -> None:
    from jev_why.cli import _load_questions

    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            {
                "urgent": {"type": "noul", "instructions": "is it urgent?"},
                "dept": {
                    "type": "choice",
                    "instructions": "which team?",
                    "criteria": {"billing": None, "tech": None},
                },
                "mood": {
                    "type": "score",
                    "instructions": "how annoyed?",
                    "criteria": ["calm", "cross", "furious"],
                },
            }
        )
    )
    built = _load_questions(str(path))
    assert {q.qtype for q in built.values()} == {"noul", "choice", "score"}


def test_an_unknown_question_type_is_rejected(tmp_path: Path) -> None:
    from jev_why.cli import _load_questions

    path = tmp_path / "q.json"
    path.write_text(json.dumps({"x": {"type": "telepathy", "instructions": "hm"}}))
    with pytest.raises(SystemExit, match="unknown type"):
        _load_questions(str(path))

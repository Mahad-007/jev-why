"""Write measured figures into the README from a run's own output.

    python scripts/update_readme.py assets/run.json

Numbers in a README should never be typed by hand. This reads what the run
actually produced and rewrites the block between the MEASURED markers, so a
figure in the README is a figure something measured, and a stale one is a
merge conflict rather than a quiet lie.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- MEASURED:START -->", "<!-- MEASURED:END -->"


def _row(label: str, value: object, note: str = "") -> str:
    return f"| {label} | {value} | {note} |"


def render(run: dict[str, object]) -> str:
    lines = [
        f"Model `{run.get('model', 'unknown')}`, via the provider named in the run.",
        "",
        "| Measurement | Value | What it means |",
        "|---|---|---|",
    ]

    if "noise_sigma" in run:
        sigma = float(run["noise_sigma"])  # type: ignore[arg-type]
        lines.append(
            _row(
                "repeat spread",
                f"{sigma:.5f}",
                "identical calls agree exactly"
                if sigma == 0
                else f"attributions under {3 * sigma:.4f} are suppressed as noise",
            )
        )

    if "artifact_score" in run:
        score = float(run["artifact_score"])  # type: ignore[arg-type]
        lines.append(
            _row(
                "mask artifact",
                f"{score:.2f}",
                "redaction is close to invisible"
                if score < 0.5
                else "redaction is plainly visible to the model, so some of the signal is the mask",
            )
        )

    for key, label, note in (
        ("explain_spans", "spans explained", ""),
        ("explain_calls", "calls", "for the whole document"),
        ("completeness", "spans measured", "the rest had calls that never returned"),
        ("baseline", "p(injection), unmodified", ""),
        ("fully_redacted", "p(injection), all spans removed", ""),
    ):
        if key in run:
            lines.append(_row(label, run[key], note))

    if run.get("top_span_is_the_injection"):
        lines.append(
            _row(
                "top-ranked span",
                f"`{run.get('top_span')}` ({run.get('top_span_phi')})",
                "the injected sentence, recovered from a page of benign text",
            )
        )
    elif "top_span" in run:
        lines.append(
            _row(
                "top-ranked span",
                f"`{run.get('top_span')}` ({run.get('top_span_phi')})",
                "**not** the injected sentence",
            )
        )

    for key, label, note in (
        ("corpus_rows", "corpus rows scored", "unmodified test split"),
        ("auroc", "AUROC", "ranking quality"),
        ("ece", "ECE", "calibration error"),
        ("slope", "calibration slope", "below 1 is overconfident"),
        ("reads_as", "reads as", ""),
        ("threshold", "auto-act threshold", ""),
        ("threshold_coverage", "coverage at that threshold", ""),
        ("faithfulness_verdict", "faithfulness", "against a random control, cross-masked"),
        ("explain_cost_usd_estimated", "estimated cost", "this provider reports no usage"),
    ):
        if key in run:
            lines.append(_row(label, run[key], note))

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    source = Path(argv[1]) if len(argv) > 1 else ROOT / "assets" / "run.json"
    if not source.exists():
        print(f"{source} does not exist; run the example first", file=sys.stderr)
        return 1

    run = json.loads(source.read_text(encoding="utf-8"))
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README is missing the MEASURED markers", file=sys.stderr)
        return 1

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{render(run)}\n{END}{tail}", encoding="utf-8")
    print(f"updated {readme} from {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

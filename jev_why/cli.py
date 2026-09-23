"""Command line surface.

argparse rather than a CLI framework: one fewer dependency for a package whose
whole argument is that the expensive part should be cheap.

Exit codes carry meaning, so a gate can be a one-liner in CI:
  0 fine, 1 internal error, 2 bad usage, 3 a check failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_CHECK_FAILED = 0, 1, 2, 3


def _load_questions(path: str) -> dict[str, Any]:
    from jev_why.questions import Choice, Noul, QuestionSpec, Score

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    built: dict[str, QuestionSpec] = {}
    for name, spec in raw.items():
        kind = spec.get("type")
        if kind == "noul":
            built[name] = Noul(instructions=spec["instructions"], criteria=spec.get("criteria"))
        elif kind == "choice":
            built[name] = Choice(instructions=spec["instructions"], criteria=spec["criteria"])
        elif kind == "score":
            built[name] = Score(instructions=spec["instructions"], criteria=spec["criteria"])
        else:
            raise SystemExit(f"question {name!r}: unknown type {kind!r}")
    return built


def _read_state(args: argparse.Namespace) -> Any:
    if args.state_json:
        return json.loads(Path(args.state_json).read_text(encoding="utf-8"))
    return Path(args.state_file).read_text(encoding="utf-8")


def cmd_explain(args: argparse.Namespace) -> int:
    from jev_why.attribution import explain
    from jev_why.render import explanation_html, explanation_text

    explanation = explain(
        _read_state(args),
        _load_questions(args.questions),
        chunker=args.chunker,
        method=args.method,
        budget=args.budget,
        cache=args.cache,
        faithfulness=args.faithfulness,
        concurrency=args.concurrency,
    )

    if args.format == "json":
        payload = {
            name: {
                "baseline": q.baseline,
                "empty": q.empty,
                "link": q.link,
                "noise_sigma": q.noise_sigma,
                "efficiency_gap": q.efficiency_gap,
                "attributions": [
                    {
                        "span": a.span.label,
                        "text": a.span.text,
                        "phi": a.phi,
                        "necessity": a.necessity,
                        "sufficiency": a.sufficiency,
                        "magnitude": a.magnitude,
                        "significant": a.significant,
                    }
                    for a in q.attributions
                ],
            }
            for name, q in explanation.questions.items()
        }
        output = json.dumps(
            {
                "questions": payload,
                "warnings": list(explanation.warnings),
                "model": explanation.model_version,
                "cost_usd": explanation.spend.cost_usd,
            },
            indent=2,
        )
    elif args.format == "html":
        output = explanation_html(explanation)
    else:
        output = "\n\n".join(
            explanation_text(explanation, name, colour=sys.stdout.isatty())
            for name in explanation.questions
        )
        if explanation.warnings:
            output += "\n\n" + "\n".join(f"warning: {w}" for w in explanation.warnings)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output)

    if args.faithfulness and any(not r.credible for r in explanation.faithfulness.values()):
        return EXIT_CHECK_FAILED
    return EXIT_OK


def cmd_calibrate(args: argparse.Namespace) -> int:
    from jev_why import calibration
    from jev_why.logs import read_log
    from jev_why.render import reliability_svg

    log = read_log(args.log, prob_field=args.prob_field, label_field=args.label_field)
    if not log.has_labels:
        print("calibration needs outcomes; pass --label-field", file=sys.stderr)
        return EXIT_USAGE

    report = calibration.report(log.probabilities, log.labels, bins=args.bins)
    print(report.summary())
    if args.plot:
        Path(args.plot).write_text(reliability_svg(report), encoding="utf-8")
        print(f"wrote {args.plot}")
    if args.max_ece is not None and report.ece > args.max_ece:
        print(f"ECE {report.ece:.4f} exceeds {args.max_ece}", file=sys.stderr)
        return EXIT_CHECK_FAILED
    return EXIT_OK


def cmd_threshold(args: argparse.Namespace) -> int:
    from jev_why.logs import read_log
    from jev_why.thresholds import selective_thresholds, threshold_for_precision

    log = read_log(args.log, prob_field=args.prob_field, label_field=args.label_field)
    if not log.has_labels:
        print("threshold solving needs outcomes; pass --label-field", file=sys.stderr)
        return EXIT_USAGE

    if args.selective:
        print(
            selective_thresholds(
                log.probabilities,
                log.labels,
                target_precision=args.target_precision,
                target_npv=args.target_npv,
                alpha=args.alpha,
            ).explain()
        )
        return EXIT_OK

    result = threshold_for_precision(
        log.probabilities, log.labels, target=args.target_precision, alpha=args.alpha
    )
    print(result.explain())
    return EXIT_OK if result.achieved else EXIT_CHECK_FAILED


def cmd_drift(args: argparse.Namespace) -> int:
    from jev_why.drift import drift_report
    from jev_why.logs import read_log

    baseline = read_log(args.baseline, prob_field=args.prob_field)
    live = read_log(args.live, prob_field=args.prob_field)
    report = drift_report(
        baseline.probabilities,
        live.probabilities,
        observed_positives=args.observed_positives,
        permutations=args.permutations,
    )
    print(report.summary())
    return EXIT_CHECK_FAILED if (report.prediction_drift or report.calibration_drift) else EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    from jev_why.client import TypeSafeJevClient
    from jev_why.doctor import run_doctor

    async def go() -> int:
        client = TypeSafeJevClient(model=args.model)
        try:
            report = await run_doctor(client, model=args.model, repeats=args.repeats)
        finally:
            await client.aclose()
        print(report.render())
        return EXIT_OK if report.all_passed else EXIT_CHECK_FAILED

    return asyncio.run(go())


def cmd_cache(args: argparse.Namespace) -> int:
    from jev_why.cache import DEFAULT_CACHE_DIR, SqliteCache

    with SqliteCache(args.path or DEFAULT_CACHE_DIR) as cache:
        if args.action == "stats":
            print(f"{cache.count()} entries at {cache.path}")
            print(f"models: {', '.join(cache.models()) or 'none'}")
        elif args.action == "prune":
            removed = cache.prune(all_entries=args.all, older_than_days=args.older_than)
            print(f"removed {removed} entries")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-why",
        description="Causal attribution and calibration for TypeSafe Jev decisions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    explain = sub.add_parser("explain", help="why did Jev decide that")
    source = explain.add_mutually_exclusive_group(required=True)
    source.add_argument("--state-file", help="a text state")
    source.add_argument("--state-json", help="a JSON state")
    explain.add_argument("--questions", required=True, help="JSON file of question specs")
    explain.add_argument(
        "--chunker", default="auto", choices=["auto", "sentences", "paragraphs", "lines", "json"]
    )
    explain.add_argument("--method", default="occlusion", choices=["occlusion", "shapley"])
    explain.add_argument("--budget", type=float, default=0.25, help="ceiling in USD")
    explain.add_argument("--cache", default=None, help="cache directory")
    explain.add_argument("--concurrency", type=int, default=16)
    explain.add_argument(
        "--faithfulness",
        action="store_true",
        help="also measure whether the explanation is faithful",
    )
    explain.add_argument("--format", default="text", choices=["text", "json", "html"])
    explain.add_argument("-o", "--output")
    explain.set_defaults(func=cmd_explain)

    calibrate = sub.add_parser("calibrate", help="do the probabilities mean what they say")
    calibrate.add_argument("--log", required=True)
    calibrate.add_argument("--prob-field", required=True)
    calibrate.add_argument("--label-field", required=True)
    calibrate.add_argument("--bins", type=int, default=10)
    calibrate.add_argument("--plot", help="write a reliability diagram here")
    calibrate.add_argument("--max-ece", type=float, help="fail if ECE exceeds this")
    calibrate.set_defaults(func=cmd_calibrate)

    threshold = sub.add_parser("threshold", help="where to draw the auto-act line")
    threshold.add_argument("--log", required=True)
    threshold.add_argument("--prob-field", required=True)
    threshold.add_argument("--label-field", required=True)
    threshold.add_argument("--target-precision", type=float, default=0.9)
    threshold.add_argument("--target-npv", type=float, default=0.95)
    threshold.add_argument("--alpha", type=float, default=0.05)
    threshold.add_argument(
        "--selective", action="store_true", help="solve an accept/reject pair with an abstain band"
    )
    threshold.set_defaults(func=cmd_threshold)

    drift = sub.add_parser("drift", help="has the distribution moved")
    drift.add_argument("--baseline", required=True)
    drift.add_argument("--live", required=True)
    drift.add_argument("--prob-field", required=True)
    drift.add_argument(
        "--observed-positives", type=int, help="how many live cases actually turned out positive"
    )
    drift.add_argument("--permutations", type=int, default=2000)
    drift.set_defaults(func=cmd_drift)

    doctor = sub.add_parser("doctor", help="measure this library's assumptions live")
    doctor.add_argument("--model", default=os.environ.get("JEV_WHY_MODEL", "jev-latest"))
    doctor.add_argument("--repeats", type=int, default=5)
    doctor.set_defaults(func=cmd_doctor)

    cache = sub.add_parser("cache", help="inspect or clear the response cache")
    cache.add_argument("action", choices=["stats", "prune"])
    cache.add_argument("--path")
    cache.add_argument("--all", action="store_true")
    cache.add_argument("--older-than", type=float, metavar="DAYS")
    cache.set_defaults(func=cmd_cache)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted; cached responses are kept", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

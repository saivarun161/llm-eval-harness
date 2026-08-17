"""Command line interface.

``evalharness demo`` is the entry point worth reading first: it runs five
simulated systems against the bundled dataset and walks through every claim the
project makes — that a visible improvement can be noise, that a formatting
change can look like a regression to the wrong scorer, that a judge has a
measurable error rate, and that a gate should refuse to fire on any of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from . import report
from .calibration import calibrate, load_labels
from .compare import DatasetMismatch, compare_runs
from .dataset import Dataset, list_builtins, load_dataset
from .gate import evaluate_gate
from .runner import evaluate, summarize
from .scorers import DEFAULT_SCORERS, available, describe
from .store import load_run, save_run
from .targets import PROFILES, resolve_target
from .types import RunResult

DEFAULT_DATASET = "builtin:qa_general"
DEFAULT_LABELS = "builtin:judge_labels"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _split_scorers(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SCORERS)
    return [name.strip() for name in raw.split(",") if name.strip()]


def _run_target(dataset: Dataset, target_name: str, scorers: Sequence[str]) -> RunResult:
    target = resolve_target(target_name, dataset.cases)
    return evaluate(
        dataset,
        target,
        scorers=scorers,
        target_name=target_name,
        label=target_name,
        created_at=_now(),
    )


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_scorers(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps({name: describe(name) for name in available()}, indent=2))
        return 0
    print("Registered scorers")
    print(report.RULE)
    for name in available():
        print(f"  {name:<16}{describe(name)}")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    rows = []
    for name in list_builtins():
        dataset = load_dataset(f"builtin:{name}")
        rows.append(
            {
                "name": dataset.name,
                "version": dataset.version,
                "fingerprint": dataset.fingerprint,
                "cases": len(dataset),
                "tags": sorted(dataset.tag_index()),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print("Bundled datasets")
    print(report.RULE)
    for row in rows:
        print(f"  {row['name']} v{row['version']}  {row['fingerprint']}  {row['cases']} cases")
        print(f"    tags: {', '.join(row['tags'])}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    if args.tag:
        dataset = dataset.filter_by_tag(args.tag)
    scorers = _split_scorers(args.scorers)
    run = _run_target(dataset, args.target, scorers)

    if args.out:
        save_run(run, args.out)

    summaries = summarize(run, resamples=args.resamples)
    if args.json:
        print(
            json.dumps(
                {
                    "run": report.render_run_header(run),
                    "summaries": [s.to_json() for s in summaries],
                },
                indent=2,
            )
        )
        return 0

    print(report.render_run_header(run))
    print()
    print(report.render_summaries(summaries, title=f"Scores for {run.name}"))
    if args.failures:
        print()
        print(report.render_failures(run, scorers[-1]))
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    scorers = _split_scorers(args.scorers) if args.scorers else list(candidate.scorers())
    try:
        comparisons = [
            compare_runs(
                baseline,
                candidate,
                scorer,
                resamples=args.resamples,
                allow_dataset_drift=args.allow_dataset_drift,
            )
            for scorer in scorers
        ]
    except DatasetMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([c.to_json() for c in comparisons], indent=2))
        return 0
    for comparison in comparisons:
        print(report.render_comparison(comparison))
        print()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    examples = load_labels(args.labels)
    thresholds = None
    if args.thresholds:
        low, _, high = args.thresholds.partition(",")
        thresholds = (float(low), float(high))
    calibration = calibrate(examples, thresholds=thresholds, search=not args.no_search)
    if args.json:
        print(json.dumps(calibration.to_json(), indent=2))
    else:
        print(report.render_calibration(calibration))
    return 0 if calibration.kappa >= args.min_kappa else 1


def cmd_gate(args: argparse.Namespace) -> int:
    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    try:
        result = evaluate_gate(
            baseline,
            candidate,
            args.scorer,
            tolerance=args.tolerance,
            mode=args.mode,
            required_mde=args.require_mde,
            resamples=args.resamples,
            allow_dataset_drift=args.allow_dataset_drift,
        )
    except (DatasetMismatch, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_json(), indent=2))
    else:
        print(report.render_gate(result))
    return result.exit_code


def cmd_demo(args: argparse.Namespace) -> int:
    resamples = args.resamples
    dataset = load_dataset(DEFAULT_DATASET)
    scorers = ["exact_match", "token_f1", "semantic", "judge"]

    print(
        f"Dataset {dataset.name} v{dataset.version} ({dataset.fingerprint}), "
        f"{len(dataset)} cases, no API keys involved."
    )
    print()

    runs = {name: _run_target(dataset, name, scorers) for name in PROFILES}

    print("1. Four scorers, one system. They do not agree, and that is the point.")
    print(report.RULE)
    print(report.render_summaries(summarize(runs["baseline"], resamples=resamples)))
    print()
    print("   Exact match is the harshest reader in the room: it fails every correct")
    print("   answer that happens to be worded differently. Pick your headline metric")
    print("   carelessly and you will optimise for phrasing.")
    print()

    print("2. A real improvement, and a change that only looks like one.")
    print(report.RULE)
    for name in ("candidate", "tweaked"):
        print(
            report.render_comparison(
                compare_runs(runs["baseline"], runs[name], "judge", resamples=resamples)
            )
        )
        print()
    print("   Both systems scored above the baseline. Only one of them beat the noise.")
    print("   A dashboard showing two point estimates would have called them both wins.")
    print()

    print("3. The same answers, more words. Watch the scorers disagree.")
    print(report.RULE)
    for scorer in ("exact_match", "judge"):
        print(
            report.render_comparison(
                compare_runs(runs["baseline"], runs["verbose"], scorer, resamples=resamples)
            )
        )
        print()
    print("   The 'verbose' system gets exactly the same answers right as the baseline")
    print("   and wraps them in a sentence. Exact match calls that a 47-point collapse.")
    print("   The judge calls it a 7-point style penalty. Same outputs, two stories, and")
    print("   only one of them is about quality.")
    print()

    print("4. How much is the judge worth? Measure it against humans.")
    print(report.RULE)
    calibration = calibrate(load_labels(DEFAULT_LABELS))
    print(report.render_calibration(calibration))
    print()

    print("5. The CI gate, on a real regression and on a harmless change.")
    print(report.RULE)
    for name in ("regressed", "tweaked"):
        result = evaluate_gate(
            runs["baseline"],
            runs[name],
            "judge",
            tolerance=0.02,
            mode="confident",
            resamples=resamples,
        )
        print(report.render_gate(result))
        print()
    print("   The gate blocks the regression and stays quiet about the noise. A gate")
    print("   that fired on both would be switched off within a fortnight, and then")
    print("   the next real regression would ship.")
    return 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalharness",
        description="Evaluate LLM applications with confidence intervals attached.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="run the full guided walkthrough (no API keys)")
    demo.add_argument("--resamples", type=int, default=2000)
    demo.set_defaults(func=cmd_demo)

    scorers_cmd = sub.add_parser("scorers", help="list registered scorers")
    scorers_cmd.add_argument("--json", action="store_true")
    scorers_cmd.set_defaults(func=cmd_scorers)

    datasets_cmd = sub.add_parser("datasets", help="list bundled datasets and their fingerprints")
    datasets_cmd.add_argument("--json", action="store_true")
    datasets_cmd.set_defaults(func=cmd_datasets)

    run_cmd = sub.add_parser("run", help="evaluate a target against a dataset")
    run_cmd.add_argument("--target", required=True, help="built-in profile or module:function")
    run_cmd.add_argument("--dataset", default=DEFAULT_DATASET)
    run_cmd.add_argument(
        "--scorers", help=f"comma separated (default: {','.join(DEFAULT_SCORERS)})"
    )
    run_cmd.add_argument("--tag", help="restrict to cases carrying this tag")
    run_cmd.add_argument("--out", help="write the run to this path")
    run_cmd.add_argument("--failures", action="store_true", help="show the worst cases")
    run_cmd.add_argument("--resamples", type=int, default=2000)
    run_cmd.add_argument("--json", action="store_true")
    run_cmd.set_defaults(func=cmd_run)

    compare_cmd = sub.add_parser("compare", help="paired A/B comparison of two saved runs")
    compare_cmd.add_argument("baseline")
    compare_cmd.add_argument("candidate")
    compare_cmd.add_argument("--scorers", help="comma separated (default: every shared scorer)")
    compare_cmd.add_argument("--resamples", type=int, default=2000)
    compare_cmd.add_argument("--allow-dataset-drift", action="store_true")
    compare_cmd.add_argument("--json", action="store_true")
    compare_cmd.set_defaults(func=cmd_compare)

    calibrate_cmd = sub.add_parser("calibrate", help="measure the judge against human labels")
    calibrate_cmd.add_argument("--labels", default=DEFAULT_LABELS)
    calibrate_cmd.add_argument("--thresholds", help="fixed cut points, e.g. 0.42,0.72")
    calibrate_cmd.add_argument("--no-search", action="store_true", help="skip the threshold search")
    calibrate_cmd.add_argument(
        "--min-kappa",
        type=float,
        default=0.0,
        help="exit non-zero when agreement falls below this",
    )
    calibrate_cmd.add_argument("--json", action="store_true")
    calibrate_cmd.set_defaults(func=cmd_calibrate)

    gate_cmd = sub.add_parser("gate", help="fail CI when quality regresses beyond the noise")
    gate_cmd.add_argument("--baseline", required=True)
    gate_cmd.add_argument("--candidate", required=True)
    gate_cmd.add_argument("--scorer", default="judge")
    gate_cmd.add_argument("--tolerance", type=float, default=0.0)
    gate_cmd.add_argument("--mode", choices=("confident", "cautious"), default="confident")
    gate_cmd.add_argument(
        "--require-mde",
        type=float,
        default=None,
        help="fail when the eval set cannot resolve an effect this small",
    )
    gate_cmd.add_argument("--resamples", type=int, default=2000)
    gate_cmd.add_argument("--allow-dataset-drift", action="store_true")
    gate_cmd.add_argument("--json", action="store_true")
    gate_cmd.set_defaults(func=cmd_gate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except (FileNotFoundError, KeyError, ValueError, ImportError, AttributeError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

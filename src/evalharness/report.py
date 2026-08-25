"""Human-readable rendering of runs, comparisons, calibrations and gates.

Reports are plain text with fixed-width columns so they read the same in a
terminal, in a CI log and in a pull-request comment. Every aggregate is printed
with its interval attached; a bare mean never appears anywhere in this module,
which is the one formatting rule the project actually cares about.
"""

from __future__ import annotations

from .calibration import LABEL_NAMES, Calibration
from .compare import Comparison
from .gate import GateResult
from .runner import ScoreSummary
from .slices import SliceReport
from .types import RunResult

RULE = "=" * 78
THIN = "-" * 78


def _bar(value: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return "#" * filled + "." * (width - filled)


def render_summaries(summaries: list[ScoreSummary], *, title: str = "") -> str:
    lines: list[str] = []
    if title:
        lines += [title, RULE]
    lines.append(f"{'scorer':<16}{'n':>5}  {'mean':>7}  {'95% interval':>18}  distribution")
    lines.append(THIN)
    for summary in summaries:
        interval = summary.interval
        lines.append(
            f"{summary.scorer:<16}{summary.n:>5}  {interval.point:>7.3f}  "
            f"[{interval.lo:>6.3f}, {interval.hi:>6.3f}]  {_bar(interval.point)}"
        )
    return "\n".join(lines)


def render_tag_breakdown(breakdown: dict[str, ScoreSummary], *, scorer: str) -> str:
    if not breakdown:
        return ""
    lines = [f"Per-tag {scorer}", RULE]
    lines.append(f"{'tag':<16}{'n':>5}  {'mean':>7}  {'95% interval':>18}")
    lines.append(THIN)
    for tag, summary in breakdown.items():
        interval = summary.interval
        lines.append(
            f"{tag:<16}{summary.n:>5}  {interval.point:>7.3f}  "
            f"[{interval.lo:>6.3f}, {interval.hi:>6.3f}]"
        )
    return "\n".join(lines)


def render_comparison(comparison: Comparison) -> str:
    verdict = {
        "improvement": "IMPROVED",
        "regression": "REGRESSED",
        "inconclusive": "INCONCLUSIVE",
    }[comparison.direction]
    lines = [
        f"{comparison.candidate_name} vs {comparison.baseline_name} on '{comparison.scorer}'",
        RULE,
        f"  baseline   {comparison.baseline_mean:.3f}",
        f"  candidate  {comparison.candidate_mean:.3f}",
        f"  delta      {comparison.delta.format()}   <- 95% CI, paired over {comparison.n} cases",
        f"  verdict    {verdict}",
        f"  wins/losses/ties  {comparison.wins}/{comparison.losses}/{comparison.ties}"
        f"   sign test p={comparison.sign_test_p:.4f}",
        f"  smallest effect this eval set could resolve: ±{comparison.min_detectable_effect:.3f}",
    ]
    if not comparison.significant:
        lines.append(
            "  The interval straddles zero: this difference is not distinguishable "
            "from resampling noise."
        )
    return "\n".join(lines)


def render_slice_report(report: SliceReport) -> str:
    """The per-tag table, with the aggregate above it for contrast."""
    verdict = {"improvement": "IMPROVED", "regression": "REGRESSED", "inconclusive": "FLAT"}
    aggregate = report.aggregate
    lines = [
        f"{aggregate.candidate_name} vs {aggregate.baseline_name} on "
        f"'{report.scorer}', sliced by tag",
        RULE,
        f"  whole set  n={aggregate.n:<4} delta {aggregate.delta.format()}  "
        f"{verdict[aggregate.direction]}",
        "",
    ]
    if not report.slices:
        lines.append("  No slice is large enough to test.")
    else:
        level = f"{report.per_slice_confidence:.2%}".rstrip("0").rstrip(".")
        lines.append(
            f"{'tag':<14}{'n':>4} {'share':>7}  {'delta':>7}  "
            f"{'interval':>18}  {'holm p':>8}  verdict"
        )
        lines.append(THIN)
        for item in report.slices:
            delta = item.comparison.delta
            lines.append(
                f"{item.tag:<14}{item.n:>4} {item.share:>6.0%}  {delta.point:>7.3f}  "
                f"[{delta.lo:>6.3f}, {delta.hi:>6.3f}]  {item.adjusted_p:>8.4f}  "
                f"{verdict[item.direction]}"
            )
        lines.append("")
        lines.append(
            f"  Intervals are {level} per slice, so that all {report.family_size} "
            f"together hold {report.confidence:.0%}."
        )

    if report.skipped:
        untested = ", ".join(f"{tag} (n={n})" for tag, n in report.skipped)
        lines.append(
            f"  Untested, under {report.min_cases} cases: {untested}. "
            "Too small to resolve anything, not known to be fine."
        )

    shaky = report.uncorroborated
    if shaky:
        names = ", ".join(f"'{item.tag}'" for item in shaky)
        lines.append(
            f"  {names}: the interval fires, the adjusted sign test does not. "
            "Re-run wider before acting."
        )

    hidden = report.hidden_regressions
    if hidden:
        names = ", ".join(f"'{item.tag}'" for item in hidden)
        lines.append("")
        lines.append(
            f"  The aggregate is not a regression, but {names} is. "
            "A headline mean would have shipped this."
        )
    return "\n".join(lines)


def render_calibration(calibration: Calibration) -> str:
    lines = [
        f"Judge calibration ({calibration.judge_name} judge, {calibration.n} human labels)",
        RULE,
        f"  cut points        partial >= {calibration.thresholds[0]:.2f}, "
        f"correct >= {calibration.thresholds[1]:.2f}",
        f"  Cohen's kappa     {calibration.kappa:.3f} "
        f"[{calibration.kappa_interval.lo:.3f}, {calibration.kappa_interval.hi:.3f}] "
        f"linear-weighted, {calibration.interpretation}",
        f"  unweighted kappa  {calibration.kappa_nominal:.3f}",
        f"  at shipped cuts   {calibration.kappa_at_default:.3f}  "
        f"(a gap here means your labels disagree with the shipped calibration)",
        f"  raw agreement     {calibration.agreement:.1%}",
        f"  correlation       pearson {calibration.pearson:.3f}, "
        f"spearman {calibration.spearman:.3f}",
        "",
        "  confusion (rows: human, columns: judge)",
        f"    {'':<10}" + "".join(f"{name:>10}" for name in LABEL_NAMES),
    ]
    for i, name in enumerate(LABEL_NAMES):
        row = "".join(f"{value:>10}" for value in calibration.confusion[i])
        lines.append(f"    {name:<10}{row}")
    lines.append("")
    if calibration.trustworthy:
        lines.append("  Substantial agreement: sound enough to gate regressions on.")
    else:
        lines.append(
            "  Below the 0.60 substantial-agreement line. Usable for triage and "
            "trend-watching; do not gate a release on this judge alone."
        )
    return "\n".join(lines)


def render_gate(result: GateResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"Regression gate: {status}",
        RULE,
        render_comparison(result.comparison),
        "",
        f"  mode {result.mode}, tolerance {result.tolerance:.3f}",
    ]
    lines += [f"  - {reason}" for reason in result.reasons]
    return "\n".join(lines)


def render_run_header(run: RunResult) -> str:
    return (
        f"{run.name}  target={run.target}  dataset={run.dataset_name} "
        f"v{run.dataset_version} ({run.dataset_fingerprint}) "
        f"cases={len(run.case_ids)}"
    )


def render_failures(run: RunResult, scorer: str, *, limit: int = 5) -> str:
    """The worst-scoring cases, because an aggregate never explains itself."""
    outputs = {p.case_id: p.output for p in run.predictions}
    scored = sorted(
        (s for s in run.scores if s.scorer == scorer),
        key=lambda s: s.score,
    )[:limit]
    if not scored:
        return ""
    lines = [f"Lowest-scoring cases by '{scorer}'", RULE]
    for score in scored:
        answer = outputs.get(score.case_id, "")
        if len(answer) > 68:
            answer = answer[:67] + "…"
        lines.append(f"  {score.score:>5.2f}  {score.case_id:<12} {answer}")
        if score.detail:
            detail = score.detail if len(score.detail) <= 68 else score.detail[:67] + "…"
            lines.append(f"         {'':<12} {detail}")
    return "\n".join(lines)

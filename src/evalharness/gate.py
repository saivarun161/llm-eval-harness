"""The CI regression gate.

A gate that fires on a raw score drop is a gate that fires on noise, and a gate
that fires on noise gets disabled within a fortnight. So this one fires on
evidence: it fails when the *confidence interval* for the paired delta says a
regression larger than the tolerance really happened.

Two modes, because the right answer depends on what is downstream:

``confident`` (default)
    Fail only when the entire delta interval lies below ``-tolerance``. Quiet,
    trustworthy, and will miss a genuine regression it cannot yet resolve.

``cautious``
    Fail as soon as the interval's lower edge crosses ``-tolerance`` — i.e.
    whenever a regression that size cannot be ruled out. Noisier, appropriate
    when shipping a regression is far more expensive than a re-run.

Both modes are blind if the eval set is too small to resolve the tolerance in
the first place. That failure is silent by construction and is the one people
actually get bitten by, so the gate measures its own resolving power and can be
told to fail when it is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compare import Comparison, compare_runs
from .stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED
from .types import RunResult

MODES = ("confident", "cautious")


@dataclass(frozen=True)
class GateResult:
    """Pass/fail plus every number that went into the decision."""

    passed: bool
    mode: str
    tolerance: float
    comparison: Comparison
    reasons: list[str] = field(default_factory=list)
    underpowered: bool = False
    required_mde: float | None = None

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def to_json(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "mode": self.mode,
            "tolerance": self.tolerance,
            "underpowered": self.underpowered,
            "required_mde": self.required_mde,
            "reasons": list(self.reasons),
            "comparison": self.comparison.to_json(),
        }


def evaluate_gate(
    baseline: RunResult,
    candidate: RunResult,
    scorer: str,
    *,
    tolerance: float = 0.0,
    mode: str = "confident",
    required_mde: float | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    allow_dataset_drift: bool = False,
) -> GateResult:
    """Decide whether ``candidate`` may ship, given ``baseline``."""
    if mode not in MODES:
        raise ValueError(f"gate mode must be one of {MODES}, got {mode!r}")
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance!r}")

    comparison = compare_runs(
        baseline,
        candidate,
        scorer,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        allow_dataset_drift=allow_dataset_drift,
    )

    threshold = -tolerance
    edge = comparison.delta.hi if mode == "confident" else comparison.delta.lo
    regressed = edge < threshold

    reasons: list[str] = []
    if regressed:
        bound = "upper" if mode == "confident" else "lower"
        reasons.append(
            f"{scorer} regressed: delta {comparison.delta.format()} — the {bound} bound "
            f"of the {int(confidence * 100)}% interval is below the {tolerance:.3f} tolerance"
        )
    else:
        reasons.append(
            f"{scorer} within tolerance: delta {comparison.delta.format()} "
            f"(tolerance {tolerance:.3f}, mode {mode})"
        )

    underpowered = False
    if required_mde is not None:
        underpowered = comparison.min_detectable_effect > required_mde
        if underpowered:
            reasons.append(
                f"eval set is underpowered: it can only resolve effects of "
                f"±{comparison.min_detectable_effect:.3f}, and {required_mde:.3f} was required. "
                f"A passing gate here means 'we could not tell', not 'nothing broke'. "
                f"Add cases or raise --require-mde."
            )

    return GateResult(
        passed=not regressed and not underpowered,
        mode=mode,
        tolerance=tolerance,
        comparison=comparison,
        reasons=reasons,
        underpowered=underpowered,
        required_mde=required_mde,
    )

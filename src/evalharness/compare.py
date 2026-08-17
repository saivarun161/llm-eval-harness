"""Pairwise A/B comparison of two runs.

The question is never "which mean is bigger" — with fifty cases and a graded
scorer, one system beats the other by a point or two almost every time, in
whichever direction the noise happens to fall. The question is whether the
difference survives resampling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Interval,
    binomial_two_sided_p,
    bootstrap_paired_delta,
    mean,
)
from .types import RunResult


class DatasetMismatch(ValueError):
    """Raised when two runs were not measured against the same cases."""


@dataclass(frozen=True)
class Comparison:
    """The result of comparing a candidate run against a baseline run."""

    scorer: str
    baseline_name: str
    candidate_name: str
    n: int
    baseline_mean: float
    candidate_mean: float
    delta: Interval
    wins: int
    losses: int
    ties: int
    sign_test_p: float

    @property
    def significant(self) -> bool:
        """True when the delta interval sits entirely on one side of zero."""
        return self.delta.excludes_zero

    @property
    def direction(self) -> str:
        if not self.significant:
            return "inconclusive"
        return "improvement" if self.delta.point > 0 else "regression"

    @property
    def min_detectable_effect(self) -> float:
        """The smallest delta this dataset could have resolved.

        Reported next to every comparison because "no significant difference"
        means something very different at ±0.01 than it does at ±0.15, and the
        second case is a statement about the eval set, not about the systems.
        """
        return self.delta.half_width

    def to_json(self) -> dict[str, object]:
        return {
            "scorer": self.scorer,
            "baseline": {"name": self.baseline_name, "mean": round(self.baseline_mean, 6)},
            "candidate": {"name": self.candidate_name, "mean": round(self.candidate_mean, 6)},
            "n": self.n,
            "delta": self.delta.to_json(),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "sign_test_p": round(self.sign_test_p, 6),
            "significant": self.significant,
            "direction": self.direction,
            "min_detectable_effect": round(self.min_detectable_effect, 6),
        }


def check_comparable(
    baseline: RunResult,
    candidate: RunResult,
    *,
    allow_dataset_drift: bool = False,
) -> None:
    """Refuse to compare runs that did not see the same cases in the same order."""
    if baseline.case_ids != candidate.case_ids:
        raise DatasetMismatch(
            f"runs cover different cases ({len(baseline.case_ids)} vs "
            f"{len(candidate.case_ids)}); a paired comparison needs the same cases"
        )
    if baseline.dataset_fingerprint != candidate.dataset_fingerprint and not allow_dataset_drift:
        raise DatasetMismatch(
            f"dataset fingerprint changed between runs "
            f"({baseline.dataset_fingerprint} -> {candidate.dataset_fingerprint}). "
            "The delta would partly measure the dataset edit. Re-run the baseline, "
            "or pass --allow-dataset-drift if you know what you are doing."
        )


def compare_runs(
    baseline: RunResult,
    candidate: RunResult,
    scorer: str,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    tie_epsilon: float = 1e-9,
    allow_dataset_drift: bool = False,
) -> Comparison:
    """Paired comparison of ``candidate`` against ``baseline`` on one scorer."""
    check_comparable(baseline, candidate, allow_dataset_drift=allow_dataset_drift)
    base_values = baseline.vector(scorer)
    cand_values = candidate.vector(scorer)

    wins = losses = ties = 0
    for c, b in zip(cand_values, base_values, strict=True):
        if abs(c - b) <= tie_epsilon:
            ties += 1
        elif c > b:
            wins += 1
        else:
            losses += 1

    return Comparison(
        scorer=scorer,
        baseline_name=baseline.name,
        candidate_name=candidate.name,
        n=len(base_values),
        baseline_mean=mean(base_values),
        candidate_mean=mean(cand_values),
        delta=bootstrap_paired_delta(
            cand_values, base_values, resamples=resamples, confidence=confidence, seed=seed
        ),
        wins=wins,
        losses=losses,
        ties=ties,
        # Ties carry no directional information, so the sign test conditions on
        # the decided cases only — the standard treatment.
        sign_test_p=binomial_two_sided_p(wins, wins + losses),
    )


def compare_all(
    baseline: RunResult,
    candidate: RunResult,
    *,
    scorers: list[str] | None = None,
    **kwargs: object,
) -> list[Comparison]:
    """Compare on every scorer both runs share."""
    names = scorers or [s for s in candidate.scorers() if s in set(baseline.scorers())]
    if not names:
        raise ValueError("the two runs have no scorer in common")
    return [compare_runs(baseline, candidate, name, **kwargs) for name in names]  # type: ignore[arg-type]

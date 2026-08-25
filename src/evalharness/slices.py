"""Per-slice paired comparison: which *part* of the eval set moved.

An aggregate delta answers "did quality change". It cannot answer "for whom",
and the two questions come apart more often than anyone would like. A prompt
rewrite that helps every prose answer and breaks every unit conversion moves
the headline number by nothing at all, and ships. The slice that broke is
usually the slice somebody cares about most.

Slicing is not free, though: eight slices tested at 95% carry roughly a
one-in-three chance that at least one of them looks significant by accident.
A per-tag table without multiplicity control is a machine for manufacturing
regressions that are not there, and a gate wired to one gets muted in a week —
which is the failure mode this whole project exists to avoid. So every interval
here is widened for the size of the family, and the sign-test p-values are
Holm-adjusted. The corrections are visible in the report rather than applied
quietly, because a reader comparing this table against ``compare`` needs to
know why the same slice has a wider interval here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compare import Comparison, check_comparable, compare_runs
from .stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    bonferroni_confidence,
    holm_adjusted,
)
from .types import RunResult

#: Slices smaller than this are reported as untested rather than tested badly.
#: Below roughly five cases the bootstrap has almost no resolution and every
#: interval spans most of [-1, 1], which reads as reassurance and is not.
DEFAULT_MIN_SLICE = 5


@dataclass(frozen=True)
class SliceComparison:
    """One tag's paired comparison, at the family-adjusted confidence level."""

    tag: str
    comparison: Comparison
    adjusted_p: float
    share: float

    @property
    def n(self) -> int:
        return self.comparison.n

    @property
    def direction(self) -> str:
        return self.comparison.direction

    @property
    def significant(self) -> bool:
        return self.comparison.significant

    def to_json(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "share": round(self.share, 6),
            "adjusted_sign_test_p": round(self.adjusted_p, 6),
            **self.comparison.to_json(),
        }


@dataclass(frozen=True)
class SliceReport:
    """Every testable slice of one comparison, plus what was left untested."""

    scorer: str
    aggregate: Comparison
    slices: tuple[SliceComparison, ...]
    skipped: tuple[tuple[str, int], ...]
    confidence: float
    per_slice_confidence: float
    min_cases: int

    @property
    def family_size(self) -> int:
        return len(self.slices)

    @property
    def regressions(self) -> tuple[SliceComparison, ...]:
        return tuple(s for s in self.slices if s.direction == "regression")

    @property
    def improvements(self) -> tuple[SliceComparison, ...]:
        return tuple(s for s in self.slices if s.direction == "improvement")

    @property
    def uncorroborated(self) -> tuple[SliceComparison, ...]:
        """Slices whose interval fires but whose adjusted sign test does not.

        The two instruments genuinely disagree on small slices, and the reason
        is not a bug: the sign test throws magnitude away and counts
        directions, so eight cases can produce at most p = 0.0078 before
        adjustment and rarely survive Holm. The interval is the verdict here —
        it is the instrument the rest of the harness gates on — but a slice
        the sign test cannot back up is a slice to re-run wider before anyone
        rewrites a prompt over it.
        """
        alpha = 1.0 - self.confidence
        return tuple(s for s in self.slices if s.significant and s.adjusted_p > alpha)

    @property
    def hidden_regressions(self) -> tuple[SliceComparison, ...]:
        """Slices that regressed while the aggregate did not.

        The reason the command exists. A green headline over one of these is
        not a wrong number — it is a correct number answering a question
        nobody asked.
        """
        if self.aggregate.direction == "regression":
            return ()
        return self.regressions

    def to_json(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer,
            "aggregate": self.aggregate.to_json(),
            "confidence": self.confidence,
            "per_slice_confidence": round(self.per_slice_confidence, 6),
            "family_size": self.family_size,
            "min_cases": self.min_cases,
            "slices": [s.to_json() for s in self.slices],
            "skipped": [{"tag": tag, "n": n} for tag, n in self.skipped],
            "hidden_regressions": [s.tag for s in self.hidden_regressions],
            "uncorroborated": [s.tag for s in self.uncorroborated],
        }


def compare_by_tag(
    baseline: RunResult,
    candidate: RunResult,
    scorer: str,
    *,
    min_cases: int = DEFAULT_MIN_SLICE,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    correct: bool = True,
    allow_dataset_drift: bool = False,
) -> SliceReport:
    """Compare two runs on every tag slice large enough to say anything about.

    ``confidence`` is the level held over the *whole table*: with ``correct``
    left on, each individual slice is tested at a stricter level so that the
    chance of any false alarm anywhere in the report stays at ``1 -
    confidence``. Pass ``correct=False`` to see the uncorrected per-slice
    intervals — useful for exploration, dishonest in a gate.
    """
    check_comparable(baseline, candidate, allow_dataset_drift=allow_dataset_drift)
    if min_cases < 2:
        raise ValueError(f"min_cases must be at least 2, got {min_cases}")

    total = len(baseline.case_ids)
    testable: list[tuple[str, tuple[str, ...]]] = []
    skipped: list[tuple[str, int]] = []
    for tag, case_ids in sorted(baseline.tags.items()):
        if len(case_ids) < min_cases:
            skipped.append((tag, len(case_ids)))
        else:
            testable.append((tag, case_ids))

    # The family is fixed by slice *size* alone, before any result is looked
    # at. Choosing which slices to correct for after seeing the deltas would
    # undo the correction it is meant to apply.
    family = len(testable)
    per_slice = bonferroni_confidence(confidence, family) if correct else confidence

    comparisons = [
        compare_runs(
            baseline.subset(case_ids),
            candidate.subset(case_ids),
            scorer,
            resamples=resamples,
            confidence=per_slice,
            seed=seed,
            allow_dataset_drift=True,  # already checked against the full runs
        )
        for _, case_ids in testable
    ]
    raw_p = [c.sign_test_p for c in comparisons]
    adjusted = holm_adjusted(raw_p) if correct else list(raw_p)

    slices = tuple(
        SliceComparison(
            tag=tag,
            comparison=comparison,
            adjusted_p=p,
            share=comparison.n / total if total else 0.0,
        )
        for (tag, _), comparison, p in zip(testable, comparisons, adjusted, strict=True)
    )
    return SliceReport(
        scorer=scorer,
        aggregate=compare_runs(
            baseline,
            candidate,
            scorer,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
            allow_dataset_drift=allow_dataset_drift,
        ),
        # Worst first: the reason anyone opens this table is to find the slice
        # that broke, and it should not be somewhere in the middle of it.
        slices=tuple(sorted(slices, key=lambda s: s.comparison.delta.point)),
        skipped=tuple(skipped),
        confidence=confidence,
        per_slice_confidence=per_slice,
        min_cases=min_cases,
    )

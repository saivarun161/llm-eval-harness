"""Running a target over a dataset and scoring what comes back."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .dataset import Dataset
from .scorers import DEFAULT_SCORERS, get_scorer
from .stats import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED, Interval, bootstrap_mean
from .types import Case, CaseScore, Prediction, RunResult

Target = Callable[[str], str]


@dataclass(frozen=True)
class ScoreSummary:
    """One scorer's aggregate over a run."""

    scorer: str
    n: int
    interval: Interval

    @property
    def mean(self) -> float:
        return self.interval.point

    def to_json(self) -> dict[str, object]:
        return {"scorer": self.scorer, "n": self.n, **self.interval.to_json()}


def evaluate(
    dataset: Dataset,
    target: Target,
    *,
    scorers: Iterable[str] = DEFAULT_SCORERS,
    target_name: str = "",
    label: str = "",
    created_at: str | None = None,
) -> RunResult:
    """Run ``target`` over every case and score each output.

    A target that raises is recorded as a failed prediction scoring zero — a
    system that crashes on an input has, for evaluation purposes, answered it
    badly. A *scorer* that raises is left to propagate, because that is a bug in
    the harness and quietly recording it as a zero would corrupt the result set
    in the one direction nobody would question.
    """
    scorer_names = list(dict.fromkeys(scorers))
    if not scorer_names:
        raise ValueError("at least one scorer is required")
    resolved = [(name, get_scorer(name)) for name in scorer_names]

    predictions: list[Prediction] = []
    scores: list[CaseScore] = []

    for case in dataset:
        prediction = _predict(case, target)
        predictions.append(prediction)
        for name, scorer in resolved:
            if prediction.error is not None:
                scores.append(CaseScore(case.id, name, 0.0, f"target failed: {prediction.error}"))
                continue
            verdict = scorer(case, prediction.output)
            scores.append(CaseScore(case.id, name, verdict.score, verdict.detail))

    return RunResult(
        target=target_name or getattr(target, "__name__", "target"),
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_fingerprint=dataset.fingerprint,
        case_ids=dataset.case_ids,
        predictions=tuple(predictions),
        scores=tuple(scores),
        tags=dataset.tag_index(),
        label=label,
        created_at=created_at,
    )


def _predict(case: Case, target: Target) -> Prediction:
    started = time.perf_counter()
    try:
        output = target(case.input)
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000.0
        return Prediction(case.id, "", elapsed, error=f"{type(exc).__name__}: {exc}")
    elapsed = (time.perf_counter() - started) * 1000.0
    return Prediction(case.id, str(output), elapsed)


def summarize(
    run: RunResult,
    *,
    scorers: Iterable[str] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> list[ScoreSummary]:
    """Mean and bootstrap CI for each scorer in a run."""
    names = list(scorers) if scorers is not None else list(run.scorers())
    summaries: list[ScoreSummary] = []
    for name in names:
        values = run.vector(name)
        summaries.append(
            ScoreSummary(
                scorer=name,
                n=len(values),
                interval=bootstrap_mean(
                    values, resamples=resamples, confidence=confidence, seed=seed
                ),
            )
        )
    return summaries


def summarize_by_tag(
    run: RunResult,
    scorer: str,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    min_cases: int = 3,
) -> dict[str, ScoreSummary]:
    """Per-tag breakdown, skipping slices too small to say anything about."""
    out: dict[str, ScoreSummary] = {}
    for tag, case_ids in sorted(run.tags.items()):
        if len(case_ids) < min_cases:
            continue
        values = run.subset(case_ids).vector(scorer)
        out[tag] = ScoreSummary(
            scorer=scorer,
            n=len(values),
            interval=bootstrap_mean(values, resamples=resamples, confidence=confidence, seed=seed),
        )
    return out

"""Calibrating a judge against human labels.

Every LLM-as-judge setup contains an unexamined assumption: that the judge
agrees with the people whose opinion the product is actually optimising for.
This module makes that assumption measurable.

Given a file of (question, reference, answer, human label) rows, it scores each
answer with the configured judge, searches for the score cut points that agree
best with the humans, and reports Cohen's kappa — agreement *after* discounting
what you would get by guessing. Raw agreement flatters a judge badly when the
labels are unbalanced: a judge that calls everything correct scores 80% raw
agreement on a set that is 80% correct, and a kappa of zero.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .dataset import BUILTIN_PREFIX
from .scorers.judge import DEFAULT_THRESHOLDS, JudgeVerdict, get_judge, label_from_score
from .stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SEED,
    Interval,
    agreement_rate,
    cohens_kappa,
    kappa_interpretation,
    pearson,
    percentile,
    spearman,
)

_DATA_DIR = Path(__file__).parent / "data"
#: Labels are ordinal: 0 wrong, 1 partially correct, 2 correct.
LABEL_NAMES = ("wrong", "partial", "correct")

Judge = Callable[[str, str, str], JudgeVerdict]


@dataclass(frozen=True)
class LabelledExample:
    """One human-graded answer."""

    id: str
    input: str
    expected: str
    prediction: str
    label: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1, 2):
            raise ValueError(f"example {self.id!r} has label {self.label!r}, expected 0, 1 or 2")


@dataclass(frozen=True)
class Calibration:
    """How well a judge tracks the humans, and where to put its cut points."""

    n: int
    judge_name: str
    thresholds: tuple[float, float]
    default_thresholds: tuple[float, float]
    kappa: float
    kappa_nominal: float
    kappa_at_default: float
    kappa_interval: Interval
    agreement: float
    pearson: float
    spearman: float
    confusion: list[list[int]]
    label_counts: tuple[int, int, int]

    @property
    def interpretation(self) -> str:
        return kappa_interpretation(self.kappa)

    @property
    def trustworthy(self) -> bool:
        """Whether this judge is solid enough to gate a release on.

        0.6 is Landis & Koch's boundary between "moderate" and "substantial".
        It is a convention, not a law of nature, and the README says so — but
        an explicit convention beats an implicit one.
        """
        return self.kappa >= 0.60

    def to_json(self) -> dict[str, object]:
        return {
            "n": self.n,
            "judge": self.judge_name,
            "thresholds": list(self.thresholds),
            "default_thresholds": list(self.default_thresholds),
            "kappa_linear": round(self.kappa, 6),
            "kappa_nominal": round(self.kappa_nominal, 6),
            "kappa_at_default_thresholds": round(self.kappa_at_default, 6),
            "kappa_interval": self.kappa_interval.to_json(),
            "agreement": round(self.agreement, 6),
            "pearson": round(self.pearson, 6),
            "spearman": round(self.spearman, 6),
            "confusion": self.confusion,
            "label_counts": list(self.label_counts),
            "interpretation": self.interpretation,
            "trustworthy": self.trustworthy,
        }


def parse_labels(text: str) -> list[LabelledExample]:
    examples: list[LabelledExample] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        if "__labels__" in payload:
            continue
        try:
            examples.append(
                LabelledExample(
                    id=str(payload.get("id", lineno)),
                    input=str(payload.get("input", "")),
                    expected=str(payload["expected"]),
                    prediction=str(payload["prediction"]),
                    label=int(payload["label"]),
                )
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    if not examples:
        raise ValueError("label file contains no examples")
    return examples


def load_labels(spec: str | Path) -> list[LabelledExample]:
    """Load labels from ``builtin:<name>`` or a path."""
    text = str(spec)
    if text.startswith(BUILTIN_PREFIX):
        path = _DATA_DIR / f"{text[len(BUILTIN_PREFIX) :]}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"unknown bundled label set {text!r}")
    else:
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"no label file at {path}")
    return parse_labels(path.read_text(encoding="utf-8"))


def search_thresholds(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    step: float = 0.02,
    weights: str | None = "linear",
) -> tuple[tuple[float, float], float]:
    """Grid-search the (partial, correct) cut points that maximise kappa.

    This is fitting two parameters on the label set, so the resulting kappa is
    optimistic by exactly as much as any in-sample fit. With two free
    parameters and several dozen labels the optimism is small, and the command
    reports the kappa at the *default* cut points alongside it so the size of
    the gap is visible rather than hidden.
    """
    best = (DEFAULT_THRESHOLDS, -2.0)
    steps = round(1.0 / step)
    for lo_i in range(1, steps):
        lo = lo_i * step
        for hi_i in range(lo_i + 1, steps):
            hi = hi_i * step
            predicted = [label_from_score(s, (lo, hi)) for s in scores]
            kappa = cohens_kappa(list(labels), predicted, weights=weights)
            if kappa > best[1]:
                best = ((round(lo, 4), round(hi, 4)), kappa)
    return best


def _bootstrap_kappa(
    labels: Sequence[int],
    predicted: Sequence[int],
    *,
    resamples: int = 500,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    weights: str | None = "linear",
) -> Interval:
    point = cohens_kappa(list(labels), list(predicted), weights=weights)
    rng = random.Random(seed)
    n = len(labels)
    draws: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(
            cohens_kappa([labels[i] for i in idx], [predicted[i] for i in idx], weights=weights)
        )
    draws.sort()
    alpha = (1.0 - confidence) / 2.0
    return Interval(point, percentile(draws, alpha), percentile(draws, 1 - alpha), confidence)


def calibrate(
    examples: Sequence[LabelledExample],
    *,
    judge: Judge | None = None,
    thresholds: tuple[float, float] | None = None,
    search: bool = True,
    seed: int = DEFAULT_SEED,
) -> Calibration:
    """Score every labelled example and measure agreement with the humans."""
    if not examples:
        raise ValueError("cannot calibrate on an empty label set")
    judge_fn = judge or get_judge()
    scores = [judge_fn(ex.input, ex.expected, ex.prediction).score for ex in examples]
    human = [ex.label for ex in examples]

    if thresholds is not None:
        chosen = thresholds
    elif search:
        chosen, _ = search_thresholds(scores, human)
    else:
        chosen = DEFAULT_THRESHOLDS

    predicted = [label_from_score(s, chosen) for s in scores]
    at_default = [label_from_score(s, DEFAULT_THRESHOLDS) for s in scores]

    confusion = [[0, 0, 0] for _ in range(3)]
    for actual, guess in zip(human, predicted, strict=True):
        confusion[actual][guess] += 1

    counts = (human.count(0), human.count(1), human.count(2))

    return Calibration(
        n=len(examples),
        judge_name=getattr(judge_fn, "name", type(judge_fn).__name__),
        thresholds=(float(chosen[0]), float(chosen[1])),
        default_thresholds=DEFAULT_THRESHOLDS,
        kappa=cohens_kappa(human, predicted, weights="linear"),
        kappa_nominal=cohens_kappa(human, predicted),
        kappa_at_default=cohens_kappa(human, at_default, weights="linear"),
        kappa_interval=_bootstrap_kappa(human, predicted, seed=seed),
        agreement=agreement_rate(human, predicted),
        pearson=pearson(scores, [float(h) for h in human]),
        spearman=spearman(scores, [float(h) for h in human]),
        confusion=confusion,
        label_counts=counts,
    )

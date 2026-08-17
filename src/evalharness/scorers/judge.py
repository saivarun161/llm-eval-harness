"""LLM-as-judge, with a deterministic judge you can actually test.

The default judge is a rubric of explicit signals — fact coverage, similarity,
numeric conflict, negation, refusal, verbosity — combined into one score. It is
not a language model, and it is not sold as one. It exists because a judge you
cannot reproduce is a judge you cannot regression-test, and because a harness
whose demo needs an API key is a harness nobody runs.

The real point is the layer above: whichever judge you plug in, the calibration
command measures it against human labels and reports Cohen's kappa. An
unmeasured judge is an opinion. A measured one is an instrument, and you know
its error bars before you let it gate a release.

Set ``EVALHARNESS_JUDGE=model`` (with ``OPENAI_API_KEY`` and the ``judge`` extra
installed) to swap in a real model behind the same interface.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..textutil import key_terms, normalize, numbers, similarity, tokens
from ..types import Case
from .base import Verdict, register

#: Cut points mapping a continuous judge score onto a wrong/partial/correct
#: rubric. These are not guesses: they are the output of running
#: `evalharness calibrate` against the bundled human labels. Run it against
#: your own labels and the numbers will move.
DEFAULT_THRESHOLDS = (0.26, 0.56)

# These are matched against normalised text, where apostrophes have already
# been stripped — hence "dont" rather than "don't".
_REFUSAL = re.compile(
    r"\b(i (?:do not|dont) know|i(?: m| am)? (?:not sure|unable)|as an ai|"
    r"cannot (?:answer|help|provide)|no information)\b"
)
_NEGATION = re.compile(r"\b(not|isnt|wasnt|arent|never|no longer|incorrect|false)\b")


@dataclass(frozen=True)
class JudgeVerdict:
    """A judge's score plus the signals that produced it."""

    score: float
    rationale: str
    signals: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Rubric:
    """Weights and penalties. Tunable, and tuned by the calibration command."""

    fact_weight: float = 0.55
    similarity_weight: float = 0.45
    numeric_conflict_penalty: float = 0.25
    negation_penalty: float = 0.35
    verbosity_ratio: float = 12.0
    verbosity_penalty: float = 0.9
    refusal_ceiling: float = 0.05


DEFAULT_RUBRIC = Rubric()


def _term_present(term: str, candidates: set[str]) -> bool:
    """Token match tolerant of inflection, intolerant of coincidence."""
    if term in candidates:
        return True
    return any(
        len(candidate) > 3 and SequenceMatcher(None, term, candidate).ratio() >= 0.86
        for candidate in candidates
    )


def _numeric_agreement(expected: str, prediction: str) -> tuple[float, bool]:
    """Fraction of reference numbers reproduced, and whether they were contradicted.

    A wrong number is a different kind of error from a missing one. "Apollo 11
    landed in 1968" is confidently false; "Apollo 11 landed some time in the
    sixties" is merely incomplete. Only the first is treated as a conflict.
    """
    want = numbers(expected)
    if not want:
        return 1.0, False
    got = numbers(prediction)
    matched = sum(1 for value in want if any(abs(value - other) < 1e-9 for other in got))
    conflict = bool(got) and matched == 0
    return matched / len(want), conflict


class HeuristicJudge:
    """A deterministic, explainable stand-in for a model judge."""

    name = "heuristic"

    def __init__(self, rubric: Rubric = DEFAULT_RUBRIC) -> None:
        self.rubric = rubric

    def __call__(self, question: str, expected: str, prediction: str) -> JudgeVerdict:
        rubric = self.rubric
        if not prediction.strip():
            return JudgeVerdict(0.0, "empty answer", {"empty": 1.0})

        terms = key_terms(expected)
        candidates = set(tokens(prediction))
        term_hits = sum(1 for term in terms if _term_present(term, candidates))
        term_coverage = term_hits / len(terms) if terms else 1.0

        numeric_coverage, numeric_conflict = _numeric_agreement(expected, prediction)
        fact_coverage = (
            0.5 * (term_coverage + numeric_coverage) if numbers(expected) else (term_coverage)
        )

        sim = similarity(prediction, expected)
        score = rubric.fact_weight * fact_coverage + rubric.similarity_weight * sim

        reasons: list[str] = [f"facts {fact_coverage:.2f}", f"similarity {sim:.2f}"]
        signals = {
            "fact_coverage": fact_coverage,
            "similarity": sim,
            "term_coverage": term_coverage,
            "numeric_coverage": numeric_coverage,
        }

        normalized_prediction = normalize(prediction, strip_articles=False)
        if _REFUSAL.search(normalized_prediction):
            score = min(score, rubric.refusal_ceiling)
            reasons.append("refusal")
            signals["refusal"] = 1.0

        if numeric_conflict:
            score *= rubric.numeric_conflict_penalty
            reasons.append("numeric conflict")
            signals["numeric_conflict"] = 1.0

        # A negation only counts against the answer when the reference does not
        # contain one itself — otherwise "the treaty was not ratified" would be
        # penalised for correctly reproducing a negative fact.
        if _NEGATION.search(normalized_prediction) and not _NEGATION.search(
            normalize(expected, strip_articles=False)
        ):
            score *= rubric.negation_penalty
            reasons.append("unmatched negation")
            signals["negation"] = 1.0

        length_ratio = len(tokens(prediction)) / max(1, len(tokens(expected)))
        signals["length_ratio"] = length_ratio
        if length_ratio > rubric.verbosity_ratio:
            score *= rubric.verbosity_penalty
            reasons.append(f"verbose ({length_ratio:.0f}x reference)")

        score = max(0.0, min(1.0, score))
        return JudgeVerdict(score, ", ".join(reasons), signals)


class ModelJudge:
    """Optional judge backed by an OpenAI-compatible chat endpoint.

    Same call signature as the heuristic judge, so nothing downstream changes —
    including the calibration command, which is the only honest way to find out
    whether the paid judge is actually better on your data.
    """

    name = "model"

    _PROMPT = (
        "You are grading an answer against a reference. Reply with a single "
        "number between 0 and 1: 1 if the answer conveys the reference, 0 if it "
        "contradicts or misses it, in between if partially correct. Reply with "
        "the number only.\n\nQuestion: {question}\nReference: {expected}\n"
        "Answer: {prediction}\nScore:"
    )

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only with the extra absent
            raise RuntimeError(
                "the model judge needs the 'judge' extra: pip install 'llm-eval-harness[judge]'"
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("EVALHARNESS_JUDGE=model requires OPENAI_API_KEY to be set")
        self.model = model or os.environ.get("EVALHARNESS_JUDGE_MODEL", "gpt-4o-mini")
        self._client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)

    def __call__(
        self, question: str, expected: str, prediction: str
    ) -> JudgeVerdict:  # pragma: no cover - needs a live endpoint
        prompt = self._PROMPT.format(question=question, expected=expected, prediction=prediction)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        )
        raw = (response.choices[0].message.content or "").strip()
        match = re.search(r"\d*\.?\d+", raw)
        if not match:
            return JudgeVerdict(0.0, f"unparseable judge reply: {raw!r}", {"parse_error": 1.0})
        score = max(0.0, min(1.0, float(match.group())))
        return JudgeVerdict(score, f"model judge ({self.model}) returned {score:.2f}")


_CACHED: dict[str, HeuristicJudge | ModelJudge] = {}


def get_judge() -> HeuristicJudge | ModelJudge:
    """Return the judge selected by ``EVALHARNESS_JUDGE`` (default heuristic)."""
    backend = os.environ.get("EVALHARNESS_JUDGE", "heuristic").strip().lower()
    if backend not in {"heuristic", "model"}:
        raise ValueError(f"EVALHARNESS_JUDGE must be 'heuristic' or 'model', got {backend!r}")
    if backend not in _CACHED:
        _CACHED[backend] = HeuristicJudge() if backend == "heuristic" else ModelJudge()
    return _CACHED[backend]


def reset_judge_cache() -> None:
    """Drop the memoised judge so a changed environment takes effect."""
    _CACHED.clear()


def label_from_score(score: float, thresholds: tuple[float, float] = DEFAULT_THRESHOLDS) -> int:
    """Bucket a continuous judge score into 0 (wrong) / 1 (partial) / 2 (correct)."""
    low, high = thresholds
    if score >= high:
        return 2
    if score >= low:
        return 1
    return 0


@register("judge", "Graded LLM-as-judge score (heuristic by default)")
def judge(case: Case, output: str) -> Verdict:
    """Rubric score for how well the output answers the case."""
    verdict = get_judge()(case.input, case.expected, output)
    return Verdict(verdict.score, verdict.rationale)


@register("judge_pass", "Judge score thresholded into a pass or fail")
def judge_pass(case: Case, output: str) -> Verdict:
    """Binary view of the judge, using the calibrated 'correct' cut point."""
    threshold = float(case.metadata.get("judge_threshold", DEFAULT_THRESHOLDS[1]))
    verdict = get_judge()(case.input, case.expected, output)
    passed = verdict.score >= threshold
    return Verdict(
        1.0 if passed else 0.0,
        f"judge {verdict.score:.2f} {'>=' if passed else '<'} {threshold:.2f} "
        f"({verdict.rationale})",
    )

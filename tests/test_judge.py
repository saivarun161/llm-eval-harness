from __future__ import annotations

import pytest

from evalharness.scorers.judge import (
    DEFAULT_THRESHOLDS,
    HeuristicJudge,
    Rubric,
    get_judge,
    label_from_score,
    reset_judge_cache,
)
from evalharness.types import Case

JUDGE = HeuristicJudge()


def grade(expected: str, prediction: str, question: str = "q") -> float:
    return JUDGE(question, expected, prediction).score


def test_exact_answer_scores_near_the_top():
    assert grade("Canberra", "Canberra") > 0.9


def test_a_correct_answer_wrapped_in_prose_stays_high():
    assert grade("Canberra", "Based on what I know, the answer is Canberra.") > 0.8


def test_a_paraphrase_outranks_a_wrong_answer_without_matching_an_exact_one():
    # Surface-level judging reads "Armstrong was the first to set foot there"
    # as only partially matching "Neil Armstrong". That is a real limitation,
    # and it is the sort of gap the calibration command is for.
    paraphrase = grade("Neil Armstrong", "Armstrong was the first to set foot there")
    assert (
        grade("Neil Armstrong", "Buzz Aldrin")
        < paraphrase
        < grade("Neil Armstrong", "Neil Armstrong")
    )


def test_wrong_entity_scores_low():
    assert grade("Canberra", "Sydney") < 0.3
    assert grade("The Inca", "The Maya") < 0.45


def test_truncated_answers_land_in_between():
    expected = "It caps how many requests a client may make in a time window"
    partial = grade(expected, "It caps how many requests a client may")
    assert grade(expected, "The sky is blue") < partial < grade(expected, expected)


def test_a_contradicting_number_is_flagged_and_penalised():
    verdict = JUDGE("q", "206 bones", "306 bones")
    assert "numeric conflict" in verdict.rationale
    assert verdict.score < 0.5 * grade("206 bones", "206 bones")


def test_spelled_out_numbers_are_a_known_blind_spot():
    # The judge is lexical: it cannot see that "two hundred" is near 206, so it
    # marks a vague-but-not-wrong answer down. The bundled human labels call
    # this one partially correct, and the resulting disagreement is visible in
    # the calibration confusion matrix rather than hidden.
    assert grade("206 bones", "Somewhere around two hundred") < 0.3


def test_a_matching_number_is_not_a_conflict():
    verdict = JUDGE("q", "1,440 minutes", "A day contains 1,440 minutes")
    assert "numeric conflict" not in verdict.rationale
    assert verdict.score > 0.7


def test_refusals_collapse_to_almost_zero():
    verdict = JUDGE("q", "Canberra", "I don't know.")
    assert verdict.score <= 0.05
    assert "refusal" in verdict.rationale


def test_empty_answers_score_zero():
    verdict = JUDGE("q", "Canberra", "   ")
    assert verdict.score == 0.0
    assert verdict.signals == {"empty": 1.0}


def test_unmatched_negation_is_penalised():
    plain = grade("The treaty was ratified", "The treaty was ratified")
    negated = grade("The treaty was ratified", "The treaty was not ratified")
    assert negated < plain


def test_a_negation_in_the_reference_is_not_punished():
    # Reproducing a negative fact must not be treated as contradicting it.
    verdict = JUDGE("q", "The treaty was not ratified", "The treaty was not ratified")
    assert "unmatched negation" not in verdict.rationale
    assert verdict.score > 0.9


def test_padding_an_answer_does_not_pay():
    short = grade("Canberra", "Canberra")
    padded = grade("Canberra", "Canberra " + "and some other words " * 20)
    assert padded < short


def test_signals_are_exposed_for_inspection():
    verdict = JUDGE("q", "206 bones", "206 bones")
    assert set(verdict.signals) >= {"fact_coverage", "similarity", "numeric_coverage"}
    assert verdict.rationale


def test_rubric_weights_are_honoured():
    lenient = HeuristicJudge(Rubric(numeric_conflict_penalty=1.0))
    strict = HeuristicJudge(Rubric(numeric_conflict_penalty=0.1))
    assert (
        lenient("q", "206 bones", "306 bones").score > strict("q", "206 bones", "306 bones").score
    )


def test_label_from_score_uses_the_cut_points():
    low, high = DEFAULT_THRESHOLDS
    assert label_from_score(high + 0.01) == 2
    assert label_from_score((low + high) / 2) == 1
    assert label_from_score(low - 0.01) == 0
    assert label_from_score(0.5, (0.4, 0.9)) == 1


def test_judge_scorer_reflects_the_configured_backend(monkeypatch):
    monkeypatch.delenv("EVALHARNESS_JUDGE", raising=False)
    reset_judge_cache()
    assert get_judge().name == "heuristic"

    monkeypatch.setenv("EVALHARNESS_JUDGE", "nonsense")
    reset_judge_cache()
    with pytest.raises(ValueError, match="EVALHARNESS_JUDGE"):
        get_judge()

    monkeypatch.delenv("EVALHARNESS_JUDGE", raising=False)
    reset_judge_cache()


def test_model_judge_refuses_to_start_without_a_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EVALHARNESS_JUDGE", "model")
    reset_judge_cache()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_judge()
    monkeypatch.delenv("EVALHARNESS_JUDGE", raising=False)
    reset_judge_cache()


def test_judge_pass_thresholds_the_graded_score():
    from evalharness.scorers import get_scorer

    case = Case("c", "What is the capital of Australia?", "Canberra")
    assert get_scorer("judge_pass")(case, "Canberra").score == 1.0
    assert get_scorer("judge_pass")(case, "Sydney").score == 0.0

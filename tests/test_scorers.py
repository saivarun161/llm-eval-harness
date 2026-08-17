from __future__ import annotations

import pytest

from evalharness.scorers import Verdict, available, describe, get_scorer, register, unregister
from evalharness.types import Case

CASE = Case("c1", "What is the capital of Australia?", "Canberra")


def score(name: str, output: str, case: Case = CASE) -> float:
    return get_scorer(name)(case, output).score


def test_every_registered_scorer_stays_in_range():
    for name in available():
        for output in ["Canberra", "", "Sydney", "Based on what I know, it is Canberra."]:
            verdict = get_scorer(name)(CASE, output)
            assert 0.0 <= verdict.score <= 1.0, name


def test_exact_match_ignores_cosmetic_differences_only():
    assert score("exact_match", "Canberra") == 1.0
    assert score("exact_match", "  canberra. ") == 1.0
    assert score("exact_match", "The answer is Canberra") == 0.0


def test_contains_finds_the_reference_in_prose():
    assert score("contains", "I think it is Canberra, actually") == 1.0
    assert score("contains", "Sydney") == 0.0


def test_token_f1_is_graded():
    case = Case("c", "q", "Canada and the United States")
    assert score("token_f1", "Canada and the United States", case) == pytest.approx(1.0)
    partial = score("token_f1", "Canada and", case)
    assert 0.0 < partial < 1.0
    assert score("token_f1", "Russia and China", case) < partial


def test_token_f1_handles_empty_sides():
    assert score("token_f1", "") == 0.0
    assert get_scorer("token_f1")(Case("c", "q", ""), "").score == 1.0


def test_fuzzy_degrades_smoothly():
    assert score("fuzzy", "Canberra") == pytest.approx(1.0)
    assert score("fuzzy", "Canbera") > 0.8
    assert score("fuzzy", "Sydney") < 0.5


def test_regex_uses_case_metadata():
    case = Case("c", "How many bones?", "206 bones", metadata={"pattern": r"\b206\b"})
    assert score("regex", "there are 206 of them", case) == 1.0
    assert score("regex", "there are 306 of them", case) == 0.0


def test_regex_falls_back_to_the_reference():
    assert score("regex", "the answer is Canberra") == 1.0


def test_regex_reports_a_broken_pattern_instead_of_raising():
    case = Case("c", "q", "x", metadata={"pattern": "([unclosed"})
    verdict = get_scorer("regex")(case, "anything")
    assert verdict.score == 0.0
    assert "invalid pattern" in verdict.detail


def test_semantic_survives_rewording_where_exact_match_does_not():
    verbose = "Based on what I know, the answer to that is Canberra. Hope that helps!"
    assert score("exact_match", verbose) == 0.0
    assert score("semantic", verbose) > 0.6
    assert score("semantic_pass", verbose) == 1.0


def test_semantic_threshold_is_per_case():
    strict = Case("c", "q", "Canberra", metadata={"semantic_threshold": 0.99})
    verbose = "the answer is Canberra"
    assert score("semantic_pass", verbose, strict) == 0.0


def test_scorer_registry_round_trip():
    @register("always_half", "A test scorer")
    def always_half(case: Case, output: str) -> Verdict:
        return Verdict(0.5, "fixed")

    try:
        assert "always_half" in available()
        assert describe("always_half") == "A test scorer"
        assert score("always_half", "anything") == 0.5
        with pytest.raises(ValueError, match="already registered"):
            register("always_half")(always_half)
    finally:
        unregister("always_half")
    assert "always_half" not in available()


def test_unknown_scorer_lists_the_alternatives():
    with pytest.raises(KeyError, match="registered scorers"):
        get_scorer("nope")


def test_verdict_range_is_enforced():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Verdict(1.5)

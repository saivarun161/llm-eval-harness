from __future__ import annotations

import pytest

from evalharness.dataset import Dataset
from evalharness.runner import evaluate, summarize, summarize_by_tag
from evalharness.targets import (
    BEHAVIOURS,
    PROFILES,
    builtin_target,
    load_target,
    render,
    resolve_target,
    stable_uniform,
)
from evalharness.types import Case


def echo_expected(question: str) -> str:
    return {"What is the capital of France?": "Paris"}.get(question, "no idea")


def test_evaluate_scores_every_case_with_every_scorer(tiny_dataset: Dataset):
    run = evaluate(
        tiny_dataset,
        lambda q: "Paris",
        scorers=["exact_match", "semantic"],
        target_name="always_paris",
    )
    assert run.case_ids == tiny_dataset.case_ids
    assert len(run.scores) == len(tiny_dataset) * 2
    assert run.scorers() == ("exact_match", "semantic")
    assert run.vector("exact_match") == [1.0, 0.0, 0.0, 0.0]


def test_a_failing_target_scores_zero_rather_than_crashing(tiny_dataset: Dataset):
    def broken(question: str) -> str:
        if "2 + 2" in question:
            raise RuntimeError("model timeout")
        return "Paris"

    run = evaluate(tiny_dataset, broken, scorers=["exact_match"])
    failed = next(p for p in run.predictions if p.case_id == "c2")
    assert failed.error is not None
    assert "model timeout" in failed.error
    assert run.vector("exact_match")[1] == 0.0
    assert "target failed" in next(s for s in run.scores if s.case_id == "c2").detail


def test_a_failing_scorer_is_not_silently_recorded_as_zero(tiny_dataset: Dataset):
    from evalharness.scorers import register, unregister

    @register("explodes")
    def explodes(case: Case, output: str):
        raise RuntimeError("scorer bug")

    try:
        with pytest.raises(RuntimeError, match="scorer bug"):
            evaluate(tiny_dataset, lambda q: "x", scorers=["explodes"])
    finally:
        unregister("explodes")


def test_duplicate_scorers_are_collapsed(tiny_dataset: Dataset):
    run = evaluate(tiny_dataset, lambda q: "x", scorers=["exact_match", "exact_match"])
    assert run.scorers() == ("exact_match",)


def test_at_least_one_scorer_is_required(tiny_dataset: Dataset):
    with pytest.raises(ValueError, match="at least one scorer"):
        evaluate(tiny_dataset, lambda q: "x", scorers=[])


def test_summaries_carry_intervals(tiny_dataset: Dataset):
    run = evaluate(tiny_dataset, lambda q: "Paris", scorers=["exact_match"])
    summary = summarize(run, resamples=300)[0]
    assert summary.n == 4
    assert summary.mean == pytest.approx(0.25)
    assert summary.interval.lo <= summary.mean <= summary.interval.hi
    assert summary.to_json()["scorer"] == "exact_match"


def test_tag_breakdown_skips_slices_that_are_too_small(tiny_dataset: Dataset):
    run = evaluate(tiny_dataset, lambda q: "Paris", scorers=["exact_match"])
    assert summarize_by_tag(run, "exact_match", resamples=100) == {}
    breakdown = summarize_by_tag(run, "exact_match", resamples=100, min_cases=1)
    assert set(breakdown) == {"geography", "literature", "math", "science"}
    assert breakdown["geography"].mean == 1.0


def test_profiles_are_valid_probability_distributions():
    for name, profile in PROFILES.items():
        assert sum(profile.weights) == pytest.approx(1.0), name
        assert len(profile.weights) == len(BEHAVIOURS)


def test_profile_weights_are_validated():
    from evalharness.targets import Profile

    with pytest.raises(ValueError, match="sum to"):
        Profile("bad", (0.5, 0.1, 0.1, 0.1, 0.1, 0.0))
    with pytest.raises(ValueError, match="needs 6 weights"):
        Profile("bad", (1.0,))


def test_stable_uniform_is_deterministic_and_bounded():
    assert stable_uniform("case-1") == stable_uniform("case-1")
    assert stable_uniform("case-1") != stable_uniform("case-2")
    assert all(0.0 <= stable_uniform(f"c{i}") < 1.0 for i in range(200))


def test_case_difficulty_is_shared_across_profiles(bundled: Dataset):
    # A case the baseline gets verbatim must not be a case the better system
    # gets wrong: the profiles are nested bands over one shared draw.
    baseline, candidate = PROFILES["baseline"], PROFILES["candidate"]
    order = {behaviour: i for i, behaviour in enumerate(BEHAVIOURS)}
    for case in bundled:
        assert order[candidate.behaviour_for(case.id)] <= order[baseline.behaviour_for(case.id)]


def test_render_produces_each_behaviour():
    case = Case(
        "c",
        "q",
        "Canada and the United States",
        metadata={"paraphrase": "the Canada-US border", "wrong": "Russia and China"},
    )
    assert render(case, "verbatim") == "Canada and the United States"
    assert "Canada and the United States" in render(case, "verbose")
    assert render(case, "paraphrase") == "the Canada-US border"
    assert render(case, "wrong") == "Russia and China"
    assert render(case, "truncate") == "Canada and the"
    assert "don't know" in render(case, "refuse")


def test_render_falls_back_when_metadata_is_absent():
    case = Case("c", "q", "Paris")
    assert render(case, "paraphrase") == "Paris"
    assert render(case, "wrong") != "Paris"


def test_always_verbose_profile_wraps_correct_answers():
    case = Case("c", "q", "Paris")
    assert render(case, "verbatim", always_verbose=True).startswith("Based on what I know")


def test_builtin_target_rejects_unknown_questions(tiny_dataset: Dataset):
    target = builtin_target("baseline", tiny_dataset.cases)
    with pytest.raises(KeyError, match="unknown question"):
        target("a question that is not in the dataset")


def test_unknown_builtin_profile_lists_alternatives(tiny_dataset: Dataset):
    with pytest.raises(KeyError, match="available"):
        builtin_target("does_not_exist", tiny_dataset.cases)


def test_targets_can_be_imported_from_a_module_path(tiny_dataset: Dataset):
    target = resolve_target(f"{__name__}:echo_expected", tiny_dataset.cases)
    assert target("What is the capital of France?") == "Paris"


def test_target_resolution_errors_are_actionable(tiny_dataset: Dataset):
    with pytest.raises(ValueError, match="module:function"):
        resolve_target("not_a_profile", tiny_dataset.cases)
    with pytest.raises(ImportError):
        load_target("no_such_module_anywhere:fn")
    with pytest.raises(AttributeError):
        load_target(f"{__name__}:missing_function")
    with pytest.raises(TypeError, match="not callable"):
        load_target(f"{__name__}:BEHAVIOURS")

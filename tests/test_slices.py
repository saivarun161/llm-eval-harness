from __future__ import annotations

import pytest

from evalharness import report
from evalharness.compare import DatasetMismatch
from evalharness.dataset import Dataset
from evalharness.runner import evaluate
from evalharness.slices import compare_by_tag
from evalharness.stats import bonferroni_confidence, holm_adjusted
from evalharness.targets import resolve_target
from evalharness.types import CaseScore, RunResult


def _run(bundled: Dataset, profile: str) -> RunResult:
    return evaluate(
        bundled,
        resolve_target(profile, bundled.cases),
        scorers=["judge"],
        target_name=profile,
        label=profile,
    )


@pytest.fixture
def patchy_run(bundled: Dataset) -> RunResult:
    return _run(bundled, "patchy")


# --------------------------------------------------------------------------- #
# the multiplicity machinery
# --------------------------------------------------------------------------- #


def test_holm_is_a_step_down_adjustment():
    assert holm_adjusted([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_never_reports_a_larger_p_as_more_significant():
    adjusted = holm_adjusted([0.001, 0.4, 0.02, 0.9])
    ordered = [p for _, p in sorted(zip([0.001, 0.4, 0.02, 0.9], adjusted, strict=True))]
    assert ordered == sorted(ordered)


def test_holm_is_less_conservative_than_plain_bonferroni():
    raw = [0.001, 0.02, 0.03]
    assert holm_adjusted(raw)[2] < min(1.0, len(raw) * raw[2]) + 1e-12
    assert holm_adjusted(raw)[0] == pytest.approx(0.003)


def test_holm_caps_at_one_and_handles_the_empty_family():
    assert holm_adjusted([0.9, 0.95]) == [1.0, 1.0]
    assert holm_adjusted([]) == []


def test_holm_rejects_values_that_are_not_probabilities():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjusted([0.5, 1.4])


def test_bonferroni_widens_with_the_family_and_is_a_no_op_for_one_test():
    assert bonferroni_confidence(0.95, 1) == 0.95
    assert bonferroni_confidence(0.95, 0) == 0.95
    assert bonferroni_confidence(0.95, 8) == pytest.approx(0.99375)
    assert bonferroni_confidence(0.95, 20) > bonferroni_confidence(0.95, 8)


# --------------------------------------------------------------------------- #
# the slice report
# --------------------------------------------------------------------------- #


def test_a_slice_regression_survives_a_flat_aggregate(baseline_run, patchy_run):
    result = compare_by_tag(baseline_run, patchy_run, "judge", resamples=800)
    assert result.aggregate.direction == "inconclusive"
    assert [s.tag for s in result.hidden_regressions] == ["geography"]
    geography = next(s for s in result.slices if s.tag == "geography")
    assert geography.direction == "regression"
    assert geography.n == 8
    assert geography.share == pytest.approx(8 / 45)


def test_a_uniform_improvement_leaves_no_hidden_regression(baseline_run, candidate_run):
    result = compare_by_tag(baseline_run, candidate_run, "judge", resamples=800)
    assert result.aggregate.direction == "improvement"
    assert result.hidden_regressions == ()


def test_a_regressed_aggregate_is_not_reported_as_hiding_anything(baseline_run, regressed_run):
    result = compare_by_tag(baseline_run, regressed_run, "judge", resamples=800)
    assert result.aggregate.direction == "regression"
    assert result.regressions  # slices did regress
    assert result.hidden_regressions == ()  # but nothing was hidden by the mean


def test_slices_are_ordered_worst_first(baseline_run, patchy_run):
    result = compare_by_tag(baseline_run, patchy_run, "judge", resamples=400)
    deltas = [s.comparison.delta.point for s in result.slices]
    assert deltas == sorted(deltas)
    assert result.slices[0].tag == "geography"


def test_correction_widens_every_interval(baseline_run, patchy_run):
    corrected = compare_by_tag(baseline_run, patchy_run, "judge", resamples=800)
    raw = compare_by_tag(baseline_run, patchy_run, "judge", resamples=800, correct=False)
    assert corrected.per_slice_confidence > raw.per_slice_confidence == 0.95
    by_tag = {s.tag: s for s in raw.slices}
    for item in corrected.slices:
        width = item.comparison.delta.hi - item.comparison.delta.lo
        assert width >= by_tag[item.tag].comparison.delta.hi - by_tag[item.tag].comparison.delta.lo
    assert all(s.adjusted_p == s.comparison.sign_test_p for s in raw.slices)


def test_small_slices_are_reported_as_untested_not_dropped(baseline_run, patchy_run):
    result = compare_by_tag(baseline_run, patchy_run, "judge", resamples=400)
    assert dict(result.skipped) == {"engineering": 3, "math": 4}
    assert result.family_size == len(result.slices) == 8
    assert not set(dict(result.skipped)) & {s.tag for s in result.slices}


def test_the_family_size_follows_min_cases(baseline_run, patchy_run):
    wide = compare_by_tag(baseline_run, patchy_run, "judge", resamples=200, min_cases=3)
    assert wide.family_size == 10
    assert wide.skipped == ()
    assert wide.per_slice_confidence == pytest.approx(bonferroni_confidence(0.95, 10))


def test_min_cases_below_two_is_refused(baseline_run, patchy_run):
    with pytest.raises(ValueError, match="at least 2"):
        compare_by_tag(baseline_run, patchy_run, "judge", min_cases=1)


def test_uncorroborated_slices_are_flagged(baseline_run, patchy_run):
    result = compare_by_tag(baseline_run, patchy_run, "judge", resamples=800)
    alpha = 1.0 - result.confidence
    for item in result.uncorroborated:
        assert item.significant and item.adjusted_p > alpha
    assert set(result.uncorroborated) <= set(result.slices)


def test_a_run_without_tags_produces_an_empty_but_honest_report(tiny_dataset: Dataset):
    untagged = Dataset(
        name="untagged",
        version="1.0.0",
        cases=tuple(type(case)(case.id, case.input, case.expected) for case in tiny_dataset.cases),
    )
    a = evaluate(untagged, lambda q: "Paris", scorers=["exact_match"], target_name="a")
    b = evaluate(untagged, lambda q: "Paris", scorers=["exact_match"], target_name="b")
    result = compare_by_tag(a, b, "exact_match", resamples=100)
    assert result.slices == ()
    assert result.family_size == 0
    assert result.per_slice_confidence == 0.95
    assert "No slice is large enough" in report.render_slice_report(result)


def test_slicing_refuses_a_drifted_dataset(baseline_run, patchy_run):
    drifted = RunResult(
        target=patchy_run.target,
        dataset_name=patchy_run.dataset_name,
        dataset_version=patchy_run.dataset_version,
        dataset_fingerprint="0" * 16,
        case_ids=patchy_run.case_ids,
        predictions=patchy_run.predictions,
        scores=patchy_run.scores,
        tags=patchy_run.tags,
    )
    with pytest.raises(DatasetMismatch, match="fingerprint changed"):
        compare_by_tag(baseline_run, drifted, "judge", resamples=100)
    assert (
        compare_by_tag(
            baseline_run, drifted, "judge", resamples=100, allow_dataset_drift=True
        ).family_size
        == 8
    )


def test_a_missing_scorer_names_the_slice_it_failed_on(baseline_run, patchy_run):
    holed = RunResult(
        target=patchy_run.target,
        dataset_name=patchy_run.dataset_name,
        dataset_version=patchy_run.dataset_version,
        dataset_fingerprint=patchy_run.dataset_fingerprint,
        case_ids=patchy_run.case_ids,
        predictions=patchy_run.predictions,
        scores=tuple(s for s in patchy_run.scores if s.scorer != "judge")
        + tuple(CaseScore(cid, "judge", 1.0) for cid in patchy_run.case_ids[:-1]),
        tags=patchy_run.tags,
    )
    with pytest.raises(KeyError, match="judge"):
        compare_by_tag(baseline_run, holed, "judge", resamples=100)


def test_the_report_json_round_trips_the_verdicts(baseline_run, patchy_run):
    result = compare_by_tag(baseline_run, patchy_run, "judge", resamples=400)
    payload = result.to_json()
    assert payload["scorer"] == "judge"
    assert payload["family_size"] == 8
    assert payload["hidden_regressions"] == ["geography"]
    assert payload["skipped"] == [{"tag": "engineering", "n": 3}, {"tag": "math", "n": 4}]
    assert payload["per_slice_confidence"] == pytest.approx(0.99375)
    first = payload["slices"][0]
    assert first["tag"] == "geography"
    assert first["direction"] == "regression"
    assert 0.0 <= first["adjusted_sign_test_p"] <= 1.0


def test_the_rendered_report_names_the_hidden_regression(baseline_run, patchy_run):
    text = report.render_slice_report(compare_by_tag(baseline_run, patchy_run, "judge"))
    assert "sliced by tag" in text
    assert "geography" in text
    assert "A headline mean would have shipped this." in text
    assert "engineering (n=3)" in text
    assert "not known to be fine" in text

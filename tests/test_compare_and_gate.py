from __future__ import annotations

from dataclasses import replace

import pytest

from evalharness.compare import DatasetMismatch, compare_all, compare_runs
from evalharness.gate import evaluate_gate
from evalharness.types import RunResult


def test_a_real_improvement_is_reported_as_significant(baseline_run, candidate_run):
    comparison = compare_runs(baseline_run, candidate_run, "judge")
    assert comparison.delta.point > 0
    assert comparison.significant
    assert comparison.direction == "improvement"
    assert comparison.wins > comparison.losses
    assert comparison.sign_test_p < 0.05
    assert comparison.n == len(baseline_run.case_ids)


def test_a_run_compared_against_itself_is_a_flat_zero(baseline_run):
    comparison = compare_runs(baseline_run, baseline_run, "judge")
    assert comparison.delta.point == 0.0
    assert not comparison.significant
    assert comparison.direction == "inconclusive"
    assert comparison.ties == comparison.n
    assert comparison.sign_test_p == 1.0


def test_a_regression_is_reported_with_a_negative_delta(baseline_run, regressed_run):
    comparison = compare_runs(baseline_run, regressed_run, "judge")
    assert comparison.delta.point < 0
    assert comparison.direction == "regression"
    assert comparison.losses > comparison.wins


def test_comparison_serialises_every_number(baseline_run, candidate_run):
    payload = compare_runs(baseline_run, candidate_run, "judge").to_json()
    assert payload["direction"] == "improvement"
    assert set(payload["delta"]) == {"point", "lo", "hi", "confidence"}
    assert payload["min_detectable_effect"] > 0


def test_compare_all_covers_shared_scorers(baseline_run, candidate_run):
    comparisons = compare_all(baseline_run, candidate_run)
    assert {c.scorer for c in comparisons} == {"exact_match", "judge"}


def test_dataset_drift_blocks_a_comparison(baseline_run, candidate_run):
    drifted = replace(candidate_run, dataset_fingerprint="0000000000000000")
    with pytest.raises(DatasetMismatch, match="fingerprint changed"):
        compare_runs(baseline_run, drifted, "judge")
    # ...and can be overridden deliberately.
    assert compare_runs(baseline_run, drifted, "judge", allow_dataset_drift=True).n > 0


def test_different_cases_block_a_comparison(baseline_run, candidate_run):
    shortened = candidate_run.subset(list(candidate_run.case_ids[:-1]))
    with pytest.raises(DatasetMismatch, match="different cases"):
        compare_runs(baseline_run, shortened, "judge")


def test_missing_scores_are_reported_clearly(baseline_run: RunResult):
    stripped = replace(
        baseline_run,
        scores=tuple(s for s in baseline_run.scores if s.case_id != baseline_run.case_ids[0]),
    )
    with pytest.raises(KeyError, match="no 'judge' score"):
        stripped.vector("judge")


def test_gate_blocks_a_real_regression(baseline_run, regressed_run):
    result = evaluate_gate(baseline_run, regressed_run, "judge", tolerance=0.02)
    assert not result.passed
    assert result.exit_code == 1
    assert "regressed" in result.reasons[0]
    assert result.to_json()["passed"] is False


def test_gate_lets_an_improvement_through(baseline_run, candidate_run):
    result = evaluate_gate(baseline_run, candidate_run, "judge", tolerance=0.02)
    assert result.passed
    assert result.exit_code == 0


def test_confident_mode_is_quieter_than_cautious_mode(baseline_run, regressed_run):
    # A regression whose interval merely touches the tolerance fails in
    # cautious mode and passes in confident mode. That difference is the whole
    # reason both modes exist.
    comparison = compare_runs(baseline_run, regressed_run, "judge")
    tolerance = abs(comparison.delta.hi) + 0.01
    confident = evaluate_gate(
        baseline_run, regressed_run, "judge", tolerance=tolerance, mode="confident"
    )
    cautious = evaluate_gate(
        baseline_run, regressed_run, "judge", tolerance=tolerance, mode="cautious"
    )
    assert confident.passed
    assert not cautious.passed


def test_a_generous_tolerance_absorbs_a_small_drop(baseline_run, regressed_run):
    assert evaluate_gate(baseline_run, regressed_run, "judge", tolerance=1.0).passed


def test_an_underpowered_eval_set_fails_the_gate(baseline_run, candidate_run):
    result = evaluate_gate(baseline_run, candidate_run, "judge", tolerance=0.02, required_mde=0.001)
    assert not result.passed
    assert result.underpowered
    assert "underpowered" in result.reasons[-1]


def test_a_sufficient_eval_set_satisfies_the_power_check(baseline_run, candidate_run):
    result = evaluate_gate(baseline_run, candidate_run, "judge", required_mde=1.0)
    assert result.passed
    assert not result.underpowered


def test_gate_arguments_are_validated(baseline_run, candidate_run):
    with pytest.raises(ValueError, match="gate mode"):
        evaluate_gate(baseline_run, candidate_run, "judge", mode="vibes")
    with pytest.raises(ValueError, match="tolerance"):
        evaluate_gate(baseline_run, candidate_run, "judge", tolerance=-0.1)

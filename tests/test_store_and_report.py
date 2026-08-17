from __future__ import annotations

import json

import pytest

from evalharness import report
from evalharness.calibration import calibrate, load_labels
from evalharness.compare import compare_runs
from evalharness.gate import evaluate_gate
from evalharness.runner import summarize, summarize_by_tag
from evalharness.store import load_run, save_run
from evalharness.types import Case, CaseScore, Prediction, RunResult


def test_run_survives_a_round_trip_through_disk(tmp_path, baseline_run):
    path = save_run(baseline_run, tmp_path / "runs" / "baseline.json")
    assert path.exists()
    reloaded = load_run(path)
    assert reloaded.case_ids == baseline_run.case_ids
    assert reloaded.dataset_fingerprint == baseline_run.dataset_fingerprint
    assert reloaded.vector("judge") == pytest.approx(baseline_run.vector("judge"))
    assert reloaded.tags == baseline_run.tags


def test_saved_runs_are_readable_json(tmp_path, baseline_run):
    path = save_run(baseline_run, tmp_path / "baseline.json")
    payload = json.loads(path.read_text())
    assert payload["dataset"]["fingerprint"] == baseline_run.dataset_fingerprint
    assert payload["format_version"] == 1
    assert len(payload["predictions"]) == len(baseline_run.case_ids)


def test_a_future_format_version_is_refused(tmp_path, baseline_run):
    path = save_run(baseline_run, tmp_path / "baseline.json")
    payload = json.loads(path.read_text())
    payload["format_version"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="format version 99"):
        load_run(path)


def test_corrupt_and_missing_files_report_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_run(broken)
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"format_version": 1}))
    with pytest.raises(ValueError, match="not a valid run file"):
        load_run(incomplete)


def test_case_score_range_is_enforced():
    with pytest.raises(ValueError, match="out-of-range"):
        CaseScore("c1", "judge", 1.5)


def test_prediction_round_trip():
    prediction = Prediction("c1", "Paris", 12.345, error=None)
    assert Prediction.from_json(prediction.to_json()) == prediction
    failed = Prediction("c2", "", 1.0, error="RuntimeError: boom")
    assert Prediction.from_json(failed.to_json()).error == "RuntimeError: boom"


def test_case_json_omits_empty_optional_fields():
    assert Case("c", "q", "a").to_json() == {"id": "c", "input": "q", "expected": "a"}


def test_subset_keeps_dataset_order():
    run = RunResult(
        target="t",
        dataset_name="d",
        dataset_version="1",
        dataset_fingerprint="f",
        case_ids=("a", "b", "c"),
        predictions=(Prediction("a", "x"), Prediction("b", "y"), Prediction("c", "z")),
        scores=(
            CaseScore("a", "s", 1.0),
            CaseScore("b", "s", 0.0),
            CaseScore("c", "s", 0.5),
        ),
        tags={"t1": ("a", "c")},
    )
    subset = run.subset(["c", "a"])
    assert subset.case_ids == ("a", "c")
    assert subset.vector("s") == [1.0, 0.5]
    assert subset.tags == {"t1": ("a", "c")}


def test_run_name_falls_back_to_the_target():
    run = RunResult("my_target", "d", "1", "f", (), (), ())
    assert run.name == "my_target"
    assert RunResult("my_target", "d", "1", "f", (), (), (), label="nightly").name == "nightly"


def test_every_report_renders_without_crashing(baseline_run, candidate_run):
    summaries = summarize(baseline_run, resamples=200)
    text = report.render_summaries(summaries, title="Scores")
    assert "exact_match" in text and "judge" in text
    assert "[" in text  # every row carries its interval

    breakdown = summarize_by_tag(baseline_run, "judge", resamples=200)
    assert "Per-tag judge" in report.render_tag_breakdown(breakdown, scorer="judge")
    assert report.render_tag_breakdown({}, scorer="judge") == ""

    comparison = compare_runs(baseline_run, candidate_run, "judge", resamples=200)
    assert "IMPROVED" in report.render_comparison(comparison)

    gate = evaluate_gate(baseline_run, candidate_run, "judge", resamples=200)
    assert "Regression gate: PASS" in report.render_gate(gate)

    assert "kappa" in report.render_calibration(calibrate(load_labels("builtin:judge_labels")))
    assert baseline_run.dataset_fingerprint in report.render_run_header(baseline_run)


def test_failure_listing_shows_the_worst_cases(baseline_run):
    text = report.render_failures(baseline_run, "judge", limit=3)
    assert "Lowest-scoring cases" in text
    assert len(text.splitlines()) <= 2 + 3 * 2
    assert report.render_failures(baseline_run, "not_a_scorer") == ""

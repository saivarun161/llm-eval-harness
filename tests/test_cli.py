from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalharness.cli import main


def run_cli(capsys, *argv: str) -> tuple[int, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    run_cli.stderr = captured.err
    return code, captured.out


def test_no_command_prints_help(capsys):
    code, out = run_cli(capsys)
    assert code == 1
    assert "usage" in out


def test_version(capsys):
    code, out = run_cli(capsys, "--version")
    assert code == 0
    assert out.strip().count(".") == 2


def test_scorers_listing(capsys):
    code, out = run_cli(capsys, "scorers")
    assert code == 0
    assert "judge" in out and "exact_match" in out

    code, out = run_cli(capsys, "scorers", "--json")
    assert code == 0
    assert "semantic" in json.loads(out)


def test_datasets_listing_shows_the_fingerprint(capsys):
    code, out = run_cli(capsys, "datasets", "--json")
    assert code == 0
    rows = json.loads(out)
    assert rows[0]["name"] == "qa_general"
    assert len(rows[0]["fingerprint"]) == 16
    assert rows[0]["cases"] >= 40


def test_run_writes_a_reusable_result(tmp_path, capsys):
    out_path = tmp_path / "baseline.json"
    code, out = run_cli(
        capsys,
        "run",
        "--target",
        "baseline",
        "--scorers",
        "exact_match,judge",
        "--resamples",
        "200",
        "--out",
        str(out_path),
        "--failures",
    )
    assert code == 0
    assert out_path.exists()
    assert "Lowest-scoring cases" in out
    assert "judge" in out


def test_run_can_slice_by_tag(tmp_path, capsys):
    code, out = run_cli(
        capsys, "run", "--target", "baseline", "--tag", "numeric", "--resamples", "100", "--json"
    )
    assert code == 0
    payload = json.loads(out)
    assert "qa_general[numeric]" in payload["run"]
    assert payload["summaries"][0]["n"] < 45


def test_run_rejects_an_unknown_target(capsys):
    code, _ = run_cli(capsys, "run", "--target", "module_that_does_not_exist:fn")
    assert code == 2


@pytest.fixture
def saved_runs(tmp_path, capsys):
    paths = {}
    for name in ("baseline", "candidate", "regressed"):
        path = tmp_path / f"{name}.json"
        assert (
            main(
                [
                    "run",
                    "--target",
                    name,
                    "--scorers",
                    "exact_match,judge",
                    "--resamples",
                    "100",
                    "--out",
                    str(path),
                ]
            )
            == 0
        )
        paths[name] = str(path)
    capsys.readouterr()
    return paths


def test_compare_reports_both_scorers(saved_runs, capsys):
    code, out = run_cli(
        capsys, "compare", saved_runs["baseline"], saved_runs["candidate"], "--resamples", "300"
    )
    assert code == 0
    assert "exact_match" in out and "judge" in out


def test_compare_json_is_machine_readable(saved_runs, capsys):
    code, out = run_cli(
        capsys,
        "compare",
        saved_runs["baseline"],
        saved_runs["candidate"],
        "--scorers",
        "judge",
        "--resamples",
        "300",
    )
    assert code == 0
    code, out = run_cli(
        capsys,
        "compare",
        saved_runs["baseline"],
        saved_runs["candidate"],
        "--scorers",
        "judge",
        "--resamples",
        "300",
        "--json",
    )
    payload = json.loads(out)
    assert payload[0]["direction"] == "improvement"


def test_compare_refuses_mismatched_datasets(tmp_path, saved_runs, capsys):
    drifted = tmp_path / "drifted.json"
    payload = json.loads(Path(saved_runs["candidate"]).read_text())
    payload["dataset"]["fingerprint"] = "deadbeefdeadbeef"
    drifted.write_text(json.dumps(payload))
    code, _ = run_cli(capsys, "compare", saved_runs["baseline"], str(drifted))
    assert code == 2
    assert "fingerprint changed" in run_cli.stderr


def test_gate_exit_codes(saved_runs, capsys):
    code, out = run_cli(
        capsys,
        "gate",
        "--baseline",
        saved_runs["baseline"],
        "--candidate",
        saved_runs["candidate"],
        "--tolerance",
        "0.02",
        "--resamples",
        "400",
    )
    assert code == 0
    assert "Regression gate: PASS" in out

    code, out = run_cli(
        capsys,
        "gate",
        "--baseline",
        saved_runs["baseline"],
        "--candidate",
        saved_runs["regressed"],
        "--tolerance",
        "0.02",
        "--resamples",
        "400",
    )
    assert code == 1
    assert "Regression gate: FAIL" in out


def test_gate_json_and_power_check(saved_runs, capsys):
    code, out = run_cli(
        capsys,
        "gate",
        "--baseline",
        saved_runs["baseline"],
        "--candidate",
        saved_runs["candidate"],
        "--require-mde",
        "0.001",
        "--resamples",
        "400",
        "--json",
    )
    assert code == 1
    payload = json.loads(out)
    assert payload["underpowered"] is True
    assert payload["passed"] is False


def test_gate_rejects_an_unknown_scorer(saved_runs, capsys):
    code, _ = run_cli(
        capsys,
        "gate",
        "--baseline",
        saved_runs["baseline"],
        "--candidate",
        saved_runs["candidate"],
        "--scorer",
        "not_a_scorer",
    )
    assert code == 2


def test_calibrate_reports_kappa(capsys):
    code, out = run_cli(capsys, "calibrate", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["kappa_linear"] >= 0.6
    assert payload["n"] >= 60


def test_calibrate_can_fail_on_low_agreement(capsys):
    code, _ = run_cli(capsys, "calibrate", "--min-kappa", "0.99")
    assert code == 1


def test_calibrate_accepts_fixed_thresholds(capsys):
    code, out = run_cli(capsys, "calibrate", "--thresholds", "0.3,0.8", "--json")
    assert code == 0
    assert json.loads(out)["thresholds"] == [0.3, 0.8]


def test_calibrate_reports_a_missing_label_file(capsys):
    code, _ = run_cli(capsys, "calibrate", "--labels", "builtin:nothing")
    assert code == 2


def test_demo_runs_end_to_end(capsys):
    code, out = run_cli(capsys, "demo", "--resamples", "300")
    assert code == 0
    for marker in (
        "Dataset qa_general",
        "Regression gate: FAIL",
        "Regression gate: PASS",
        "Cohen's kappa",
        "INCONCLUSIVE",
    ):
        assert marker in out, marker

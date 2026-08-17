from __future__ import annotations

import pytest

from evalharness.calibration import (
    LabelledExample,
    calibrate,
    load_labels,
    parse_labels,
    search_thresholds,
)
from evalharness.scorers.judge import DEFAULT_THRESHOLDS, JudgeVerdict


@pytest.fixture
def bundled_labels() -> list[LabelledExample]:
    return load_labels("builtin:judge_labels")


def test_bundled_labels_are_well_formed(bundled_labels):
    assert len(bundled_labels) >= 60
    assert {ex.label for ex in bundled_labels} == {0, 1, 2}
    for example in bundled_labels:
        assert example.expected and example.prediction


def test_labels_cover_every_grade_reasonably(bundled_labels):
    # A label set that is 95% one class produces a flattering raw agreement and
    # a meaningless kappa, so the balance is worth asserting on.
    for label in (0, 1, 2):
        share = sum(1 for ex in bundled_labels if ex.label == label) / len(bundled_labels)
        assert 0.15 < share < 0.55, label


def test_the_shipped_judge_reaches_substantial_agreement(bundled_labels):
    calibration = calibrate(bundled_labels)
    assert calibration.n == len(bundled_labels)
    assert calibration.kappa >= 0.60, f"judge agreement dropped to {calibration.kappa:.3f}"
    assert calibration.interpretation == "substantial"
    assert calibration.trustworthy
    assert calibration.kappa_interval.lo < calibration.kappa < calibration.kappa_interval.hi


def test_shipped_thresholds_match_what_the_search_finds(bundled_labels):
    # The defaults in the judge module are the output of this search. If the
    # judge changes and they drift apart, this test says so.
    calibration = calibrate(bundled_labels)
    assert calibration.thresholds == pytest.approx(DEFAULT_THRESHOLDS, abs=0.02)
    assert calibration.kappa_at_default == pytest.approx(calibration.kappa, abs=0.02)


def test_judge_scores_correlate_with_human_grades(bundled_labels):
    calibration = calibrate(bundled_labels)
    assert calibration.spearman > 0.6
    assert calibration.pearson > 0.6
    assert 0.0 <= calibration.agreement <= 1.0


def test_confusion_matrix_totals_match_the_label_counts(bundled_labels):
    calibration = calibrate(bundled_labels)
    assert sum(sum(row) for row in calibration.confusion) == calibration.n
    for label in (0, 1, 2):
        assert sum(calibration.confusion[label]) == calibration.label_counts[label]


def test_a_perfect_judge_calibrates_to_kappa_one():
    examples = [
        LabelledExample(f"e{i}", "q", "reference", f"answer graded {label}", label)
        for i, label in enumerate([0, 1, 2] * 8)
    ]
    scores = {"0": 0.05, "1": 0.45, "2": 0.95}

    def oracle(question: str, expected: str, prediction: str) -> JudgeVerdict:
        return JudgeVerdict(scores[prediction[-1]], "oracle")

    calibration = calibrate(examples, judge=oracle)
    assert calibration.kappa == pytest.approx(1.0)
    assert calibration.agreement == pytest.approx(1.0)


def test_a_constant_judge_gets_no_credit():
    examples = [
        LabelledExample(f"e{i}", "q", "reference", "answer", label)
        for i, label in enumerate([2] * 16 + [0] * 4)
    ]
    calibration = calibrate(
        examples, judge=lambda q, e, p: JudgeVerdict(0.99, "always correct"), search=False
    )
    assert calibration.agreement == pytest.approx(0.8)
    assert calibration.kappa == pytest.approx(0.0)
    assert not calibration.trustworthy


def test_threshold_search_beats_a_bad_fixed_choice():
    scores = [0.1, 0.15, 0.5, 0.55, 0.9, 0.95]
    labels = [0, 0, 1, 1, 2, 2]
    (low, high), kappa = search_thresholds(scores, labels)
    assert 0.15 < low < 0.5
    assert 0.55 < high < 0.9
    assert kappa == pytest.approx(1.0)


def test_explicit_thresholds_win_over_the_search(bundled_labels):
    calibration = calibrate(bundled_labels, thresholds=(0.1, 0.2))
    assert calibration.thresholds == (0.1, 0.2)


def test_label_parsing_validates_the_scale():
    with pytest.raises(ValueError, match="expected 0, 1 or 2"):
        parse_labels('{"id": "x", "expected": "a", "prediction": "b", "label": 5}')
    with pytest.raises(ValueError, match="line 1"):
        parse_labels('{"id": "x", "expected": "a"}')
    with pytest.raises(ValueError, match="no examples"):
        parse_labels("\n")


def test_missing_label_files_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labels("builtin:not_a_label_set")
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path / "absent.jsonl")


def test_calibration_of_an_empty_set_is_an_error():
    with pytest.raises(ValueError, match="empty label set"):
        calibrate([])

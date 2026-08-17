from __future__ import annotations

import math

import pytest

from evalharness.stats import (
    Interval,
    agreement_rate,
    binomial_two_sided_p,
    bootstrap_mean,
    bootstrap_paired_delta,
    cohens_kappa,
    kappa_interpretation,
    mean,
    pearson,
    percentile,
    spearman,
    stdev,
)


def test_percentile_interpolates():
    values = [0.0, 1.0, 2.0, 3.0]
    assert percentile(values, 0.0) == 0.0
    assert percentile(values, 1.0) == 3.0
    assert percentile(values, 0.5) == pytest.approx(1.5)


def test_mean_and_stdev():
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert stdev([1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert stdev([4.0]) == 0.0
    with pytest.raises(ValueError):
        mean([])


def test_bootstrap_mean_brackets_the_point_estimate():
    values = [1.0, 0.0] * 30
    interval = bootstrap_mean(values)
    assert interval.point == pytest.approx(0.5)
    assert interval.lo < 0.5 < interval.hi
    assert interval.confidence == 0.95


def test_bootstrap_is_reproducible_and_seed_sensitive():
    values = [0.2, 0.9, 0.4, 0.7, 0.1, 0.8, 0.55, 0.3]
    assert bootstrap_mean(values, seed=7) == bootstrap_mean(values, seed=7)
    assert bootstrap_mean(values, seed=7) != bootstrap_mean(values, seed=8)


def test_degenerate_sample_gets_a_zero_width_interval():
    interval = bootstrap_mean([0.4] * 10)
    assert (interval.lo, interval.point, interval.hi) == (0.4, 0.4, 0.4)
    assert interval.half_width == 0.0


def test_more_cases_shrink_the_interval():
    small = bootstrap_mean([1.0, 0.0] * 10)
    large = bootstrap_mean([1.0, 0.0] * 200)
    assert large.half_width < small.half_width


def test_paired_delta_is_tighter_than_unpaired_for_correlated_runs():
    # Shared per-case difficulty plus a constant improvement: pairing should
    # cancel the difficulty entirely and leave a very tight interval.
    difficulty = [(i % 7) / 7.0 for i in range(60)]
    baseline = difficulty
    candidate = [min(1.0, d + 0.1) for d in difficulty]
    paired = bootstrap_paired_delta(candidate, baseline)
    assert paired.point == pytest.approx(mean(candidate) - mean(baseline))
    assert paired.excludes_zero
    unpaired_width = bootstrap_mean(candidate).half_width + bootstrap_mean(baseline).half_width
    assert paired.half_width < unpaired_width


def test_paired_delta_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal-length"):
        bootstrap_paired_delta([0.1, 0.2], [0.1])


def test_identical_runs_produce_a_zero_delta():
    values = [0.1, 0.5, 0.9, 0.3]
    interval = bootstrap_paired_delta(values, values)
    assert (interval.lo, interval.point, interval.hi) == (0.0, 0.0, 0.0)
    assert not interval.excludes_zero


def test_binomial_two_sided_p_known_values():
    assert binomial_two_sided_p(0, 5) == pytest.approx(2 * (0.5**5))
    assert binomial_two_sided_p(10, 10) == pytest.approx(2 * (0.5**10))
    assert binomial_two_sided_p(5, 10) == pytest.approx(1.0)
    assert binomial_two_sided_p(0, 0) == 1.0


def test_binomial_rejects_impossible_counts():
    with pytest.raises(ValueError):
        binomial_two_sided_p(6, 5)


def test_kappa_known_values():
    assert cohens_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)
    # Marginals identical, agreement exactly at chance.
    assert cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)
    # po = 0.75, pe = 0.5 -> kappa = 0.5
    assert cohens_kappa([0, 0, 0, 1], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_kappa_punishes_the_always_correct_judge():
    # 8 of 10 truly correct; a judge that says "correct" every time gets 80%
    # raw agreement and no credit at all.
    human = [1] * 8 + [0] * 2
    judge = [1] * 10
    assert agreement_rate(human, judge) == pytest.approx(0.8)
    assert cohens_kappa(human, judge) == pytest.approx(0.0)


def test_weighted_kappa_forgives_near_misses():
    human = [0, 1, 2, 2, 1, 0]
    near = [0, 2, 2, 2, 1, 0]  # one adjacent slip
    far = [2, 1, 2, 2, 1, 0]  # one opposite-end slip
    assert cohens_kappa(human, near, weights="linear") > cohens_kappa(human, far, weights="linear")
    assert cohens_kappa(human, far, weights="quadratic") < cohens_kappa(
        human, far, weights="linear"
    )


def test_kappa_with_a_single_label_is_zero_not_one():
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 0.0


def test_kappa_input_validation():
    with pytest.raises(ValueError, match="equal-length"):
        cohens_kappa([0, 1], [0])
    with pytest.raises(ValueError, match="empty"):
        cohens_kappa([], [])
    with pytest.raises(ValueError, match="weighting"):
        cohens_kappa([0, 1], [1, 0], weights="cubic")


def test_correlations():
    x = [1.0, 2.0, 3.0, 4.0]
    assert pearson(x, [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)
    assert pearson(x, [8.0, 6.0, 4.0, 2.0]) == pytest.approx(-1.0)
    assert pearson(x, [1.0, 1.0, 1.0, 1.0]) == 0.0
    # Monotone but non-linear: Spearman sees the order, Pearson does not fully.
    y = [1.0, 4.0, 9.0, 16.0]
    assert spearman(x, y) == pytest.approx(1.0)
    assert pearson(x, y) < 1.0


def test_spearman_handles_ties():
    assert spearman([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1.0)


def test_interval_helpers():
    interval = Interval(0.02, -0.08, 0.12)
    assert interval.half_width == pytest.approx(0.1)
    assert not interval.excludes_zero
    assert Interval(0.5, 0.1, 0.9).excludes_zero
    assert Interval(-0.5, -0.9, -0.1).excludes_zero
    assert "0.020" in interval.format()
    assert math.isclose(interval.to_json()["lo"], -0.08)


def test_kappa_interpretation_bands():
    assert kappa_interpretation(-0.1) == "worse than chance"
    assert kappa_interpretation(0.1) == "slight"
    assert kappa_interpretation(0.5) == "moderate"
    assert kappa_interpretation(0.7) == "substantial"
    assert kappa_interpretation(0.95) == "almost perfect"


def test_confidence_bounds_are_validated():
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean([0.1, 0.9], confidence=1.5)

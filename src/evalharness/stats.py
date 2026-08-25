"""Resampling statistics and agreement measures.

Two questions run through the whole harness, and both are answered here:

1. *Is this score difference bigger than the noise?* — answered by a paired
   percentile bootstrap over cases.
2. *Does this judge agree with a human?* — answered by Cohen's kappa, which
   discounts the agreement you would get by chance.

Everything is seeded and pure-Python: two runs of the same command produce
byte-identical intervals, which is what makes the CI gate reproducible.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 20260817


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    point: float
    lo: float
    hi: float
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def half_width(self) -> float:
        """Half the interval width — the smallest effect this run could detect."""
        return (self.hi - self.lo) / 2.0

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def format(self, digits: int = 3) -> str:
        return f"{self.point:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"

    def to_json(self) -> dict[str, float]:
        return {
            "point": round(self.point, 6),
            "lo": round(self.lo, 6),
            "hi": round(self.hi, 6),
            "confidence": self.confidence,
        }


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence")
    return sum(values) / len(values)


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _tail_bounds(confidence: float) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    alpha = (1.0 - confidence) / 2.0
    return alpha, 1.0 - alpha


def bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Percentile bootstrap CI for the mean of per-case scores."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    point = mean(values)
    if len(values) == 1 or all(v == values[0] for v in values):
        # A degenerate sample has no spread to resample; report a zero-width
        # interval rather than a misleadingly narrow one built from noise.
        return Interval(point=point, lo=point, hi=point, confidence=confidence)
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    lo_q, hi_q = _tail_bounds(confidence)
    return Interval(
        point=point,
        lo=percentile(draws, lo_q),
        hi=percentile(draws, hi_q),
        confidence=confidence,
    )


def bootstrap_paired_delta(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """CI for ``mean(candidate) - mean(baseline)`` over the same cases.

    Case indices are resampled once and applied to *both* runs, so the
    per-case difficulty that both systems share cancels out. Pairing is the
    whole reason this interval is usually far tighter than the difference of
    two independent intervals, and it is why comparing runs from different
    dataset fingerprints is rejected upstream.
    """
    if len(candidate) != len(baseline):
        raise ValueError(
            "paired comparison needs equal-length samples, got "
            f"{len(candidate)} and {len(baseline)}"
        )
    if not candidate:
        raise ValueError("cannot bootstrap an empty sample")
    diffs = [c - b for c, b in zip(candidate, baseline, strict=True)]
    point = mean(diffs)
    if all(d == diffs[0] for d in diffs):
        return Interval(point=point, lo=point, hi=point, confidence=confidence)
    rng = random.Random(seed)
    n = len(diffs)
    draws = sorted(mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    lo_q, hi_q = _tail_bounds(confidence)
    return Interval(
        point=point,
        lo=percentile(draws, lo_q),
        hi=percentile(draws, hi_q),
        confidence=confidence,
    )


def binomial_two_sided_p(successes: int, trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test, used for the win/loss sign test.

    Sums the probability of every outcome no more likely than the observed one,
    which is the standard exact construction and needs no normal approximation.
    """
    if trials <= 0:
        return 1.0
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be within [0, {trials}], got {successes}")

    def pmf(k: int) -> float:
        return math.comb(trials, k) * (p**k) * ((1 - p) ** (trials - k))

    observed = pmf(successes)
    # Floating-point slack so that outcomes with mathematically equal
    # probability (the symmetric partner of the observed one) are not dropped.
    tolerance = observed * 1e-9
    total = sum(pmf(k) for k in range(trials + 1) if pmf(k) <= observed + tolerance)
    return min(1.0, total)


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, returned in input order.

    Testing eight slices at 5% gives a one-in-three chance of at least one
    false alarm, and a per-slice report is exactly the place that happens.
    Holm controls the family-wise error rate under *arbitrary* dependence
    between the tests, which matters here because slices share cases: a case
    tagged both ``numeric`` and ``units`` is in two of the tests at once.

    Adjusted values are forced non-decreasing along the sorted order, so a
    p-value can never be reported as more significant than a smaller one.
    """
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {p!r}")
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def bonferroni_confidence(confidence: float, tests: int) -> float:
    """The per-test confidence level that holds ``confidence`` over ``tests``.

    Bonferroni rather than Šidák: Šidák is exact only when the tests are
    independent, and overlapping tag slices are anything but. Boole's
    inequality needs no independence assumption at all, and the cost — a
    slightly wider interval — is the right side to err on for a report whose
    whole job is to not cry wolf.
    """
    if tests <= 1:
        return confidence
    alpha = 1.0 - confidence
    return 1.0 - alpha / tests


def cohens_kappa(
    a: Sequence[int],
    b: Sequence[int],
    *,
    weights: str | None = None,
) -> float:
    """Cohen's kappa between two label sequences.

    ``weights`` may be ``None`` (nominal), ``"linear"`` or ``"quadratic"``.
    Weighted variants matter for ordinal rubrics: on a wrong/partial/correct
    scale, calling a *partial* answer *correct* is a smaller mistake than
    calling a *wrong* answer *correct*, and unweighted kappa cannot express
    that.
    """
    if len(a) != len(b):
        raise ValueError(f"kappa needs equal-length sequences, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("kappa of an empty sample")
    if weights not in (None, "linear", "quadratic"):
        raise ValueError(f"unknown kappa weighting {weights!r}")

    labels = sorted(set(a) | set(b))
    if len(labels) == 1:
        # Both raters used a single label. They agree completely, but there is
        # no variance for chance correction to work with; kappa is undefined
        # and 0.0 is the conventional, conservative answer.
        return 0.0
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    n = len(a)

    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1.0

    rows = [sum(row) for row in observed]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        distance = abs(labels[i] - labels[j]) / (labels[-1] - labels[0])
        return distance if weights == "linear" else distance**2

    disagreement_o = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k)) / n
    disagreement_e = sum(weight(i, j) * rows[i] * cols[j] for i in range(k) for j in range(k)) / (
        n * n
    )
    if disagreement_e == 0.0:
        return 1.0 if disagreement_o == 0.0 else 0.0
    return 1.0 - disagreement_o / disagreement_e


def agreement_rate(a: Sequence[int], b: Sequence[int]) -> float:
    """Fraction of items where two raters used the identical label."""
    if len(a) != len(b):
        raise ValueError("agreement needs equal-length sequences")
    if not a:
        raise ValueError("agreement of an empty sample")
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("correlation needs equal-length sequences")
    if len(x) < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denom = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return 0.0 if denom == 0.0 else num / denom


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0  # ties share the mean of their rank block
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation — the right measure when a judge's scale is arbitrary."""
    return pearson(_ranks(x), _ranks(y))


def kappa_interpretation(kappa: float) -> str:
    """Landis & Koch's conventional bands, kept honest at the low end."""
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"

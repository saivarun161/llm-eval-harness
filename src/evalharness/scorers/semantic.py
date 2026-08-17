"""Reference similarity that survives paraphrase.

Backed by the local hashing embedder in :mod:`evalharness.textutil`, so it needs
no model download and returns the same number on every machine. It reads
paraphrase and verbosity far better than exact match and materially worse than a
real sentence encoder — the README says so plainly, and the calibration command
is there to quantify how much that costs on your data rather than leaving it to
faith.
"""

from __future__ import annotations

from ..textutil import cosine, coverage, similarity
from ..types import Case
from .base import Verdict, register


@register("semantic", "Blended cosine and coverage over local hashed embeddings")
def semantic(case: Case, output: str) -> Verdict:
    """Graded similarity between the output and the reference answer."""
    score = similarity(output, case.expected)
    direct = max(0.0, cosine(output, case.expected))
    contained = coverage(output, case.expected)
    return Verdict(score, f"cosine {direct:.2f}, coverage {contained:.2f}")


@register("semantic_pass", "Semantic similarity thresholded into a pass or fail")
def semantic_pass(case: Case, output: str) -> Verdict:
    """Binary view of :func:`semantic`, for pass-rate style reporting.

    The threshold comes from ``case.metadata['semantic_threshold']`` when the
    case sets one, so a dataset can hold stricter cases next to looser ones.
    """
    threshold = float(case.metadata.get("semantic_threshold", 0.62))
    score = similarity(output, case.expected)
    passed = score >= threshold
    return Verdict(
        1.0 if passed else 0.0,
        f"similarity {score:.2f} {'>=' if passed else '<'} threshold {threshold:.2f}",
    )

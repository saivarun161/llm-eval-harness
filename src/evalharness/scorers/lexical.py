"""Surface-level scorers: exact match, containment, token F1 and fuzzy ratio.

These are cheap, transparent and completely deterministic. They are also the
scorers most likely to be *wrong about a correct answer*, which is the point of
running them alongside a judge: when exact match and the judge disagree by
fifteen points, the disagreement itself is the finding.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..textutil import normalize, tokens
from ..types import Case
from .base import Verdict, register


@register("exact_match", "Normalised string equality")
def exact_match(case: Case, output: str) -> Verdict:
    """1.0 when the normalised strings are identical."""
    got, want = normalize(output), normalize(case.expected)
    if got == want:
        return Verdict(1.0, "exact match after normalisation")
    return Verdict(0.0, f"expected {want!r}, got {got!r}")


@register("contains", "Reference appears verbatim inside the output")
def contains(case: Case, output: str) -> Verdict:
    """1.0 when the normalised reference is a substring of the output."""
    got, want = normalize(output), normalize(case.expected)
    if want and want in got:
        return Verdict(1.0, "reference found in output")
    return Verdict(0.0, f"reference {want!r} not present")


@register("token_f1", "Harmonic mean of token precision and recall")
def token_f1(case: Case, output: str) -> Verdict:
    """Bag-of-tokens F1 — forgiving about word order, strict about content."""
    got, want = tokens(output), tokens(case.expected)
    if not want:
        return Verdict(1.0 if not got else 0.0, "empty reference")
    if not got:
        return Verdict(0.0, "empty output")
    overlap = 0
    remaining = list(want)
    for token in got:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return Verdict(0.0, "no shared tokens")
    precision = overlap / len(got)
    recall = overlap / len(want)
    f1 = 2 * precision * recall / (precision + recall)
    return Verdict(f1, f"precision {precision:.2f}, recall {recall:.2f}")


@register("fuzzy", "Character-level similarity ratio")
def fuzzy(case: Case, output: str) -> Verdict:
    """Longest-matching-block similarity, the classic edit-distance stand-in."""
    got, want = normalize(output), normalize(case.expected)
    if not want:
        return Verdict(1.0 if not got else 0.0, "empty reference")
    ratio = SequenceMatcher(None, got, want).ratio()
    return Verdict(ratio, f"ratio {ratio:.2f}")


@register("regex", "Output matches the pattern in case metadata")
def regex(case: Case, output: str) -> Verdict:
    """Matches ``case.metadata['pattern']``; falls back to the reference.

    Useful for format assertions — "answer with a bare integer" — where the
    reference text is the wrong thing to compare against.
    """
    pattern = case.metadata.get("pattern")
    if not pattern:
        pattern = re.escape(case.expected)
    try:
        compiled = re.compile(str(pattern), re.IGNORECASE)
    except re.error as exc:
        return Verdict(0.0, f"invalid pattern for case {case.id}: {exc}")
    if compiled.search(output):
        return Verdict(1.0, "pattern matched")
    return Verdict(0.0, f"pattern {pattern!r} did not match")

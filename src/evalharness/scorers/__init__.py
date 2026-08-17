"""Built-in scorers.

Importing this package registers every bundled scorer. Third-party scorers join
the same registry with :func:`evalharness.scorers.register`, and from that point
the runner, the reports and the CI gate treat them identically to the built-ins.
"""

from __future__ import annotations

from .base import Scorer, Verdict, available, describe, get_scorer, register, unregister
from .judge import (
    DEFAULT_THRESHOLDS,
    HeuristicJudge,
    JudgeVerdict,
    ModelJudge,
    Rubric,
    get_judge,
    label_from_score,
    reset_judge_cache,
)
from .lexical import contains, exact_match, fuzzy, regex, token_f1
from .semantic import semantic, semantic_pass

#: A sensible default set: one strict, two graded, one judged.
DEFAULT_SCORERS = ("exact_match", "token_f1", "semantic", "judge")

__all__ = [
    "DEFAULT_SCORERS",
    "DEFAULT_THRESHOLDS",
    "HeuristicJudge",
    "JudgeVerdict",
    "ModelJudge",
    "Rubric",
    "Scorer",
    "Verdict",
    "available",
    "contains",
    "describe",
    "exact_match",
    "fuzzy",
    "get_judge",
    "get_scorer",
    "label_from_score",
    "regex",
    "register",
    "reset_judge_cache",
    "semantic",
    "semantic_pass",
    "token_f1",
    "unregister",
]

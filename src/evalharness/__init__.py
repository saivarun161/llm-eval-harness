"""An evaluation harness for LLM applications.

The organising idea is that an evaluation score is an estimate, not a
measurement. Two runs of the same system on the same eval set differ; two
different systems differ by some amount that may or may not exceed that noise.
Everything here — the paired bootstrap, the judge calibration, the CI gate —
exists to keep that distinction visible instead of rounding it away into a
single number on a dashboard.
"""

from __future__ import annotations

from .calibration import Calibration, LabelledExample, calibrate, load_labels
from .compare import Comparison, DatasetMismatch, compare_all, compare_runs
from .dataset import Dataset, list_builtins, load_dataset
from .gate import GateResult, evaluate_gate
from .runner import ScoreSummary, evaluate, summarize, summarize_by_tag
from .scorers import DEFAULT_SCORERS, Verdict, available, get_scorer, register
from .slices import SliceComparison, SliceReport, compare_by_tag
from .stats import (
    Interval,
    bootstrap_mean,
    bootstrap_paired_delta,
    cohens_kappa,
    holm_adjusted,
)
from .store import load_run, save_run
from .targets import PROFILES, resolve_target
from .types import Case, CaseScore, Prediction, RunResult

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SCORERS",
    "PROFILES",
    "Calibration",
    "Case",
    "CaseScore",
    "Comparison",
    "Dataset",
    "DatasetMismatch",
    "GateResult",
    "Interval",
    "LabelledExample",
    "Prediction",
    "RunResult",
    "ScoreSummary",
    "SliceComparison",
    "SliceReport",
    "Verdict",
    "__version__",
    "available",
    "bootstrap_mean",
    "bootstrap_paired_delta",
    "calibrate",
    "cohens_kappa",
    "compare_all",
    "compare_by_tag",
    "compare_runs",
    "evaluate",
    "evaluate_gate",
    "get_scorer",
    "holm_adjusted",
    "list_builtins",
    "load_dataset",
    "load_labels",
    "load_run",
    "register",
    "resolve_target",
    "save_run",
    "summarize",
    "summarize_by_tag",
]

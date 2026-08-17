from __future__ import annotations

import pytest

from evalharness.dataset import Dataset, load_dataset
from evalharness.runner import evaluate
from evalharness.targets import resolve_target
from evalharness.types import Case


@pytest.fixture
def tiny_dataset() -> Dataset:
    return Dataset(
        name="tiny",
        version="1.0.0",
        cases=(
            Case("c1", "What is the capital of France?", "Paris", ("geography",)),
            Case("c2", "What is 2 + 2?", "4", ("math",)),
            Case("c3", "Who wrote Hamlet?", "William Shakespeare", ("literature",)),
            Case("c4", "What colour is the sky on a clear day?", "Blue", ("science",)),
        ),
    )


@pytest.fixture
def bundled() -> Dataset:
    return load_dataset("builtin:qa_general")


@pytest.fixture
def baseline_run(bundled: Dataset):
    target = resolve_target("baseline", bundled.cases)
    return evaluate(bundled, target, scorers=["exact_match", "judge"], target_name="baseline")


@pytest.fixture
def candidate_run(bundled: Dataset):
    target = resolve_target("candidate", bundled.cases)
    return evaluate(bundled, target, scorers=["exact_match", "judge"], target_name="candidate")


@pytest.fixture
def regressed_run(bundled: Dataset):
    target = resolve_target("regressed", bundled.cases)
    return evaluate(bundled, target, scorers=["exact_match", "judge"], target_name="regressed")

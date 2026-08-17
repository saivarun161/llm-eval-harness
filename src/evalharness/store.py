"""Persisting run results.

Runs are plain JSON on purpose. The baseline a CI gate compares against is
typically committed to the repository or pulled from an artefact store, and it
needs to survive a library upgrade, be diffable in a pull request, and be
readable by whoever is arguing about the number six months from now.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import RunResult


def save_run(run: RunResult, path: str | Path) -> Path:
    """Write ``run`` to ``path``, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(run.to_json(), indent=2, sort_keys=False, ensure_ascii=False)
    target.write_text(payload + "\n", encoding="utf-8")
    return target


def load_run(path: str | Path) -> RunResult:
    """Read a run written by :func:`save_run`."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no run file at {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc.msg}") from exc
    try:
        return RunResult.from_json(payload)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{source} is not a valid run file: {exc}") from exc

"""The scorer contract and its registry.

A scorer is any callable that turns one (case, output) pair into a number in
``[0, 1]`` plus a human-readable reason. Keeping the interface this narrow is
what lets the statistics layer stay ignorant of scoring: exact match, a fuzzy
ratio and an LLM judge all arrive at the bootstrap as the same kind of vector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..types import Case


@dataclass(frozen=True)
class Verdict:
    """One scorer's opinion about one output."""

    score: float
    detail: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score!r}")


class Scorer(Protocol):
    """Callable protocol implemented by every scorer."""

    def __call__(self, case: Case, output: str) -> Verdict: ...


_REGISTRY: dict[str, Scorer] = {}
_DESCRIPTIONS: dict[str, str] = {}


def register(name: str, description: str = "") -> Callable[[Scorer], Scorer]:
    """Decorator registering a scorer under ``name``."""

    def decorate(scorer: Scorer) -> Scorer:
        if name in _REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        _REGISTRY[name] = scorer
        _DESCRIPTIONS[name] = description or (scorer.__doc__ or "").strip().split("\n")[0]
        return scorer

    return decorate


def get_scorer(name: str) -> Scorer:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown scorer {name!r}; registered scorers: {', '.join(available())}"
        ) from None


def available() -> list[str]:
    return sorted(_REGISTRY)


def describe(name: str) -> str:
    return _DESCRIPTIONS.get(name, "")


def unregister(name: str) -> None:
    """Remove a scorer. Exists so tests can register throwaway scorers cleanly."""
    _REGISTRY.pop(name, None)
    _DESCRIPTIONS.pop(name, None)

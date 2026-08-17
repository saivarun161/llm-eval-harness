"""Systems under test.

A target is any ``Callable[[str], str]``: give it the case input, get back the
answer. Real ones are resolved from ``module:function`` so the harness never
needs to know what is behind them.

The bundled ones are simulated on purpose. A demo that calls a real model would
need a key, a network and a budget, and it would produce different numbers every
run — which would make it useless for showing that the *statistics* are correct.
These stand-ins instead reproduce the failure modes that make evaluation hard:

``verbatim``    the reference answer
``verbose``     correct, wrapped in prose — exact match hates it, the judge does not
``paraphrase``  correct, worded differently — lexical scorers lose it
``truncate``    the first part of the answer only — partially correct
``wrong``       a confident, plausible, incorrect answer
``refuse``      a non-answer

Every profile draws from one uniform value per case, so a case that is hard for
one system is hard for all of them. That shared difficulty is exactly the
structure the paired bootstrap exploits, and it is what makes the demo's
comparison intervals tighter than the two runs' own intervals.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass

from .types import Case

Target = Callable[[str], str]

#: Behaviours ordered best to worst; profiles differ only in where they cut.
BEHAVIOURS = ("verbatim", "verbose", "paraphrase", "truncate", "wrong", "refuse")

_FALLBACK_WRONG = "I believe the answer is roughly forty-two."
_VERBOSE_TEMPLATE = "Based on what I know, the answer to that is {answer}. Hope that helps!"


def stable_uniform(key: str) -> float:
    """A reproducible float in ``[0, 1)`` derived from ``key``.

    ``random.Random(key)`` would also be reproducible, but hashing keeps every
    case independent of iteration order, so adding a case to the dataset does
    not change the behaviour of the cases around it.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


@dataclass(frozen=True)
class Profile:
    """A simulated system: a name, behaviour weights, and an optional style."""

    name: str
    weights: tuple[float, ...]
    always_verbose: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if len(self.weights) != len(BEHAVIOURS):
            raise ValueError(f"profile {self.name!r} needs {len(BEHAVIOURS)} weights")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"profile {self.name!r} weights sum to {total}, expected 1.0")

    def behaviour_for(self, case_id: str) -> str:
        u = stable_uniform(case_id)
        cumulative = 0.0
        for behaviour, weight in zip(BEHAVIOURS, self.weights, strict=True):
            cumulative += weight
            if u < cumulative:
                return behaviour
        return BEHAVIOURS[-1]


PROFILES: dict[str, Profile] = {
    "baseline": Profile(
        name="baseline",
        weights=(0.45, 0.16, 0.15, 0.08, 0.14, 0.02),
        description="the system already in production",
    ),
    "candidate": Profile(
        name="candidate",
        weights=(0.60, 0.14, 0.13, 0.05, 0.07, 0.01),
        description="a genuinely better system",
    ),
    "tweaked": Profile(
        name="tweaked",
        weights=(0.45, 0.16, 0.15, 0.16, 0.06, 0.02),
        description="a barely-different system, well inside the noise floor",
    ),
    "verbose": Profile(
        name="verbose",
        weights=(0.45, 0.16, 0.15, 0.08, 0.14, 0.02),
        always_verbose=True,
        description="same accuracy as the baseline, chattier prose",
    ),
    "regressed": Profile(
        name="regressed",
        weights=(0.26, 0.13, 0.11, 0.12, 0.32, 0.06),
        description="a real quality regression",
    ),
}


def render(case: Case, behaviour: str, *, always_verbose: bool = False) -> str:
    """Turn a behaviour into the text a system would have produced."""
    expected = case.expected
    if behaviour == "refuse":
        return "I don't know."
    if behaviour == "wrong":
        answer = str(case.metadata.get("wrong") or _FALLBACK_WRONG)
    elif behaviour == "paraphrase":
        answer = str(case.metadata.get("paraphrase") or expected)
    elif behaviour == "truncate":
        words = expected.split()
        keep = max(1, int(len(words) * 0.6))
        answer = " ".join(words[:keep])
    else:
        answer = expected
    if behaviour == "verbose" or (always_verbose and behaviour == "verbatim"):
        return _VERBOSE_TEMPLATE.format(answer=answer)
    return answer


def builtin_target(profile_name: str, cases: list[Case] | tuple[Case, ...]) -> Target:
    """Build a callable simulating ``profile_name`` over the given cases."""
    try:
        profile = PROFILES[profile_name]
    except KeyError:
        raise KeyError(
            f"unknown built-in target {profile_name!r}; available: {', '.join(sorted(PROFILES))}"
        ) from None
    by_input = {case.input: case for case in cases}

    def run(question: str) -> str:
        case = by_input.get(question)
        if case is None:
            raise KeyError(f"built-in target received an unknown question: {question!r}")
        behaviour = profile.behaviour_for(case.id)
        return render(case, behaviour, always_verbose=profile.always_verbose)

    run.__name__ = f"builtin_{profile_name}"
    return run


def load_target(spec: str) -> Target:
    """Import a target from ``module:function``."""
    if ":" not in spec:
        raise ValueError(
            f"target {spec!r} is neither a built-in ({', '.join(sorted(PROFILES))}) "
            "nor a 'module:function' path"
        )
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"could not import {module_name!r} for target {spec!r}: {exc}") from exc
    try:
        target = getattr(module, attribute)
    except AttributeError:
        raise AttributeError(f"{module_name!r} has no attribute {attribute!r}") from None
    if not callable(target):
        raise TypeError(f"target {spec!r} is not callable")
    return target


def resolve_target(spec: str, cases: list[Case] | tuple[Case, ...]) -> Target:
    """Resolve a target spec to a callable, preferring built-ins."""
    if spec in PROFILES:
        return builtin_target(spec, cases)
    return load_target(spec)

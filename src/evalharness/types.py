"""Core value objects passed between the dataset, the scorers and the reports.

Everything here is a frozen dataclass with a plain-JSON representation. The
harness writes run results to disk and reads them back in a later CI job, so the
wire format is part of the contract and lives next to the types it describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Bumped whenever the on-disk run format changes incompatibly.
RUN_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Case:
    """A single evaluation example."""

    id: str
    input: str
    expected: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("case id must not be empty")
        if not self.input:
            raise ValueError(f"case {self.id!r} has an empty input")

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
        }
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Case:
        return cls(
            id=str(payload["id"]),
            input=str(payload["input"]),
            expected=str(payload.get("expected", "")),
            tags=tuple(payload.get("tags", ())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class Prediction:
    """What the system under test produced for one case."""

    case_id: str
    output: str
    latency_ms: float = 0.0
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "output": self.output,
            "latency_ms": round(self.latency_ms, 3),
        }
        if self.error:
            payload["error"] = self.error
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Prediction:
        return cls(
            case_id=str(payload["case_id"]),
            output=str(payload.get("output", "")),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            error=payload.get("error"),
        )


@dataclass(frozen=True)
class CaseScore:
    """One scorer's verdict on one prediction.

    ``score`` is always in ``[0, 1]``. Scorers that are naturally binary emit
    0.0 or 1.0 so that a mean over cases is a pass rate, and scorers that are
    naturally graded emit anything in between; the statistics layer never needs
    to know which kind it is looking at.
    """

    case_id: str
    scorer: str
    score: float
    detail: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"{self.scorer} produced an out-of-range score: {self.score!r}")

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "scorer": self.scorer,
            "score": round(self.score, 6),
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CaseScore:
        return cls(
            case_id=str(payload["case_id"]),
            scorer=str(payload["scorer"]),
            score=float(payload["score"]),
            detail=str(payload.get("detail", "")),
        )


@dataclass(frozen=True)
class RunResult:
    """The full record of evaluating one target against one dataset version."""

    target: str
    dataset_name: str
    dataset_version: str
    dataset_fingerprint: str
    case_ids: tuple[str, ...]
    predictions: tuple[Prediction, ...]
    scores: tuple[CaseScore, ...]
    tags: dict[str, tuple[str, ...]] = field(default_factory=dict)
    label: str = ""
    created_at: str | None = None

    @property
    def name(self) -> str:
        return self.label or self.target

    def scorers(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for score in self.scores:
            seen.setdefault(score.scorer, None)
        return tuple(seen)

    def vector(self, scorer: str) -> list[float]:
        """Per-case scores for ``scorer``, in dataset order.

        Comparison and gating are *paired* — case i of one run is lined up with
        case i of the other — so the ordering here has to be the dataset's
        ordering, not whatever order the scores happened to be appended in.
        """
        by_case = {s.case_id: s.score for s in self.scores if s.scorer == scorer}
        missing = [cid for cid in self.case_ids if cid not in by_case]
        if missing:
            raise KeyError(
                f"run {self.name!r} has no {scorer!r} score for {len(missing)} case(s), "
                f"first missing: {missing[0]!r}"
            )
        return [by_case[cid] for cid in self.case_ids]

    def subset(self, case_ids: list[str] | tuple[str, ...]) -> RunResult:
        """A view of this run restricted to ``case_ids`` (used for tag slices)."""
        keep = set(case_ids)
        ordered = tuple(cid for cid in self.case_ids if cid in keep)
        return RunResult(
            target=self.target,
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
            dataset_fingerprint=self.dataset_fingerprint,
            case_ids=ordered,
            predictions=tuple(p for p in self.predictions if p.case_id in keep),
            scores=tuple(s for s in self.scores if s.case_id in keep),
            tags={k: tuple(c for c in v if c in keep) for k, v in self.tags.items()},
            label=self.label,
            created_at=self.created_at,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": RUN_FORMAT_VERSION,
            "label": self.label,
            "target": self.target,
            "created_at": self.created_at,
            "dataset": {
                "name": self.dataset_name,
                "version": self.dataset_version,
                "fingerprint": self.dataset_fingerprint,
            },
            "case_ids": list(self.case_ids),
            "tags": {k: list(v) for k, v in sorted(self.tags.items())},
            "predictions": [p.to_json() for p in self.predictions],
            "scores": [s.to_json() for s in self.scores],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunResult:
        version = int(payload.get("format_version", 0))
        if version != RUN_FORMAT_VERSION:
            raise ValueError(
                f"run file uses format version {version}, this build reads "
                f"version {RUN_FORMAT_VERSION}"
            )
        dataset = payload["dataset"]
        return cls(
            target=str(payload["target"]),
            dataset_name=str(dataset["name"]),
            dataset_version=str(dataset["version"]),
            dataset_fingerprint=str(dataset["fingerprint"]),
            case_ids=tuple(payload["case_ids"]),
            predictions=tuple(Prediction.from_json(p) for p in payload.get("predictions", [])),
            scores=tuple(CaseScore.from_json(s) for s in payload.get("scores", [])),
            tags={k: tuple(v) for k, v in payload.get("tags", {}).items()},
            label=str(payload.get("label", "")),
            created_at=payload.get("created_at"),
        )

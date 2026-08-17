"""Versioned evaluation datasets.

A dataset is a JSONL file plus two identifiers:

``version``
    Declared by whoever edits the file. It is documentation.

``fingerprint``
    Derived from the content. It is enforcement.

The distinction matters because the failure mode this guards against is not
"someone forgot to bump the version" — it is "the eval set quietly changed
between the baseline run and the candidate run, and the resulting score delta
measured the dataset edit rather than the model change". The gate refuses to
compare runs whose fingerprints differ, so that mistake becomes an error rather
than a plausible-looking number.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .types import Case

BUILTIN_PREFIX = "builtin:"
_DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Dataset:
    """An ordered, fingerprinted collection of cases."""

    name: str
    version: str
    cases: tuple[Case, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id {case.id!r} in dataset {self.name!r}")
            seen.add(case.id)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.id for case in self.cases)

    @property
    def fingerprint(self) -> str:
        """A short content hash covering every case and its order.

        Canonical JSON with sorted keys keeps the hash stable across Python
        versions and across cosmetic reformatting of the source file.
        """
        digest = hashlib.sha256()
        for case in self.cases:
            digest.update(json.dumps(case.to_json(), sort_keys=True, ensure_ascii=False).encode())
            digest.update(b"\x1e")
        return digest.hexdigest()[:16]

    def tag_index(self) -> dict[str, tuple[str, ...]]:
        """Map each tag to the case ids carrying it, for per-slice reporting."""
        index: dict[str, list[str]] = {}
        for case in self.cases:
            for tag in case.tags:
                index.setdefault(tag, []).append(case.id)
        return {tag: tuple(ids) for tag, ids in sorted(index.items())}

    def filter_by_tag(self, tag: str) -> Dataset:
        selected = tuple(case for case in self.cases if tag in case.tags)
        if not selected:
            raise ValueError(f"no cases in {self.name!r} carry the tag {tag!r}")
        return Dataset(name=f"{self.name}[{tag}]", version=self.version, cases=selected)

    def to_jsonl(self) -> str:
        header = {"__dataset__": self.name, "__version__": self.version}
        lines = [json.dumps(header, ensure_ascii=False)]
        lines.extend(json.dumps(case.to_json(), ensure_ascii=False) for case in self.cases)
        return "\n".join(lines) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_jsonl(), encoding="utf-8")


def parse_jsonl(text: str, *, default_name: str = "dataset") -> Dataset:
    """Parse dataset JSONL.

    The first line may be a header object carrying ``__dataset__`` and
    ``__version__``; every other line is a case.
    """
    name = default_name
    version = "0"
    cases: list[Case] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {lineno} is not valid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"line {lineno} is not a JSON object")
        if "__dataset__" in payload or "__version__" in payload:
            name = str(payload.get("__dataset__", name))
            version = str(payload.get("__version__", version))
            continue
        try:
            cases.append(Case.from_json(payload))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    if not cases:
        raise ValueError("dataset contains no cases")
    return Dataset(name=name, version=version, cases=tuple(cases))


def load_dataset(spec: str | Path) -> Dataset:
    """Load a dataset from ``builtin:<name>`` or a filesystem path."""
    text = str(spec)
    if text.startswith(BUILTIN_PREFIX):
        return load_builtin(text[len(BUILTIN_PREFIX) :])
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"no dataset at {path}. Bundled datasets are addressed as "
            f"'builtin:<name>' — available: {', '.join(list_builtins()) or 'none'}"
        )
    return parse_jsonl(path.read_text(encoding="utf-8"), default_name=path.stem)


def load_builtin(name: str) -> Dataset:
    path = _DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"unknown bundled dataset {name!r}; available: {', '.join(list_builtins())}"
        )
    return parse_jsonl(path.read_text(encoding="utf-8"), default_name=name)


def list_builtins() -> list[str]:
    return sorted(p.stem for p in _DATA_DIR.glob("*.jsonl") if not p.stem.endswith("labels"))


def build_dataset(name: str, version: str, cases: Iterable[Case]) -> Dataset:
    return Dataset(name=name, version=version, cases=tuple(cases))

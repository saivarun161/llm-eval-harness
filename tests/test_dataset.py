from __future__ import annotations

import pytest

from evalharness.dataset import (
    Dataset,
    build_dataset,
    list_builtins,
    load_builtin,
    load_dataset,
    parse_jsonl,
)
from evalharness.types import Case

SAMPLE = """
{"__dataset__": "sample", "__version__": "2.1.0"}
# a comment line is ignored
{"id": "a", "input": "q1", "expected": "one", "tags": ["x"]}
{"id": "b", "input": "q2", "expected": "two"}
"""


def test_parse_reads_header_and_cases():
    dataset = parse_jsonl(SAMPLE)
    assert (dataset.name, dataset.version) == ("sample", "2.1.0")
    assert dataset.case_ids == ("a", "b")
    assert dataset.cases[0].tags == ("x",)


def test_fingerprint_is_stable_and_content_sensitive():
    first = parse_jsonl(SAMPLE)
    reformatted = parse_jsonl(SAMPLE.replace('{"id": "a"', '{ "id": "a"'))
    assert first.fingerprint == reformatted.fingerprint

    edited = parse_jsonl(SAMPLE.replace('"one"', '"uno"'))
    assert edited.fingerprint != first.fingerprint


def test_fingerprint_tracks_case_order():
    a = build_dataset("d", "1", [Case("a", "q", "x"), Case("b", "q", "y")])
    b = build_dataset("d", "1", [Case("b", "q", "y"), Case("a", "q", "x")])
    assert a.fingerprint != b.fingerprint


def test_version_alone_does_not_change_the_fingerprint():
    # The declared version is documentation; only content is enforced.
    a = build_dataset("d", "1.0.0", [Case("a", "q", "x")])
    b = build_dataset("d", "9.9.9", [Case("a", "q", "x")])
    assert a.fingerprint == b.fingerprint


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate case id"):
        Dataset("d", "1", (Case("a", "q", "x"), Case("a", "q", "y")))


def test_bad_lines_report_their_position():
    with pytest.raises(ValueError, match="line 2"):
        parse_jsonl('{"id": "a", "input": "q", "expected": "x"}\nnot json\n')
    with pytest.raises(ValueError, match="line 1"):
        parse_jsonl('{"input": "q"}\n')


def test_empty_dataset_is_an_error():
    with pytest.raises(ValueError, match="no cases"):
        parse_jsonl("\n# nothing here\n")


def test_tag_index_and_filtering():
    dataset = parse_jsonl(SAMPLE)
    assert dataset.tag_index() == {"x": ("a",)}
    only_x = dataset.filter_by_tag("x")
    assert only_x.case_ids == ("a",)
    assert only_x.name == "sample[x]"
    with pytest.raises(ValueError, match="no cases"):
        dataset.filter_by_tag("missing")


def test_roundtrip_through_jsonl(tmp_path):
    dataset = parse_jsonl(SAMPLE)
    path = tmp_path / "out.jsonl"
    dataset.write(path)
    reloaded = load_dataset(path)
    assert reloaded.fingerprint == dataset.fingerprint
    assert reloaded.version == dataset.version


def test_builtin_dataset_is_well_formed():
    assert "qa_general" in list_builtins()
    dataset = load_builtin("qa_general")
    assert len(dataset) >= 40
    assert len(dataset) == len(set(dataset.case_ids))
    for case in dataset:
        assert case.expected, f"{case.id} has no reference answer"
        assert case.tags, f"{case.id} has no tags"
        assert case.metadata.get("wrong"), f"{case.id} has no wrong-answer variant"
        assert case.metadata.get("paraphrase"), f"{case.id} has no paraphrase variant"
        assert case.metadata["wrong"] != case.expected


def test_builtin_lookup_errors_are_helpful():
    with pytest.raises(FileNotFoundError, match="available"):
        load_dataset("builtin:nope")
    with pytest.raises(FileNotFoundError, match="builtin:"):
        load_dataset("/nonexistent/path.jsonl")


def test_case_validation():
    with pytest.raises(ValueError, match="case id"):
        Case("", "q", "a")
    with pytest.raises(ValueError, match="empty input"):
        Case("a", "", "a")


def test_dataset_is_iterable_and_sized():
    dataset = parse_jsonl(SAMPLE)
    assert len(dataset) == 2
    assert [c.id for c in dataset] == ["a", "b"]

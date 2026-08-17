from __future__ import annotations

import os
import subprocess
import sys

import pytest

from evalharness.textutil import (
    cosine,
    coverage,
    embed,
    key_terms,
    normalize,
    numbers,
    similarity,
    tokens,
)


def test_normalize_folds_case_accents_and_punctuation():
    assert normalize("The  Café!") == "cafe"
    assert normalize("Gabriel García Márquez") == "gabriel garcia marquez"
    assert normalize("Paris.") == "paris"


def test_normalize_can_keep_articles():
    assert normalize("the not") == "not"
    assert normalize("the not", strip_articles=False) == "the not"


def test_tokens_and_numbers():
    assert tokens("Canada and the United States") == ["canada", "and", "united", "states"]
    assert numbers("1,440 minutes in 1 day") == [1440.0, 1.0]
    assert numbers("no digits here") == []


def test_embedding_is_unit_length_and_deterministic():
    vector = embed("the capital of France")
    assert sum(v * v for v in vector) == pytest.approx(1.0)
    assert embed("the capital of France") == vector
    assert embed("") == tuple([0.0] * len(vector))


def test_embedding_is_stable_across_processes():
    # Python salts str.__hash__ per process; a hashing vectoriser built on it
    # would silently invalidate every stored baseline. This guards the choice
    # of BLAKE2b.
    script = "from evalharness.textutil import embed; print(round(sum(embed('Paris')), 12))"
    local = round(sum(embed("Paris")), 12)
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "random"},
    )
    assert float(out.stdout.strip()) == pytest.approx(local)


def test_cosine_orders_related_text_above_unrelated():
    assert cosine("Paris", "Paris") == pytest.approx(1.0)
    assert cosine("the capital of France", "the capital of Spain") > cosine(
        "the capital of France", "photosynthesis in plants"
    )


def test_coverage_is_asymmetric():
    verbose = "Based on what I know, the answer is Paris. Hope that helps!"
    assert coverage(verbose, "Paris") > 0.9
    assert coverage("Paris", verbose) < 0.4


def test_similarity_rewards_a_correct_answer_buried_in_prose():
    verbose = "Based on what I know, the answer to that is Canberra. Hope that helps!"
    assert similarity(verbose, "Canberra") > 0.6
    assert similarity("Sydney", "Canberra") < 0.3


def test_similarity_handles_empty_input():
    assert similarity("", "Paris") == 0.0
    assert similarity("Paris", "") == 0.0
    assert similarity("", "") == 1.0


def test_similarity_stays_in_range():
    for prediction, expected in [
        ("Paris", "Paris"),
        ("something else entirely", "Paris"),
        ("a" * 500, "Paris"),
    ]:
        assert 0.0 <= similarity(prediction, expected) <= 1.0


def test_key_terms_drops_connective_words():
    terms = key_terms("The tilt of the rotational axis")
    assert "tilt" in terms
    assert "rotational" in terms
    assert "of" not in terms

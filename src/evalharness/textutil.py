"""Text normalisation and a deterministic local embedding.

The embedding is a signed hashing vectoriser over character n-grams and word
unigrams. It is not a language model and it is not pretending to be one — it
captures surface similarity, which is exactly the signal a reference-based
scorer needs, and it does so with no download, no API key and no variance
between runs.

One subtlety worth stating: Python's built-in ``hash()`` for strings is salted
per process, so a vectoriser built on it would produce different vectors in
different processes. Baselines recorded last week have to be comparable to a run
made today, so this module hashes with BLAKE2b instead.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from functools import lru_cache

DEFAULT_DIM = 512
_CHAR_NGRAMS = (3, 4)
_WORD_WEIGHT = 1.5

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s%°/.-]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[\w][\w.'%-]*", re.UNICODE)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")


def normalize(text: str, *, strip_articles: bool = True) -> str:
    """Lowercase, de-accent, drop stray punctuation and collapse whitespace.

    This is the normalisation applied before *every* comparison in the harness.
    Making it shared, rather than per-scorer, means a disagreement between two
    scorers is a real disagreement about meaning and not an artefact of one of
    them keeping a trailing full stop.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.lower()
    folded = folded.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    # Apostrophes are deleted rather than turned into a space, so "don't" and
    # "isn't" stay single tokens for the refusal and negation patterns.
    folded = folded.replace("'", "")
    folded = _PUNCT.sub(" ", folded)
    if strip_articles:
        folded = _ARTICLES.sub(" ", folded)
    folded = folded.replace(".", " ").replace("-", " ")
    return _WHITESPACE.sub(" ", folded).strip()


def tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def numbers(text: str) -> list[float]:
    """Numeric literals in the order they appear, thousands separators removed."""
    found: list[float] = []
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "")
        try:
            found.append(float(cleaned))
        except ValueError:  # pragma: no cover - regex already constrains this
            continue
    return found


def _hash_feature(feature: str) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    # Signed hashing: the sign bit makes collisions cancel on average instead of
    # accumulating, which keeps cosine similarity usable at a modest dimension.
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return value % DEFAULT_DIM, sign


def _features(text: str) -> dict[str, float]:
    norm = normalize(text)
    counts: dict[str, float] = {}
    if not norm:
        return counts
    padded = f" {norm} "
    for n in _CHAR_NGRAMS:
        if len(padded) < n:
            continue
        for i in range(len(padded) - n + 1):
            gram = padded[i : i + n]
            counts[f"c{n}:{gram}"] = counts.get(f"c{n}:{gram}", 0.0) + 1.0
    for token in _WORD.findall(norm):
        counts[f"w:{token}"] = counts.get(f"w:{token}", 0.0) + _WORD_WEIGHT
    return counts


@lru_cache(maxsize=4096)
def embed(text: str) -> tuple[float, ...]:
    """An L2-normalised dense vector for ``text``."""
    vector = [0.0] * DEFAULT_DIM
    for feature, count in _features(text).items():
        index, sign = _hash_feature(feature)
        # Sub-linear term weighting: a word repeated ten times should not
        # dominate a vector ten times over.
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(v / norm for v in vector)


def cosine(a: str, b: str) -> float:
    va, vb = embed(a), embed(b)
    return sum(x * y for x, y in zip(va, vb, strict=True))


@lru_cache(maxsize=4096)
def _feature_mass(text: str) -> tuple[dict[str, float], float]:
    features = _features(text)
    return features, sum(features.values())


def coverage(container: str, contained: str) -> float:
    """How much of ``contained``'s feature mass appears inside ``container``.

    Cosine alone punishes a correct answer for being wordy: "Paris" and "The
    capital of France is Paris" point in noticeably different directions. This
    asymmetric measure asks the question a reference-based scorer actually
    cares about — *is the expected answer in there somewhere?* — and the
    semantic scorer blends the two.
    """
    target, mass = _feature_mass(contained)
    if mass == 0.0:
        return 0.0
    source, _ = _feature_mass(container)
    matched = sum(min(count, source.get(feature, 0.0)) for feature, count in target.items())
    return matched / mass


def similarity(prediction: str, expected: str) -> float:
    """Blended surface similarity in ``[0, 1]``."""
    if not expected.strip():
        return 1.0 if not prediction.strip() else 0.0
    if not prediction.strip():
        return 0.0
    direct = max(0.0, cosine(prediction, expected))
    contains = coverage(prediction, expected)
    return max(0.0, min(1.0, 0.5 * direct + 0.5 * contains))


def key_terms(text: str) -> list[str]:
    """Content words plus numeric literals — the facts an answer must carry.

    Stopwords are deliberately a short, fixed list rather than a linguistic
    resource: the point is to drop connective tissue, not to do morphology.
    """
    stop = {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "it",
        "its",
        "that",
        "this",
        "with",
        "by",
        "from",
        "as",
        "his",
        "her",
        "their",
        "there",
        "which",
        "who",
        "what",
    }
    seen: dict[str, None] = {}
    for token in tokens(text):
        if token in stop or len(token) < 2:
            continue
        seen.setdefault(token, None)
    return list(seen)

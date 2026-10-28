"""Fuzzy string grader based on edit distance.

Robust to typos and minor surface variation. The similarity is ``1 - normalized
Levenshtein distance``, computed with a dynamic-programming edit distance in O(n*m) time
and O(min(n, m)) space. Useful for free-form short answers where an exact match is too
brittle but token overlap is too loose.
"""

from __future__ import annotations

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.graders.normalization import normalize_text


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Keep the shorter string on the inner axis to bound memory.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (char_a != char_b)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """Return a similarity in ``[0, 1]`` as ``1 - normalized edit distance``."""
    if not a and not b:
        return 1.0
    distance = levenshtein(a, b)
    return 1.0 - distance / max(len(a), len(b))


class FuzzyMatchGrader:
    """Scores a prediction by normalized edit-distance similarity to the target.

    Args:
        threshold: Minimum similarity to count as correct.
        normalize: Whether to canonicalize text before comparing.
    """

    def __init__(self, *, threshold: float = 0.85, normalize: bool = True) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._threshold = threshold
        self._normalize = normalize

    @property
    def name(self) -> str:
        return "fuzzy_match"

    def grade(self, example: Example, prediction: str) -> Grade:
        if self._normalize:
            predicted = normalize_text(prediction)
            target = normalize_text(example.target)
        else:
            predicted = prediction.strip()
            target = example.target.strip()
        score = similarity(predicted, target)
        passed = score >= self._threshold
        rationale = f"similarity {score:.2f} (threshold {self._threshold:.2f})"
        return Grade(1.0 if passed else 0.0, rationale)

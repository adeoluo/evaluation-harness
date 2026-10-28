"""Exact-match grader with configurable normalization."""

from __future__ import annotations

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.graders.normalization import normalize_text


class ExactMatchGrader:
    """Awards full marks when the normalized prediction equals the normalized target.

    By default the match is lenient to surface-form differences (case, punctuation,
    articles, whitespace). Pass ``normalize=False`` for byte-exact comparison.
    """

    def __init__(self, *, normalize: bool = True) -> None:
        self._normalize = normalize

    @property
    def name(self) -> str:
        return "exact_match" if self._normalize else "exact_match_strict"

    def grade(self, example: Example, prediction: str) -> Grade:
        if self._normalize:
            expected = normalize_text(example.target)
            actual = normalize_text(prediction)
        else:
            expected = example.target
            actual = prediction
        correct = actual == expected
        rationale = None if correct else f"expected {expected!r}, got {actual!r}"
        return Grade(1.0 if correct else 0.0, rationale)

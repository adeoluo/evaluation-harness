"""Grader that checks whether the target appears within the prediction.

Useful when a model is expected to produce a short answer embedded in free-form text
(for example "The capital is Paris." should match the target "Paris").
"""

from __future__ import annotations

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.graders.normalization import normalize_text


class IncludesGrader:
    """Awards full marks when the normalized target is a substring of the prediction."""

    def __init__(self, *, normalize: bool = True) -> None:
        self._normalize = normalize

    @property
    def name(self) -> str:
        return "includes"

    def grade(self, example: Example, prediction: str) -> Grade:
        if self._normalize:
            needle = normalize_text(example.target)
            haystack = normalize_text(prediction)
        else:
            needle = example.target
            haystack = prediction
        correct = bool(needle) and needle in haystack
        rationale = None if correct else f"{needle!r} not found in prediction"
        return Grade(1.0 if correct else 0.0, rationale)

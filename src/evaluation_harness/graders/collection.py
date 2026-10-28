"""Graders for set-valued and list-valued answers.

Some tasks expect a collection of items - "list the primary colors", "which of these are
prime". Scoring these as one string is wrong; order and separators are irrelevant while
membership is everything. :class:`SetMatchGrader` scores by Jaccard overlap (or exact set
equality), and :class:`ListMatchGrader` scores ordered sequences by position.
"""

from __future__ import annotations

import re

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.graders.normalization import normalize_text

_SPLIT = re.compile(r"[,;\n]+")


def parse_items(text: str) -> list[str]:
    """Split ``text`` into normalized, non-empty items on commas, semicolons, newlines.

    Article stripping is disabled so single-letter items such as "a" survive.
    """
    items = [normalize_text(part, strip_articles=False) for part in _SPLIT.split(text)]
    return [item for item in items if item]


class SetMatchGrader:
    """Scores unordered collections by Jaccard overlap or exact set equality.

    Args:
        exact: If ``True``, award full marks only on exact set equality; otherwise return
            the Jaccard similarity as partial credit.
    """

    def __init__(self, *, exact: bool = False) -> None:
        self._exact = exact

    @property
    def name(self) -> str:
        return "set_match"

    def grade(self, example: Example, prediction: str) -> Grade:
        predicted = set(parse_items(prediction))
        target = set(parse_items(example.target))
        if not predicted and not target:
            return Grade(1.0)
        union = predicted | target
        jaccard = len(predicted & target) / len(union) if union else 0.0
        if self._exact:
            passed = predicted == target
            return Grade(1.0 if passed else 0.0, f"Jaccard {jaccard:.2f}")
        return Grade(jaccard, f"Jaccard {jaccard:.2f}")


class ListMatchGrader:
    """Scores ordered sequences by the fraction of positions that match."""

    @property
    def name(self) -> str:
        return "list_match"

    def grade(self, example: Example, prediction: str) -> Grade:
        predicted = parse_items(prediction)
        target = parse_items(example.target)
        if not target:
            return Grade(1.0 if not predicted else 0.0)
        matches = sum(
            1
            for index, item in enumerate(target)
            if index < len(predicted) and predicted[index] == item
        )
        score = matches / len(target)
        return Grade(score, f"{matches}/{len(target)} positions correct")

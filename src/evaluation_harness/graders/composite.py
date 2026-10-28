"""Graders that combine other graders.

Real evaluation often needs more than one signal. These combinators let graders be
composed without changing the :class:`~evaluation_harness.core.grader.Grader` protocol:

- :class:`AnyGrader` - pass if any child passes (the maximum score).
- :class:`AllGrader` - the minimum child score (every child must be satisfied).
- :class:`WeightedGrader` - a weighted average of child scores.
"""

from __future__ import annotations

from collections.abc import Sequence

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade, Grader


class AnyGrader:
    """Returns the maximum score across child graders (logical OR)."""

    def __init__(self, graders: Sequence[Grader], *, name: str = "any") -> None:
        if not graders:
            raise ValueError("AnyGrader requires at least one child grader")
        self._graders = tuple(graders)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def grade(self, example: Example, prediction: str) -> Grade:
        grades = [grader.grade(example, prediction) for grader in self._graders]
        best = max(grades, key=lambda grade: grade.score)
        return Grade(best.score, best.rationale)


class AllGrader:
    """Returns the minimum score across child graders (logical AND)."""

    def __init__(self, graders: Sequence[Grader], *, name: str = "all") -> None:
        if not graders:
            raise ValueError("AllGrader requires at least one child grader")
        self._graders = tuple(graders)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def grade(self, example: Example, prediction: str) -> Grade:
        grades = [grader.grade(example, prediction) for grader in self._graders]
        worst = min(grades, key=lambda grade: grade.score)
        return Grade(worst.score, worst.rationale)


class WeightedGrader:
    """Returns a weighted average of child grader scores.

    Args:
        weighted_graders: Pairs of ``(grader, weight)``. Weights are normalized.
    """

    def __init__(
        self,
        weighted_graders: Sequence[tuple[Grader, float]],
        *,
        name: str = "weighted",
    ) -> None:
        if not weighted_graders:
            raise ValueError("WeightedGrader requires at least one child grader")
        if any(weight < 0 for _, weight in weighted_graders):
            raise ValueError("weights must be non-negative")
        total = sum(weight for _, weight in weighted_graders)
        if total == 0:
            raise ValueError("weights must not all be zero")
        self._weighted = tuple(weighted_graders)
        self._total = total
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def grade(self, example: Example, prediction: str) -> Grade:
        weighted_sum = sum(
            grader.grade(example, prediction).score * weight for grader, weight in self._weighted
        )
        return Grade(weighted_sum / self._total)

"""The :class:`Grader`: scores a model prediction against an example.

A grader returns a :class:`Grade` - a continuous score in ``[0, 1]`` plus an optional
human-readable rationale. Continuous scores (rather than bare booleans) let the same
interface express exact match, partial credit, and rubric-based grading uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from evaluation_harness.core.example import Example


@dataclass(frozen=True, slots=True)
class Grade:
    """The outcome of grading a single prediction.

    Attributes:
        score: A value in ``[0, 1]``; ``1.0`` is a fully correct answer.
        rationale: Optional explanation, surfaced in reports and error analysis.
    """

    score: float
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Grade.score must be in [0, 1], got {self.score}")

    @property
    def passed(self) -> bool:
        """Whether the grade counts as correct (score of exactly ``1.0``)."""
        return self.score == 1.0


@runtime_checkable
class Grader(Protocol):
    """Scores a prediction against the reference answer in an example."""

    @property
    def name(self) -> str:
        """A short identifier used in reports (for example ``exact_match``)."""
        ...

    def grade(self, example: Example, prediction: str) -> Grade:
        """Score ``prediction`` against ``example.target``."""
        ...

"""The :class:`Task`: a named benchmark pairing examples with a grader."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grader


@dataclass(frozen=True, slots=True)
class Task:
    """A benchmark: a set of examples scored by a single grader.

    Attributes:
        name: Stable identifier used in reports (for example ``arithmetic``).
        examples: The evaluation instances.
        grader: The grader applied to every prediction on this task.
        description: Optional human-readable summary of what the task measures.
    """

    name: str
    examples: Sequence[Example]
    grader: Grader
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task.name must be a non-empty string")
        if not self.examples:
            raise ValueError(f"Task {self.name!r} has no examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterable[Example]:
        return iter(self.examples)

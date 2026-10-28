"""Result containers produced by a :class:`~evaluation_harness.core.runner.Runner`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.statistics import Aggregate, bootstrap_aggregate


@dataclass(frozen=True, slots=True)
class ExampleResult:
    """The outcome of evaluating one example.

    Attributes:
        example: The evaluated instance.
        prediction: The model's raw output.
        grade: The grader's verdict.
        latency_seconds: Wall-clock time spent generating the prediction.
    """

    example: Example
    prediction: str
    grade: Grade
    latency_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Aggregated results for a model evaluated on a task.

    Attributes:
        task_name: Name of the evaluated task.
        model_name: Name of the evaluated model.
        example_results: Per-example outcomes, in evaluation order.
        seed: Seed used for statistical aggregation, recorded for reproducibility.
    """

    task_name: str
    model_name: str
    example_results: Sequence[ExampleResult] = field(default_factory=tuple)
    seed: int = 0

    @property
    def scores(self) -> list[float]:
        """The per-example scores."""
        return [result.grade.score for result in self.example_results]

    @property
    def count(self) -> int:
        """Number of examples evaluated."""
        return len(self.example_results)

    def accuracy(self, *, confidence: float = 0.95) -> Aggregate:
        """Mean score with a bootstrap confidence interval."""
        return bootstrap_aggregate(self.scores, confidence=confidence, seed=self.seed)

    @property
    def mean_latency_seconds(self) -> float:
        """Average generation latency across examples."""
        if not self.example_results:
            return 0.0
        total = sum(result.latency_seconds for result in self.example_results)
        return total / len(self.example_results)

"""Regression detection between two evaluation runs.

Continuous evaluation needs an automatic answer to "did this change make the model
worse?". Given a baseline run and a candidate run, :func:`detect_regressions` compares
accuracy per task and flags any task whose accuracy dropped by more than a tolerance, and
also surfaces per-example flips (previously correct, now wrong) for error analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation_harness.persistence import RunRecord


@dataclass(frozen=True, slots=True)
class TaskDelta:
    """The change in accuracy for one task between baseline and candidate."""

    task_name: str
    baseline_accuracy: float
    candidate_accuracy: float
    newly_failing: tuple[str, ...]
    newly_passing: tuple[str, ...]

    @property
    def delta(self) -> float:
        return self.candidate_accuracy - self.baseline_accuracy


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """The outcome of comparing a candidate run against a baseline."""

    deltas: tuple[TaskDelta, ...]
    tolerance: float

    @property
    def regressions(self) -> tuple[TaskDelta, ...]:
        """Tasks whose accuracy dropped by more than the tolerance."""
        return tuple(d for d in self.deltas if d.delta < -self.tolerance)

    @property
    def improvements(self) -> tuple[TaskDelta, ...]:
        """Tasks whose accuracy rose by more than the tolerance."""
        return tuple(d for d in self.deltas if d.delta > self.tolerance)

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions)


def _accuracy(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def detect_regressions(
    baseline: RunRecord,
    candidate: RunRecord,
    *,
    tolerance: float = 0.0,
) -> RegressionReport:
    """Compare a candidate run against a baseline, task by task.

    Args:
        baseline: The reference run.
        candidate: The new run being checked.
        tolerance: Accuracy drop allowed before a task counts as a regression.

    Returns:
        A :class:`RegressionReport` with per-task deltas and example-level flips.
    """
    deltas: list[TaskDelta] = []
    for candidate_result in candidate.results:
        baseline_result = baseline.result_for(candidate_result.task_name)
        if baseline_result is None:
            continue

        baseline_by_id = {
            item.example.identifier: item.grade.passed for item in baseline_result.example_results
        }
        newly_failing: list[str] = []
        newly_passing: list[str] = []
        for item in candidate_result.example_results:
            example_id = item.example.identifier
            if example_id not in baseline_by_id:
                continue
            was_passing = baseline_by_id[example_id]
            now_passing = item.grade.passed
            if was_passing and not now_passing:
                newly_failing.append(example_id)
            elif not was_passing and now_passing:
                newly_passing.append(example_id)

        deltas.append(
            TaskDelta(
                task_name=candidate_result.task_name,
                baseline_accuracy=_accuracy(baseline_result.scores),
                candidate_accuracy=_accuracy(candidate_result.scores),
                newly_failing=tuple(newly_failing),
                newly_passing=tuple(newly_passing),
            )
        )

    return RegressionReport(deltas=tuple(deltas), tolerance=tolerance)

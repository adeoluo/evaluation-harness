"""The :class:`Runner`: executes a model over a task and collects results."""

from __future__ import annotations

import time
from collections.abc import Callable

from evaluation_harness.core.example import Example
from evaluation_harness.core.model import Model
from evaluation_harness.core.result import ExampleResult, TaskResult
from evaluation_harness.core.task import Task

ProgressCallback = Callable[[int, int], None]


class Runner:
    """Evaluates models on tasks.

    The runner is intentionally thin: it owns the evaluation loop (generate, time,
    grade, collect) and nothing else. Models, graders, and tasks are injected, so the
    loop is identical whether the model is a frontier API or a deterministic stub.
    """

    def __init__(self, *, seed: int = 0) -> None:
        """Create a runner.

        Args:
            seed: Seed recorded on results and used for statistical aggregation.
        """
        self._seed = seed

    def run(
        self,
        model: Model,
        task: Task,
        *,
        progress: ProgressCallback | None = None,
    ) -> TaskResult:
        """Evaluate ``model`` on ``task``.

        Args:
            model: The system under evaluation.
            task: The benchmark to run.
            progress: Optional callback invoked as ``(completed, total)`` after
                each example, for progress reporting.

        Returns:
            A :class:`TaskResult` with per-example outcomes and the recorded seed.
        """
        total = len(task)
        results: list[ExampleResult] = []
        for index, example in enumerate(task.examples, start=1):
            results.append(self._evaluate_one(model, task, example))
            if progress is not None:
                progress(index, total)
        return TaskResult(
            task_name=task.name,
            model_name=model.name,
            example_results=tuple(results),
            seed=self._seed,
        )

    def _evaluate_one(self, model: Model, task: Task, example: Example) -> ExampleResult:
        start = time.perf_counter()
        prediction = model.generate(example)
        latency = time.perf_counter() - start
        grade = task.grader.grade(example, prediction)
        return ExampleResult(
            example=example,
            prediction=prediction,
            grade=grade,
            latency_seconds=latency,
        )

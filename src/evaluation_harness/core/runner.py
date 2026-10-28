"""The :class:`Runner`: executes a model over a task and collects results.

The runner owns the evaluation loop - generate, time, grade, collect - and nothing else.
Models, graders, and tasks are injected, so the loop is identical whether the model is a
frontier API or a deterministic stub.

Two production concerns are handled here without leaking into the rest of the framework:

- **Concurrency.** Provider calls are I/O-bound, so a thread pool (``max_workers``)
  evaluates examples in parallel while results are reassembled in task order.
- **Transient failures.** A model call may fail intermittently; ``max_retries`` retries
  with exponential backoff before giving up, and a persistent failure is recorded as a
  zero-scored result rather than aborting the whole run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.core.model import Model
from evaluation_harness.core.result import ExampleResult, TaskResult
from evaluation_harness.core.task import Task

ProgressCallback = Callable[[int, int], None]


class Runner:
    """Evaluates models on tasks, optionally concurrently and with retries."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_workers: int = 1,
        max_retries: int = 0,
        retry_base_delay: float = 0.1,
    ) -> None:
        """Create a runner.

        Args:
            seed: Seed recorded on results and used for statistical aggregation.
            max_workers: Number of worker threads. ``1`` runs sequentially.
            max_retries: Retries for a failing model call before recording a failure.
            retry_base_delay: Base delay (seconds) for exponential backoff between retries.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self._seed = seed
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    def run(
        self,
        model: Model,
        task: Task,
        *,
        progress: ProgressCallback | None = None,
    ) -> TaskResult:
        """Evaluate ``model`` on ``task`` and return a :class:`TaskResult`."""
        total = len(task)
        if self._max_workers == 1:
            results = self._run_sequential(model, task, progress, total)
        else:
            results = self._run_concurrent(model, task, progress, total)
        return TaskResult(
            task_name=task.name,
            model_name=model.name,
            example_results=tuple(results),
            seed=self._seed,
        )

    def _run_sequential(
        self,
        model: Model,
        task: Task,
        progress: ProgressCallback | None,
        total: int,
    ) -> list[ExampleResult]:
        results: list[ExampleResult] = []
        for index, example in enumerate(task.examples, start=1):
            results.append(self._evaluate_one(model, task, example))
            if progress is not None:
                progress(index, total)
        return results

    def _run_concurrent(
        self,
        model: Model,
        task: Task,
        progress: ProgressCallback | None,
        total: int,
    ) -> list[ExampleResult]:
        # Preallocate by index so results stay in task order regardless of completion order.
        slots: list[ExampleResult | None] = [None] * total
        completed = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_index = {
                executor.submit(self._evaluate_one, model, task, example): index
                for index, example in enumerate(task.examples)
            }
            for future in as_completed(future_to_index):
                slots[future_to_index[future]] = future.result()
                completed += 1
                if progress is not None:
                    progress(completed, total)
        return [result for result in slots if result is not None]

    def _evaluate_one(self, model: Model, task: Task, example: Example) -> ExampleResult:
        start = time.perf_counter()
        try:
            prediction = self._generate_with_retries(model, example)
        except Exception as error:
            latency = time.perf_counter() - start
            return ExampleResult(
                example=example,
                prediction="",
                grade=Grade(0.0, f"model error: {error}"),
                latency_seconds=latency,
            )
        latency = time.perf_counter() - start
        grade = task.grader.grade(example, prediction)
        return ExampleResult(
            example=example,
            prediction=prediction,
            grade=grade,
            latency_seconds=latency,
        )

    def _generate_with_retries(self, model: Model, example: Example) -> str:
        attempt = 0
        while True:
            try:
                return model.generate(example)
            except Exception:
                if attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_base_delay * (2**attempt))
                attempt += 1

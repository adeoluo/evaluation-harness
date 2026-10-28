"""Core abstractions for the evaluation harness.

The framework is built around five small, composable concepts:

- :class:`~evaluation_harness.core.example.Example` - a single evaluation instance.
- :class:`~evaluation_harness.core.model.Model` - a system under evaluation.
- :class:`~evaluation_harness.core.grader.Grader` - scores a prediction.
- :class:`~evaluation_harness.core.task.Task` - a named benchmark (examples + grader).
- :class:`~evaluation_harness.core.runner.Runner` - executes a model over a task.

Results flow through :mod:`evaluation_harness.core.result`, which carries per-example
scores and aggregate statistics (mean with a bootstrap confidence interval).
"""

from __future__ import annotations

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade, Grader
from evaluation_harness.core.model import Model
from evaluation_harness.core.result import ExampleResult, TaskResult
from evaluation_harness.core.runner import Runner
from evaluation_harness.core.task import Task

__all__ = [
    "Example",
    "ExampleResult",
    "Grade",
    "Grader",
    "Model",
    "Runner",
    "Task",
    "TaskResult",
]

"""Evaluation Harness - a principled framework for evaluating language models.

Public API: the five core abstractions and the statistical aggregation used to turn
raw scores into honest, error-barred metrics.
"""

from __future__ import annotations

from evaluation_harness.core import (
    Example,
    ExampleResult,
    Grade,
    Grader,
    Model,
    Runner,
    Task,
    TaskResult,
)
from evaluation_harness.statistics import Aggregate, bootstrap_aggregate

__version__ = "0.1.0"

__all__ = [
    "Aggregate",
    "Example",
    "ExampleResult",
    "Grade",
    "Grader",
    "Model",
    "Runner",
    "Task",
    "TaskResult",
    "__version__",
    "bootstrap_aggregate",
]

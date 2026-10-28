"""Built-in tasks and a small registry for loading them by name."""

from __future__ import annotations

from collections.abc import Callable

from evaluation_harness.core.task import Task
from evaluation_harness.tasks.arithmetic import build_arithmetic_task
from evaluation_harness.tasks.entailment import build_entailment_task
from evaluation_harness.tasks.factual_qa import build_factual_qa_task
from evaluation_harness.tasks.sentiment import build_sentiment_task

_BUILDERS: dict[str, Callable[[], Task]] = {
    "arithmetic": build_arithmetic_task,
    "sentiment": build_sentiment_task,
    "entailment": build_entailment_task,
    "factual_qa": build_factual_qa_task,
}


def available_tasks() -> list[str]:
    """Return the names of tasks loadable via :func:`load_task`."""
    return sorted(_BUILDERS)


def load_task(name: str) -> Task:
    """Load a built-in task by name.

    Raises:
        KeyError: If ``name`` is not a registered built-in task.
    """
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown task {name!r}. Available: {', '.join(available_tasks())}"
        ) from None
    return builder()


__all__ = [
    "available_tasks",
    "build_arithmetic_task",
    "build_entailment_task",
    "build_factual_qa_task",
    "build_sentiment_task",
    "load_task",
]

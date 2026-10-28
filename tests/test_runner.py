"""Tests for the task container and the evaluation runner."""

from __future__ import annotations

import pytest

from evaluation_harness.core import Example, Runner, Task

from .helpers import AlwaysPassGrader, EchoTargetModel, ExactGrader, FixedModel


def _examples() -> list[Example]:
    return [
        Example(input="2 + 2", target="4"),
        Example(input="3 + 3", target="6"),
        Example(input="5 + 1", target="6"),
    ]


def test_task_rejects_empty_examples() -> None:
    with pytest.raises(ValueError, match="no examples"):
        Task(name="empty", examples=[], grader=ExactGrader())


def test_task_requires_a_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Task(name="", examples=_examples(), grader=ExactGrader())


def test_task_len_and_iter() -> None:
    task = Task(name="arith", examples=_examples(), grader=ExactGrader())
    assert len(task) == 3
    assert [e.target for e in task] == ["4", "6", "6"]


def test_runner_oracle_scores_perfectly() -> None:
    task = Task(name="arith", examples=_examples(), grader=ExactGrader())
    result = Runner(seed=0).run(EchoTargetModel(), task)
    assert result.accuracy().mean == 1.0
    assert result.model_name == "oracle"
    assert result.task_name == "arith"


def test_runner_wrong_model_scores_partially() -> None:
    task = Task(name="arith", examples=_examples(), grader=ExactGrader())
    # Always answers "6": correct for two of three examples.
    result = Runner().run(FixedModel(name="const-6", reply="6"), task)
    assert result.accuracy().mean == pytest.approx(2 / 3)


def test_runner_reports_progress() -> None:
    task = Task(name="arith", examples=_examples(), grader=AlwaysPassGrader())
    seen: list[tuple[int, int]] = []
    Runner().run(EchoTargetModel(), task, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_runner_records_latency() -> None:
    task = Task(name="arith", examples=_examples(), grader=AlwaysPassGrader())
    result = Runner().run(EchoTargetModel(), task)
    assert all(r.latency_seconds >= 0.0 for r in result.example_results)

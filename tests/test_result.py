"""Tests for the grade type and result aggregation."""

from __future__ import annotations

import pytest

from evaluation_harness.core import Example, ExampleResult, Grade, TaskResult


def test_grade_rejects_out_of_range_score() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Grade(1.5)


def test_grade_passed_flag() -> None:
    assert Grade(1.0).passed is True
    assert Grade(0.99).passed is False


def _result(score: float, latency: float = 0.1) -> ExampleResult:
    example = Example(input="q", target="a")
    return ExampleResult(example, prediction="a", grade=Grade(score), latency_seconds=latency)


def test_task_result_scores_and_count() -> None:
    result = TaskResult(
        task_name="t",
        model_name="m",
        example_results=(_result(1.0), _result(0.0), _result(1.0)),
    )
    assert result.scores == [1.0, 0.0, 1.0]
    assert result.count == 3


def test_task_result_accuracy_has_interval() -> None:
    result = TaskResult(
        task_name="t",
        model_name="m",
        example_results=tuple(_result(1.0) for _ in range(10)),
    )
    accuracy = result.accuracy()
    assert accuracy.mean == 1.0
    assert accuracy.count == 10


def test_mean_latency() -> None:
    result = TaskResult(
        task_name="t",
        model_name="m",
        example_results=(_result(1.0, 0.2), _result(0.0, 0.4)),
    )
    assert result.mean_latency_seconds == pytest.approx(0.3)


def test_mean_latency_empty_is_zero() -> None:
    result = TaskResult(task_name="t", model_name="m", example_results=())
    assert result.mean_latency_seconds == 0.0

"""Tests for result reporting (text table and JSON)."""

from __future__ import annotations

import json

from evaluation_harness.core import Example, ExampleResult, Grade, TaskResult
from evaluation_harness.reporting import (
    format_results_table,
    results_to_dict,
    results_to_json,
)


def _result(model: str = "m", task: str = "t") -> TaskResult:
    examples = [
        ExampleResult(
            example=Example(input="q1", target="a", id="e1"),
            prediction="a",
            grade=Grade(1.0),
            latency_seconds=0.01,
        ),
        ExampleResult(
            example=Example(input="q2", target="b", id="e2"),
            prediction="c",
            grade=Grade(0.0, rationale="wrong"),
            latency_seconds=0.03,
        ),
    ]
    return TaskResult(task_name=task, model_name=model, example_results=tuple(examples))


def test_table_contains_headers_and_values() -> None:
    table = format_results_table([_result()])
    assert "Model" in table and "Accuracy" in table
    assert "m" in table and "t" in table


def test_table_empty_results() -> None:
    assert format_results_table([]) == "(no results)"


def test_dict_structure_round_trips_through_json() -> None:
    payload = results_to_dict([_result()])
    text = results_to_json([_result()])
    assert json.loads(text) == payload


def test_dict_includes_per_example_records() -> None:
    payload = results_to_dict([_result()])
    result = payload["results"][0]
    assert result["count"] == 2
    assert result["accuracy"]["mean"] == 0.5
    assert len(result["examples"]) == 2
    assert result["examples"][1]["rationale"] == "wrong"


def test_table_aligns_multiple_models() -> None:
    table = format_results_table([_result("alpha"), _result("beta-model")])
    lines = table.splitlines()
    # Header, separator, and one row per result.
    assert len(lines) == 4

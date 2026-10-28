"""Tests for persistence, regression detection, and extended reporting."""

from __future__ import annotations

from pathlib import Path

from evaluation_harness.core import Example, ExampleResult, Grade, TaskResult
from evaluation_harness.persistence import RunRecord
from evaluation_harness.regression import detect_regressions
from evaluation_harness.reporting_ext import (
    analyze_errors,
    format_error_analysis,
    format_html_report,
    format_markdown_table,
)


def _result(scores: list[float], task: str = "t", model: str = "m") -> TaskResult:
    items = tuple(
        ExampleResult(
            example=Example(input=f"q{i}", target="a", id=f"e{i}"),
            prediction="a" if s == 1.0 else "x",
            grade=Grade(s),
            latency_seconds=0.01,
        )
        for i, s in enumerate(scores)
    )
    return TaskResult(task_name=task, model_name=model, example_results=items)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_run_record_round_trips(tmp_path: Path) -> None:
    record = RunRecord(results=(_result([1.0, 0.0, 1.0]),), label="commit-abc")
    path = tmp_path / "run.json"
    record.save(path)
    loaded = RunRecord.load(path)
    assert loaded.label == "commit-abc"
    assert loaded.results[0].accuracy().mean == record.results[0].accuracy().mean
    assert loaded.results[0].example_results[0].example.identifier == "e0"


def test_run_record_result_for() -> None:
    record = RunRecord(results=(_result([1.0], task="alpha"),))
    assert record.result_for("alpha") is not None
    assert record.result_for("missing") is None


def test_run_record_has_timestamp() -> None:
    record = RunRecord(results=(_result([1.0]),))
    assert "T" in record.created_at  # ISO 8601


# --------------------------------------------------------------------------- #
# Regression detection
# --------------------------------------------------------------------------- #
def test_detects_regression() -> None:
    baseline = RunRecord(results=(_result([1.0, 1.0, 1.0, 1.0]),))
    candidate = RunRecord(results=(_result([1.0, 0.0, 0.0, 1.0]),))
    report = detect_regressions(baseline, candidate)
    assert report.has_regression
    assert report.regressions[0].delta < 0
    assert len(report.regressions[0].newly_failing) == 2


def test_detects_improvement() -> None:
    baseline = RunRecord(results=(_result([0.0, 0.0]),))
    candidate = RunRecord(results=(_result([1.0, 1.0]),))
    report = detect_regressions(baseline, candidate)
    assert not report.has_regression
    assert report.improvements[0].delta > 0
    assert len(report.improvements[0].newly_passing) == 2


def test_tolerance_suppresses_small_drop() -> None:
    baseline = RunRecord(results=(_result([1.0, 1.0, 1.0, 1.0]),))
    candidate = RunRecord(results=(_result([1.0, 1.0, 1.0, 0.0]),))
    report = detect_regressions(baseline, candidate, tolerance=0.5)
    assert not report.has_regression


def test_regression_ignores_unmatched_tasks() -> None:
    baseline = RunRecord(results=(_result([1.0], task="a"),))
    candidate = RunRecord(results=(_result([0.0], task="b"),))
    report = detect_regressions(baseline, candidate)
    assert report.deltas == ()


# --------------------------------------------------------------------------- #
# Extended reporting
# --------------------------------------------------------------------------- #
def test_markdown_table() -> None:
    table = format_markdown_table([_result([1.0, 0.0])])
    assert table.startswith("| Model | Task |")
    assert "| m | t |" in table


def test_markdown_empty() -> None:
    assert "no results" in format_markdown_table([])


def test_html_report_is_self_contained() -> None:
    page = format_html_report([_result([1.0, 0.0])])
    assert page.startswith("<!doctype html>")
    assert "<table>" in page
    assert "Evaluation Report" in page


def test_html_escapes_content() -> None:
    page = format_html_report([_result([1.0], model="<script>")])
    assert "<script>" not in page.split("<style>")[1]
    assert "&lt;script&gt;" in page


def test_error_analysis_surfaces_worst() -> None:
    result = _result([1.0, 0.0, 1.0, 0.0, 1.0])
    analysis = analyze_errors(result, top_k=2)
    assert analysis.failures == 2
    assert analysis.failure_rate == 0.4
    assert len(analysis.worst_examples) == 2
    assert all(item.grade.score == 0.0 for item in analysis.worst_examples)


def test_format_error_analysis_text() -> None:
    text = format_error_analysis(analyze_errors(_result([1.0, 0.0])))
    assert "failures: 1/2" in text
    assert "worst examples" in text

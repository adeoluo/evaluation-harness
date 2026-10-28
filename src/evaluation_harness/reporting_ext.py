"""Markdown and HTML reports, plus error analysis.

The base :mod:`evaluation_harness.reporting` module renders text tables and JSON. This
module adds publication-ready outputs - a Markdown results table suitable for a README or
pull request, a self-contained HTML page, and an error-analysis summary that groups the
lowest-scoring examples for inspection.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

from evaluation_harness.core.result import ExampleResult, TaskResult


def format_markdown_table(results: Sequence[TaskResult], *, confidence: float = 0.95) -> str:
    """Render results as a GitHub-flavored Markdown table."""
    if not results:
        return "_(no results)_"
    lines = [
        "| Model | Task | Accuracy | N | Latency |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        accuracy = result.accuracy(confidence=confidence)
        lines.append(
            f"| {result.model_name} | {result.task_name} | "
            f"{accuracy.format()} | {result.count} | "
            f"{result.mean_latency_seconds * 1000:.1f} ms |"
        )
    return "\n".join(lines)


def format_html_report(results: Sequence[TaskResult], *, confidence: float = 0.95) -> str:
    """Render results as a self-contained HTML page."""
    rows = []
    for result in results:
        accuracy = result.accuracy(confidence=confidence)
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.model_name)}</td>"
            f"<td>{html.escape(result.task_name)}</td>"
            f"<td style='text-align:right'>{html.escape(accuracy.format())}</td>"
            f"<td style='text-align:right'>{result.count}</td>"
            f"<td style='text-align:right'>{result.mean_latency_seconds * 1000:.1f} ms</td>"
            "</tr>"
        )
    body = "\n".join(rows) or "<tr><td colspan='5'>(no results)</td></tr>"
    return (
        "<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
        "<title>Evaluation Report</title>\n"
        "<style>body{font-family:Inter,system-ui,sans-serif;margin:2rem;color:#1E293B}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #E2E8F0;padding:.5rem .75rem}"
        "th{background:#6366F1;color:#fff;text-align:left}"
        "tr:nth-child(even){background:#F8FAFC}</style>\n</head>\n<body>\n"
        "<h1>Evaluation Report</h1>\n<table>\n"
        "<thead><tr><th>Model</th><th>Task</th><th>Accuracy</th><th>N</th>"
        "<th>Latency</th></tr></thead>\n<tbody>\n"
        f"{body}\n</tbody>\n</table>\n</body>\n</html>\n"
    )


@dataclass(frozen=True, slots=True)
class ErrorAnalysis:
    """A summary of a task's failures for inspection."""

    task_name: str
    model_name: str
    total: int
    failures: int
    worst_examples: tuple[ExampleResult, ...]

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0


def analyze_errors(result: TaskResult, *, top_k: int = 5) -> ErrorAnalysis:
    """Summarize the lowest-scoring examples in a task result.

    Args:
        result: The task result to analyze.
        top_k: How many of the worst examples to surface.
    """
    ordered = sorted(result.example_results, key=lambda item: item.grade.score)
    failures = sum(1 for item in result.example_results if not item.grade.passed)
    return ErrorAnalysis(
        task_name=result.task_name,
        model_name=result.model_name,
        total=result.count,
        failures=failures,
        worst_examples=tuple(ordered[:top_k]),
    )


def format_error_analysis(analysis: ErrorAnalysis) -> str:
    """Render an error analysis as readable text."""
    lines = [
        f"Error analysis: {analysis.model_name} on {analysis.task_name}",
        f"  failures: {analysis.failures}/{analysis.total} ({analysis.failure_rate * 100:.1f}%)",
        "  worst examples:",
    ]
    for item in analysis.worst_examples:
        lines.append(
            f"    [{item.grade.score:.2f}] {item.example.input!r} "
            f"-> predicted {item.prediction!r}, expected {item.example.target!r}"
        )
    return "\n".join(lines)

"""Rendering evaluation results as text tables and JSON.

Reports are the product's public face: they must be unambiguous (every metric with its
confidence interval), comparable (models side by side on each task), and machine
readable (JSON for dashboards and regression tracking). Both renderers consume the same
:class:`~evaluation_harness.core.result.TaskResult` objects.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from evaluation_harness.core.result import TaskResult


def results_to_dict(results: Sequence[TaskResult], *, confidence: float = 0.95) -> dict[str, Any]:
    """Serialize results to a JSON-ready dictionary.

    The structure includes per-example records so a run is fully reproducible and
    auditable, alongside the aggregate accuracy and its confidence interval.
    """
    payload: list[dict[str, Any]] = []
    for result in results:
        accuracy = result.accuracy(confidence=confidence)
        payload.append(
            {
                "task": result.task_name,
                "model": result.model_name,
                "seed": result.seed,
                "count": result.count,
                "accuracy": {
                    "mean": accuracy.mean,
                    "low": accuracy.low,
                    "high": accuracy.high,
                    "confidence": accuracy.confidence,
                },
                "mean_latency_seconds": result.mean_latency_seconds,
                "examples": [
                    {
                        "id": item.example.identifier,
                        "prediction": item.prediction,
                        "target": item.example.target,
                        "score": item.grade.score,
                        "rationale": item.grade.rationale,
                    }
                    for item in result.example_results
                ],
            }
        )
    return {"confidence": confidence, "results": payload}


def results_to_json(
    results: Sequence[TaskResult], *, confidence: float = 0.95, indent: int = 2
) -> str:
    """Serialize results to a JSON string."""
    return json.dumps(results_to_dict(results, confidence=confidence), indent=indent)


def format_results_table(results: Sequence[TaskResult], *, confidence: float = 0.95) -> str:
    """Render results as an aligned, human-readable table.

    Columns: model, task, accuracy (mean with +/- margin), example count, and mean
    latency in milliseconds.
    """
    if not results:
        return "(no results)"

    headers = ("Model", "Task", f"Accuracy ({int(confidence * 100)}% CI)", "N", "Latency")
    rows: list[tuple[str, str, str, str, str]] = []
    for result in results:
        accuracy = result.accuracy(confidence=confidence)
        rows.append(
            (
                result.model_name,
                result.task_name,
                accuracy.format(),
                str(result.count),
                f"{result.mean_latency_seconds * 1000:.1f} ms",
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]

    def render(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))

    separator = "  ".join("-" * width for width in widths)
    lines = [render(headers), separator]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)

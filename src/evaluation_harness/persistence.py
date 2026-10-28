"""Persisting and reloading evaluation runs.

Evaluation is only cumulative if runs can be saved and reloaded. A :class:`RunRecord`
bundles the results of one evaluation session with metadata (a timestamp and an arbitrary
label such as a git commit), serializes losslessly to JSON, and reloads into the same
structured objects. This is the substrate for regression detection and historical
tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.core.result import ExampleResult, TaskResult


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A saved evaluation session: one or more task results plus metadata."""

    results: tuple[TaskResult, ...]
    label: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dictionary (lossless)."""
        return {
            "label": self.label,
            "created_at": self.created_at,
            "results": [_result_to_dict(result) for result in self.results],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunRecord:
        """Reconstruct a record from :meth:`to_dict` output."""
        return cls(
            results=tuple(_result_from_dict(item) for item in payload["results"]),
            label=payload.get("label", ""),
            created_at=payload.get("created_at", ""),
        )

    def save(self, path: Path) -> None:
        """Write the record to ``path`` as JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RunRecord:
        """Read a record previously written by :meth:`save`."""
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def result_for(self, task_name: str) -> TaskResult | None:
        """Return the result for ``task_name``, or ``None`` if absent."""
        for result in self.results:
            if result.task_name == task_name:
                return result
        return None


def _result_to_dict(result: TaskResult) -> dict[str, Any]:
    return {
        "task_name": result.task_name,
        "model_name": result.model_name,
        "seed": result.seed,
        "examples": [
            {
                "input": item.example.input,
                "target": item.example.target,
                "id": item.example.identifier,
                "prediction": item.prediction,
                "score": item.grade.score,
                "rationale": item.grade.rationale,
                "latency_seconds": item.latency_seconds,
            }
            for item in result.example_results
        ],
    }


def _result_from_dict(payload: dict[str, Any]) -> TaskResult:
    example_results = tuple(
        ExampleResult(
            example=Example(input=item["input"], target=item["target"], id=item.get("id")),
            prediction=item["prediction"],
            grade=Grade(item["score"], item.get("rationale")),
            latency_seconds=item.get("latency_seconds", 0.0),
        )
        for item in payload["examples"]
    )
    return TaskResult(
        task_name=payload["task_name"],
        model_name=payload["model_name"],
        example_results=example_results,
        seed=payload.get("seed", 0),
    )

"""The bundled ``entailment`` task: natural language inference.

Given a premise and a hypothesis, decide whether the premise entails, contradicts, or is
neutral toward the hypothesis. This three-way classification exercises the choice grader
with more than two options and provides a setting where a majority-class baseline is well
below a competent model.
"""

from __future__ import annotations

import json
from importlib import resources

from evaluation_harness.core.example import Example
from evaluation_harness.core.task import Task
from evaluation_harness.graders.choice import ChoiceGrader

_CHOICES = ("entailment", "contradiction", "neutral")
_DATA_PACKAGE = "evaluation_harness.tasks.data"
_DATA_FILE = "entailment.jsonl"


def build_entailment_task() -> Task:
    """Construct the entailment task from the bundled dataset."""
    examples = [
        Example(
            input=f"Premise: {record['premise']}\nHypothesis: {record['hypothesis']}",
            target=record["label"],
            id=f"entailment-{index:03d}",
            choices=_CHOICES,
            metadata={"source": _DATA_FILE},
        )
        for index, record in enumerate(_load_records())
    ]
    return Task(
        name="entailment",
        examples=examples,
        grader=ChoiceGrader(),
        description="Three-way natural language inference (entail / contradict / neutral).",
    )


def _load_records() -> list[dict[str, str]]:
    raw = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

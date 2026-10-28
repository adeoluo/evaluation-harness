"""The bundled ``sentiment`` task: binary sentiment classification.

Loads short, labeled review sentences from a bundled JSON Lines file and presents them
as a two-option multiple-choice task. It exercises the choice grader and provides a
setting where a lexicon heuristic should clearly beat a majority-class baseline.
"""

from __future__ import annotations

import json
from importlib import resources

from evaluation_harness.core.example import Example
from evaluation_harness.core.task import Task
from evaluation_harness.graders.choice import ChoiceGrader

_CHOICES = ("positive", "negative")
_DATA_PACKAGE = "evaluation_harness.tasks.data"
_DATA_FILE = "sentiment.jsonl"


def build_sentiment_task() -> Task:
    """Construct the sentiment task from the bundled dataset."""
    examples = [
        Example(
            input=record["text"],
            target=record["label"],
            id=f"sentiment-{index:03d}",
            choices=_CHOICES,
            metadata={"source": _DATA_FILE},
        )
        for index, record in enumerate(_load_records())
    ]
    return Task(
        name="sentiment",
        examples=examples,
        grader=ChoiceGrader(),
        description="Binary sentiment classification over short review sentences.",
    )


def _load_records() -> list[dict[str, str]]:
    raw = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE).read_text(encoding="utf-8")
    records: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records

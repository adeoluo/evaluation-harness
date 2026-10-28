"""The bundled ``factual_qa`` task: short-answer factual questions.

Open-ended short answers are the natural home for the token-F1 grader: "Franklin D.
Roosevelt" and "Franklin Delano Roosevelt" should earn credit for their overlap. The
task pairs each question with a reference answer and is graded by token F1 with a
threshold, so partial overlap is rewarded without accepting unrelated text.
"""

from __future__ import annotations

import json
from importlib import resources

from evaluation_harness.core.example import Example
from evaluation_harness.core.task import Task
from evaluation_harness.graders.f1 import F1Grader

_DATA_PACKAGE = "evaluation_harness.tasks.data"
_DATA_FILE = "factual_qa.jsonl"


def build_factual_qa_task(*, threshold: float = 0.5) -> Task:
    """Construct the factual question-answering task.

    Args:
        threshold: Token-F1 threshold above which an answer counts as correct.
    """
    examples = [
        Example(
            input=record["question"],
            target=record["answer"],
            id=f"factual-qa-{index:03d}",
            metadata={"source": _DATA_FILE},
        )
        for index, record in enumerate(_load_records())
    ]
    return Task(
        name="factual_qa",
        examples=examples,
        grader=F1Grader(threshold=threshold),
        description="Short-answer factual questions graded by token F1.",
    )


def _load_records() -> list[dict[str, str]]:
    raw = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

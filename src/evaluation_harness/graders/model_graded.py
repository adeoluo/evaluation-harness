"""Model-graded (rubric) grading.

Some answers cannot be checked by string or numeric rules - open-ended explanations,
summaries, or reasoning. Here a separate *judge* model scores the prediction against a
rubric. Because the judge is injected as an ordinary :class:`Model`, this grader is
fully testable offline with a deterministic stub, and in production can use a stronger
model than the one under evaluation.

The judge is prompted to end with a machine-parseable verdict, which this grader
extracts defensively (it tolerates extra commentary around the verdict).
"""

from __future__ import annotations

import re

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.core.model import Model

DEFAULT_RUBRIC = (
    "You are grading whether a response correctly answers a question.\n"
    "Question:\n{input}\n\n"
    "Reference answer:\n{target}\n\n"
    "Response to grade:\n{prediction}\n\n"
    "Decide if the response is correct. Reply with a single final line in the form "
    "'Verdict: correct' or 'Verdict: incorrect'."
)

_VERDICT = re.compile(r"verdict\s*[:\-]?\s*(correct|incorrect|yes|no)", re.IGNORECASE)
_SCORE = re.compile(r"score\s*[:\-]?\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)
_POSITIVE = {"correct", "yes"}


class ModelGradedGrader:
    """Scores predictions with a judge model against a rubric prompt.

    Args:
        judge: The model used to grade.
        rubric: A format string with ``{input}``, ``{target}``, and ``{prediction}``
            placeholders.
    """

    def __init__(self, judge: Model, *, rubric: str = DEFAULT_RUBRIC) -> None:
        self._judge = judge
        self._rubric = rubric

    @property
    def name(self) -> str:
        return "model_graded"

    def grade(self, example: Example, prediction: str) -> Grade:
        prompt = self._rubric.format(
            input=example.input,
            target=example.target,
            prediction=prediction,
        )
        judge_example = Example(input=prompt, target="")
        verdict_text = self._judge.generate(judge_example)
        return self._parse(verdict_text)

    @staticmethod
    def _parse(verdict_text: str) -> Grade:
        score_match = _SCORE.search(verdict_text)
        if score_match:
            return Grade(float(score_match.group(1)), rationale=verdict_text.strip())

        verdict_match = _VERDICT.search(verdict_text)
        if verdict_match:
            positive = verdict_match.group(1).lower() in _POSITIVE
            return Grade(1.0 if positive else 0.0, rationale=verdict_text.strip())

        raise ValueError(f"Judge output did not contain a parseable verdict: {verdict_text!r}")

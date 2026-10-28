"""Tests for model-graded (rubric) grading with a deterministic judge."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from evaluation_harness.core import Example
from evaluation_harness.graders.model_graded import ModelGradedGrader


@dataclass(frozen=True)
class ScriptedJudge:
    """A judge model that returns a fixed verdict string, for deterministic tests."""

    verdict: str
    name: str = "scripted-judge"

    def generate(self, example: Example) -> str:
        return self.verdict


def _example() -> Example:
    return Example(input="What is 2+2?", target="4")


def test_judge_correct_verdict() -> None:
    grader = ModelGradedGrader(ScriptedJudge("Reasoning... Verdict: correct"))
    assert grader.grade(_example(), "4").passed


def test_judge_incorrect_verdict() -> None:
    grader = ModelGradedGrader(ScriptedJudge("Verdict: incorrect"))
    assert grader.grade(_example(), "5").score == 0.0


def test_judge_yes_is_positive() -> None:
    grader = ModelGradedGrader(ScriptedJudge("verdict: yes"))
    assert grader.grade(_example(), "4").passed


def test_judge_numeric_score_is_used() -> None:
    grader = ModelGradedGrader(ScriptedJudge("Score: 0.5 - partially correct"))
    grade = grader.grade(_example(), "about 4")
    assert grade.score == 0.5


def test_judge_unparseable_output_raises() -> None:
    grader = ModelGradedGrader(ScriptedJudge("I have no opinion"))
    with pytest.raises(ValueError, match="parseable verdict"):
        grader.grade(_example(), "4")


def test_judge_receives_rendered_rubric() -> None:
    captured: dict[str, str] = {}

    @dataclass(frozen=True)
    class CapturingJudge:
        name: str = "capturing"

        def generate(self, example: Example) -> str:
            captured["prompt"] = example.input
            return "Verdict: correct"

    ModelGradedGrader(CapturingJudge()).grade(_example(), "4")
    assert "What is 2+2?" in captured["prompt"]
    assert "Response to grade:" in captured["prompt"]

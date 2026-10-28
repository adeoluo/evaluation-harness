"""Shared test fixtures and helpers."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation_harness.core import Example, Grade


@dataclass(frozen=True)
class FixedModel:
    """A model that returns the same string for every example."""

    name: str
    reply: str

    def generate(self, example: Example) -> str:
        return self.reply


@dataclass(frozen=True)
class EchoTargetModel:
    """An oracle model that always returns the example's target (scores 1.0)."""

    name: str = "oracle"

    def generate(self, example: Example) -> str:
        return example.target


class AlwaysPassGrader:
    """A grader that awards full marks to any prediction."""

    name = "always_pass"

    def grade(self, example: Example, prediction: str) -> Grade:
        return Grade(1.0)


class ExactGrader:
    """Minimal exact-match grader for exercising the core loop."""

    name = "exact"

    def grade(self, example: Example, prediction: str) -> Grade:
        return Grade(1.0 if prediction.strip() == example.target.strip() else 0.0)

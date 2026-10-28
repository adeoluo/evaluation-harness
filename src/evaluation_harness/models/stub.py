"""Deterministic, offline models.

These exist so the entire framework - runner, graders, statistics, reporting, CLI -
can be exercised and demonstrated with no network access or API keys, which keeps
continuous integration hermetic and the published numbers reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation_harness.core.example import Example


@dataclass(frozen=True)
class ConstantModel:
    """Always returns the same answer. A deliberate weak baseline / sanity check."""

    answer: str
    name: str = "constant"

    def generate(self, example: Example) -> str:
        return self.answer


@dataclass(frozen=True)
class EchoModel:
    """Returns the example input unchanged. Useful for pipeline tests."""

    name: str = "echo"

    def generate(self, example: Example) -> str:
        return example.input

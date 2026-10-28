"""Grader that matches the prediction against a regular expression."""

from __future__ import annotations

import re

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade


class RegexGrader:
    """Awards full marks when a pattern matches the prediction.

    The pattern may be supplied at construction (shared across the task) or read
    per-example from ``example.metadata[pattern_key]``, enabling answer-specific
    patterns. A per-example pattern takes precedence when present.
    """

    def __init__(
        self,
        pattern: str | None = None,
        *,
        flags: int = 0,
        pattern_key: str = "pattern",
    ) -> None:
        if pattern is None and not pattern_key:
            raise ValueError("Provide a pattern or a non-empty pattern_key")
        self._pattern = re.compile(pattern, flags) if pattern is not None else None
        self._flags = flags
        self._pattern_key = pattern_key

    @property
    def name(self) -> str:
        return "regex"

    def grade(self, example: Example, prediction: str) -> Grade:
        pattern = self._resolve_pattern(example)
        correct = pattern.search(prediction) is not None
        rationale = None if correct else f"no match for /{pattern.pattern}/"
        return Grade(1.0 if correct else 0.0, rationale)

    def _resolve_pattern(self, example: Example) -> re.Pattern[str]:
        raw = example.metadata.get(self._pattern_key)
        if isinstance(raw, str):
            return re.compile(raw, self._flags)
        if self._pattern is None:
            raise ValueError(
                f"No pattern for example {example.identifier!r}: provide a default "
                f"pattern or set metadata[{self._pattern_key!r}]"
            )
        return self._pattern

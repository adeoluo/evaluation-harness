"""Graders for structured (JSON) model output.

As models increasingly emit structured data, evaluation must check structure, not just
surface text. :class:`JsonValidGrader` checks that the prediction parses as JSON and,
optionally, that required keys are present. :class:`JsonFieldGrader` extracts a specific
field and compares it to the target, so a single field can be graded regardless of the
surrounding JSON. Both tolerate JSON embedded in surrounding prose (for example inside a
fenced code block), which is how models usually return it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.graders.normalization import normalize_text

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Return the first JSON value found in ``text``, or ``None`` if there is none.

    Tries, in order: a fenced code block, the whole string, then the first balanced
    ``{...}`` or ``[...]`` span.
    """
    fence = _FENCE.search(text)
    candidates: list[str] = []
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text.strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


class JsonValidGrader:
    """Awards full marks when the prediction is valid JSON with all required keys.

    Args:
        required_keys: Keys that must be present at the top level of a JSON object.
    """

    def __init__(self, *, required_keys: tuple[str, ...] = ()) -> None:
        self._required_keys = required_keys

    @property
    def name(self) -> str:
        return "json_valid"

    def grade(self, example: Example, prediction: str) -> Grade:
        value = extract_json(prediction)
        if value is None:
            return Grade(0.0, "no valid JSON found")
        if self._required_keys:
            if not isinstance(value, dict):
                return Grade(0.0, "JSON is not an object")
            missing = [key for key in self._required_keys if key not in value]
            if missing:
                return Grade(0.0, f"missing keys: {', '.join(missing)}")
        return Grade(1.0)


class JsonFieldGrader:
    """Extracts a top-level field from JSON output and compares it to the target."""

    def __init__(self, field: str) -> None:
        if not field:
            raise ValueError("field must be a non-empty key")
        self._field = field

    @property
    def name(self) -> str:
        return "json_field"

    def grade(self, example: Example, prediction: str) -> Grade:
        value = extract_json(prediction)
        if not isinstance(value, dict):
            return Grade(0.0, "prediction is not a JSON object")
        if self._field not in value:
            return Grade(0.0, f"field {self._field!r} not present")
        actual = normalize_text(str(value[self._field]))
        expected = normalize_text(example.target)
        passed = actual == expected
        rationale = None if passed else f"expected {expected!r}, got {actual!r}"
        return Grade(1.0 if passed else 0.0, rationale)

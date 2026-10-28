"""Grader for numeric answers with absolute or relative tolerance.

Math benchmarks are routinely mis-scored by exact string matching: ``"1,000"``,
``"1000"``, ``"1000.0"``, and "the answer is 1000" are the same number. This grader
extracts the final number from the prediction and compares it numerically, which is
both more correct and more faithful to what the task actually measures.
"""

from __future__ import annotations

import re

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade

# Matches integers and decimals, with optional sign and thousands separators.
_NUMBER = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?|[-+]?\.\d+")


def extract_last_number(text: str) -> float | None:
    """Return the last numeric value in ``text``, or ``None`` if there is none.

    The *last* number is used because models typically state their final answer at
    the end of a chain of reasoning.
    """
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


class NumericGrader:
    """Awards full marks when the predicted number is within tolerance of the target.

    Args:
        absolute_tolerance: Maximum allowed absolute difference.
        relative_tolerance: Maximum allowed difference relative to the target's
            magnitude. The prediction passes if it is within *either* tolerance.
    """

    def __init__(
        self,
        *,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 0.0,
    ) -> None:
        if absolute_tolerance < 0 or relative_tolerance < 0:
            raise ValueError("Tolerances must be non-negative")
        self._atol = absolute_tolerance
        self._rtol = relative_tolerance

    @property
    def name(self) -> str:
        return "numeric"

    def grade(self, example: Example, prediction: str) -> Grade:
        target = extract_last_number(example.target)
        if target is None:
            raise ValueError(
                f"Target of example {example.identifier!r} has no number: {example.target!r}"
            )
        predicted = extract_last_number(prediction)
        if predicted is None:
            return Grade(0.0, "no number found in prediction")

        difference = abs(predicted - target)
        allowed = max(self._atol, self._rtol * abs(target))
        correct = difference <= allowed
        rationale = None if correct else f"expected {target}, got {predicted}"
        return Grade(1.0 if correct else 0.0, rationale)

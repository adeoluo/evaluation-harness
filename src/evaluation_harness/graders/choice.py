"""Grader for multiple-choice tasks.

Resolves both the target and the prediction to a choice index, then compares. It
accepts either a letter answer (``A``, ``(B)``, ``Answer: C``) or the full choice
text, so it works whether a model replies with a label or restates the option.
"""

from __future__ import annotations

import re
import string

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.graders.normalization import normalize_text

_LETTERS = string.ascii_uppercase


class ChoiceGrader:
    """Awards full marks when the selected option matches the correct option.

    The example must carry ``choices``. The ``target`` may be a letter (``"B"``) or
    the exact text of the correct option.
    """

    @property
    def name(self) -> str:
        return "choice"

    def grade(self, example: Example, prediction: str) -> Grade:
        if not example.choices:
            raise ValueError(
                f"ChoiceGrader requires choices; example {example.identifier!r} has none"
            )
        choices = example.choices
        correct_index = self._resolve_target(example.target, choices)
        selected_index = self._resolve_prediction(prediction, choices)

        if selected_index is None:
            return Grade(0.0, "no option could be parsed from the prediction")
        correct = selected_index == correct_index
        rationale = (
            None
            if correct
            else f"selected {_LETTERS[selected_index]}, correct is {_LETTERS[correct_index]}"
        )
        return Grade(1.0 if correct else 0.0, rationale)

    @staticmethod
    def _resolve_target(target: str, choices: tuple[str, ...]) -> int:
        stripped = target.strip()
        if len(stripped) == 1 and stripped.upper() in _LETTERS[: len(choices)]:
            return _LETTERS.index(stripped.upper())
        normalized_target = normalize_text(target)
        for index, choice in enumerate(choices):
            if normalize_text(choice) == normalized_target:
                return index
        raise ValueError(f"Target {target!r} does not match any choice")

    @staticmethod
    def _resolve_prediction(prediction: str, choices: tuple[str, ...]) -> int | None:
        valid_letters = _LETTERS[: len(choices)]
        # Prefer an explicit letter, for example "Answer: C" or "(B)".
        letter_match = re.search(rf"\b([{valid_letters}])\b", prediction.upper())
        if letter_match:
            return _LETTERS.index(letter_match.group(1))
        # Fall back to matching the full option text.
        normalized_prediction = normalize_text(prediction)
        for index, choice in enumerate(choices):
            if normalize_text(choice) in normalized_prediction:
                return index
        return None

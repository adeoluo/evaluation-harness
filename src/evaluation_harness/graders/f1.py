"""Token-level F1 grader for extractive and short-answer tasks.

Exact match is too strict for answers phrased in different but overlapping words
("Franklin Delano Roosevelt" versus "Franklin D. Roosevelt"). Token F1 - the harmonic
mean of token precision and recall after normalization - is the standard partial-credit
metric from the extractive question-answering literature (for example SQuAD) and returns
a graded score in ``[0, 1]`` rather than a bare pass/fail.
"""

from __future__ import annotations

from collections import Counter

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.graders.normalization import normalize_text


def _tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def token_f1(prediction: str, target: str) -> float:
    """Return the token-level F1 score between ``prediction`` and ``target``."""
    predicted_tokens = _tokens(prediction)
    target_tokens = _tokens(target)
    if not predicted_tokens and not target_tokens:
        return 1.0
    if not predicted_tokens or not target_tokens:
        return 0.0

    common = Counter(predicted_tokens) & Counter(target_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


class F1Grader:
    """Scores a prediction by token-level F1 against the target.

    Args:
        threshold: If set, the continuous F1 is binarized to ``1.0`` when it meets or
            exceeds the threshold and ``0.0`` otherwise. If ``None`` (the default), the
            continuous F1 score is returned, giving partial credit.
    """

    def __init__(self, *, threshold: float | None = None) -> None:
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1] or None")
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "f1"

    def grade(self, example: Example, prediction: str) -> Grade:
        score = token_f1(prediction, example.target)
        if self._threshold is not None:
            passed = score >= self._threshold
            return Grade(1.0 if passed else 0.0, f"token F1 {score:.2f}")
        return Grade(score, f"token F1 {score:.2f}")

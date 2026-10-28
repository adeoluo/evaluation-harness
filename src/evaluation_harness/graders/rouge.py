"""ROUGE-L grader for summary and long-answer overlap.

ROUGE-L scores the longest common subsequence (LCS) between prediction and reference,
rewarding answers that preserve the reference's word order without requiring contiguous
matches. It is the standard recall-oriented metric for summarization and long-form
generation, and it returns graded partial credit.
"""

from __future__ import annotations

from evaluation_harness.core.example import Example
from evaluation_harness.core.grader import Grade
from evaluation_harness.graders.normalization import normalize_text


def lcs_length(a: list[str], b: list[str]) -> int:
    """Return the length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, target: str, *, beta: float = 1.2) -> float:
    """Return the ROUGE-L F-measure between ``prediction`` and ``target``.

    ``beta`` weights recall relative to precision, following the original ROUGE paper's
    convention of favoring recall.
    """
    predicted_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not predicted_tokens and not target_tokens:
        return 1.0
    if not predicted_tokens or not target_tokens:
        return 0.0
    lcs = lcs_length(predicted_tokens, target_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(predicted_tokens)
    recall = lcs / len(target_tokens)
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (recall + beta_squared * precision)


class RougeLGrader:
    """Scores a prediction by ROUGE-L F-measure against the target."""

    def __init__(self, *, threshold: float | None = None, beta: float = 1.2) -> None:
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1] or None")
        self._threshold = threshold
        self._beta = beta

    @property
    def name(self) -> str:
        return "rouge_l"

    def grade(self, example: Example, prediction: str) -> Grade:
        score = rouge_l(prediction, example.target, beta=self._beta)
        if self._threshold is not None:
            passed = score >= self._threshold
            return Grade(1.0 if passed else 0.0, f"ROUGE-L {score:.2f}")
        return Grade(score, f"ROUGE-L {score:.2f}")

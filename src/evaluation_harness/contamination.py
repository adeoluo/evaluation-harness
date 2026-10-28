"""N-gram contamination detection.

A benchmark result is meaningless if the test items leaked into the model's training
data. This module provides the standard first line of defense: n-gram *containment*
between each evaluation example and a reference corpus (for example a sample of the
training set). An example is flagged when a large fraction of its n-grams already
appear in the reference, which is a strong signal of overlap.

Containment, rather than Jaccard similarity, is used because an example can be fully
contained in a much larger corpus; symmetric similarity would wash that signal out.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evaluation_harness.core.task import Task

_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class FlaggedExample:
    """An example whose n-gram containment exceeds the threshold."""

    example_id: str
    containment: float


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    """The outcome of a contamination scan over a task.

    Attributes:
        task_name: The scanned task.
        n: The n-gram size used.
        threshold: Containment above which an example is flagged.
        flagged: Flagged examples, sorted by descending containment.
        total: Total number of examples scanned.
    """

    task_name: str
    n: int
    threshold: float
    flagged: Sequence[FlaggedExample]
    total: int

    @property
    def flagged_count(self) -> int:
        return len(self.flagged)

    @property
    def flagged_fraction(self) -> float:
        return self.flagged_count / self.total if self.total else 0.0

    @property
    def is_clean(self) -> bool:
        """True when no example exceeded the contamination threshold."""
        return self.flagged_count == 0


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization used for n-gram construction."""
    return _TOKEN.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Return the set of ``n``-grams in ``tokens`` (empty if too few tokens)."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def containment(text: str, reference_ngrams: set[tuple[str, ...]], *, n: int) -> float:
    """Fraction of ``text``'s n-grams that appear in ``reference_ngrams``.

    Returns ``0.0`` when ``text`` has no n-grams of size ``n``.
    """
    grams = ngrams(tokenize(text), n)
    if not grams:
        return 0.0
    overlap = len(grams & reference_ngrams)
    return overlap / len(grams)


def detect_contamination(
    task: Task,
    reference_texts: Iterable[str],
    *,
    n: int = 13,
    threshold: float = 0.5,
) -> ContaminationReport:
    """Scan ``task`` for overlap against a reference corpus.

    Args:
        task: The benchmark to scan.
        reference_texts: Documents representing (a sample of) the training data.
        n: N-gram size. The default of 13 follows common practice for decontamination.
        threshold: Containment above which an example is flagged.

    Returns:
        A :class:`ContaminationReport`.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    reference_ngrams: set[tuple[str, ...]] = set()
    for text in reference_texts:
        reference_ngrams |= ngrams(tokenize(text), n)

    flagged: list[FlaggedExample] = []
    for example in task.examples:
        probe = f"{example.input} {example.target}"
        score = containment(probe, reference_ngrams, n=n)
        if score >= threshold:
            flagged.append(FlaggedExample(example.identifier, score))

    flagged.sort(key=lambda item: item.containment, reverse=True)
    return ContaminationReport(
        task_name=task.name,
        n=n,
        threshold=threshold,
        flagged=tuple(flagged),
        total=len(task),
    )

"""Statistical aggregation for evaluation scores.

A single accuracy number is not a result - its uncertainty is. This module computes
the mean of a set of per-example scores together with a bootstrap confidence interval,
so every reported metric carries an honest error bar.

The bootstrap is used because evaluation scores are often non-normal (bounded in
``[0, 1]``, frequently bimodal for pass/fail graders), where the normal approximation
of the standard error is unreliable, especially on small benchmarks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A point estimate with a bootstrap confidence interval.

    Attributes:
        mean: The sample mean of the scores.
        low: Lower bound of the confidence interval.
        high: Upper bound of the confidence interval.
        confidence: The interval's confidence level (for example ``0.95``).
        count: Number of scores aggregated.
    """

    mean: float
    low: float
    high: float
    confidence: float
    count: int

    @property
    def margin(self) -> float:
        """Half-width of the interval, i.e. the plus-or-minus error bar."""
        return (self.high - self.low) / 2.0

    def format(self, *, pct: bool = True, digits: int = 1) -> str:
        """Render as ``mean +/- margin`` (as a percentage by default)."""
        scale = 100.0 if pct else 1.0
        suffix = "%" if pct else ""
        return (
            f"{self.mean * scale:.{digits}f}{suffix} +/- {self.margin * scale:.{digits}f}{suffix}"
        )


def bootstrap_aggregate(
    scores: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> Aggregate:
    """Aggregate ``scores`` into a mean and a percentile bootstrap interval.

    Args:
        scores: Per-example scores, each in ``[0, 1]``.
        confidence: Confidence level for the interval; must be in ``(0, 1)``.
        resamples: Number of bootstrap resamples.
        seed: Seed for the resampling generator, for reproducibility.

    Returns:
        An :class:`Aggregate` with the mean and interval bounds.

    Raises:
        ValueError: If ``scores`` is empty or ``confidence`` is out of range.
    """
    if not scores:
        raise ValueError("Cannot aggregate an empty sequence of scores")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    values = np.asarray(scores, dtype=np.float64)
    mean = float(values.mean())

    if values.size == 1:
        # A single observation carries no spread; the interval collapses to the point.
        return Aggregate(mean, mean, mean, confidence, 1)

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    resample_means = values[indices].mean(axis=1)

    alpha = 1.0 - confidence
    low = float(np.quantile(resample_means, alpha / 2.0))
    high = float(np.quantile(resample_means, 1.0 - alpha / 2.0))
    return Aggregate(mean, low, high, confidence, values.size)

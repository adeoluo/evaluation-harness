"""Statistical significance testing for model comparisons.

A difference in accuracy is only meaningful if it is unlikely to be noise. This module
provides paired significance tests over per-example scores, which are more powerful than
unpaired tests because the two models are evaluated on the *same* examples:

- :func:`paired_bootstrap_test` - a general paired bootstrap for any score in ``[0, 1]``.
- :func:`mcnemar_test` - the exact test for paired binary (pass/fail) outcomes.
- :func:`cohens_d` - a standardized effect size, so "significant" is reported alongside
  "how large".

All randomness is seeded for reproducibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The outcome of comparing two models on the same examples.

    Attributes:
        mean_difference: Mean of (model A score - model B score) per example.
        p_value: Probability of a difference this large under the null hypothesis.
        significant: Whether ``p_value`` is below the chosen significance level.
        effect_size: Cohen's d for the paired difference.
        test: Name of the test used.
    """

    mean_difference: float
    p_value: float
    significant: bool
    effect_size: float
    test: str


def _validate_pair(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError("paired tests require equal-length score sequences")
    if not a:
        raise ValueError("cannot test empty score sequences")


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Return Cohen's d for the paired difference ``a - b``.

    Uses the standard deviation of the per-example differences (a paired effect size).
    Returns ``0.0`` when there is no variance and no mean difference.
    """
    _validate_pair(a, b)
    differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    std = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    mean = float(differences.mean())
    if std == 0.0:
        return 0.0
    return mean / std


def paired_bootstrap_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ComparisonResult:
    """Paired bootstrap test of whether model A outscores model B.

    The two-sided p-value is twice the smaller tail probability that a bootstrap resample
    of the per-example differences has a mean on the opposite side of zero.
    """
    _validate_pair(a, b)
    differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = float(differences.mean())

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(resamples, differences.size))
    resample_means = differences[indices].mean(axis=1)

    # Fraction of resamples on the opposite side of zero from the observed difference.
    if observed >= 0:
        tail = float(np.mean(resample_means <= 0.0))
    else:
        tail = float(np.mean(resample_means >= 0.0))
    p_value = min(1.0, 2.0 * tail)

    return ComparisonResult(
        mean_difference=observed,
        p_value=p_value,
        significant=p_value < alpha,
        effect_size=cohens_d(a, b),
        test="paired_bootstrap",
    )


def mcnemar_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Exact McNemar test for paired binary outcomes.

    Scores are treated as pass (>= 0.5) or fail. The test considers only the discordant
    pairs - examples where exactly one model is correct - and computes an exact two-sided
    binomial p-value. This is the right test for comparing two classifiers on a shared
    test set.
    """
    _validate_pair(a, b)
    a_pass = [score >= 0.5 for score in a]
    b_pass = [score >= 0.5 for score in b]
    a_only = sum(1 for pa, pb in zip(a_pass, b_pass, strict=True) if pa and not pb)
    b_only = sum(1 for pa, pb in zip(a_pass, b_pass, strict=True) if pb and not pa)

    n = a_only + b_only
    if n == 0:
        p_value = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(comb(n, i) for i in range(k + 1)) * (0.5**n)
        p_value = min(1.0, 2.0 * tail)

    mean_difference = float(np.mean(a) - np.mean(b))
    return ComparisonResult(
        mean_difference=mean_difference,
        p_value=p_value,
        significant=p_value < alpha,
        effect_size=cohens_d(a, b),
        test="mcnemar",
    )

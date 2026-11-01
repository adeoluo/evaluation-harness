"""Independent validation of the from-scratch statistics against ``scipy.stats``.

The significance and aggregation code is written from scratch (bootstrap
confidence intervals, a paired bootstrap test, the exact McNemar test, and
Cohen's d). Correctness is only credible if it agrees with an established
reference, so this test cross-checks each quantity against ``scipy.stats`` on
shared synthetic data:

- **Cohen's d** equals the paired t-statistic divided by ``sqrt(n)``
  (``scipy.stats.ttest_rel``) to machine precision.
- **McNemar's exact p-value** equals the two-sided binomial p-value over the
  discordant pairs (``scipy.stats.binomtest``) to machine precision.
- **The bootstrap confidence interval** matches ``scipy.stats.bootstrap`` with
  the percentile method within Monte-Carlo tolerance.
- **The paired bootstrap p-value** tracks ``scipy.stats.ttest_rel`` - the same
  significance decision on every case and a close p-value on separated data.

``scipy`` is an optional ``validation`` dependency, not a runtime dependency, so
this test is skipped when scipy is not installed and the significance code stays
reference-free. (``numpy`` is already a runtime dependency and is used directly.)
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from scipy import stats

from evaluation_harness.significance import (
    cohens_d,
    mcnemar_test,
    paired_bootstrap_test,
)
from evaluation_harness.statistics import bootstrap_aggregate


def _paired_samples(seed: int, n: int, shift: float) -> tuple[list[float], list[float]]:
    """Return two correlated real-valued score vectors on the same examples."""

    rng = np.random.default_rng(seed)
    base = rng.random(n)
    a = np.clip(base + rng.normal(shift, 0.1, size=n), 0.0, 1.0)
    b = np.clip(base + rng.normal(0.0, 0.1, size=n), 0.0, 1.0)
    return a.tolist(), b.tolist()


def _binary_pair(seed: int, n: int, p_a: float, p_b: float) -> tuple[list[float], list[float]]:
    """Return two binary (pass/fail) score vectors with independent noise."""

    rng = np.random.default_rng(seed)
    a = (rng.random(n) < p_a).astype(float)
    b = (rng.random(n) < p_b).astype(float)
    return a.tolist(), b.tolist()


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 99])
def test_cohens_d_matches_scipy_ttest(seed: int) -> None:
    a, b = _paired_samples(seed, n=64, shift=0.15)
    reference = float(stats.ttest_rel(a, b).statistic) / np.sqrt(len(a))
    assert cohens_d(a, b) == pytest.approx(reference, abs=1e-9)


@pytest.mark.parametrize(
    ("seed", "p_a", "p_b"),
    [(0, 0.8, 0.5), (1, 0.6, 0.6), (3, 0.9, 0.4), (5, 0.55, 0.5), (8, 0.7, 0.3)],
)
def test_mcnemar_matches_scipy_binomtest(seed: int, p_a: float, p_b: float) -> None:
    a, b = _binary_pair(seed, n=120, p_a=p_a, p_b=p_b)
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x >= 0.5 and y < 0.5)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y >= 0.5 and x < 0.5)
    n = a_only + b_only
    reference = 1.0 if n == 0 else float(stats.binomtest(min(a_only, b_only), n, 0.5).pvalue)
    assert mcnemar_test(a, b).p_value == pytest.approx(reference, abs=1e-9)


def test_mcnemar_no_discordant_pairs_matches_scipy() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]
    # No discordant pairs: both the from-scratch test and binomtest give p = 1.
    assert mcnemar_test(scores, scores).p_value == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 4, 11])
def test_bootstrap_ci_matches_scipy_bootstrap(seed: int) -> None:
    rng = np.random.default_rng(seed)
    data = rng.random(96)
    ours = bootstrap_aggregate(data.tolist(), confidence=0.95, resamples=20_000, seed=seed)
    reference = stats.bootstrap(
        (data,),
        np.mean,
        confidence_level=0.95,
        method="percentile",
        n_resamples=20_000,
        random_state=seed + 100,
    )
    # Two independent percentile bootstraps over the same data agree within a
    # small Monte-Carlo tolerance; the mean is exact.
    assert ours.mean == pytest.approx(float(data.mean()), abs=1e-12)
    assert ours.low == pytest.approx(float(reference.confidence_interval.low), abs=0.02)
    assert ours.high == pytest.approx(float(reference.confidence_interval.high), abs=0.02)


@pytest.mark.parametrize(
    ("seed", "shift"),
    [(0, 0.0), (1, 0.05), (2, 0.2), (7, 0.3), (99, -0.25)],
)
def test_paired_bootstrap_pvalue_tracks_scipy_ttest(seed: int, shift: float) -> None:
    a, b = _paired_samples(seed, n=80, shift=shift)
    ours = paired_bootstrap_test(a, b, resamples=20_000, seed=seed)
    reference = stats.ttest_rel(a, b)

    # Mean difference is computed identically.
    assert ours.mean_difference == pytest.approx(float(np.mean(a) - np.mean(b)), abs=1e-12)
    # The bootstrap p-value and the parametric p-value agree on significance at
    # alpha = 0.05 and are numerically close (both are two-sided tests of the
    # same paired difference).
    assert ours.significant == bool(reference.pvalue < 0.05)
    assert ours.p_value == pytest.approx(float(reference.pvalue), abs=0.05)

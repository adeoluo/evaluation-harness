"""Tests for statistical significance testing."""

from __future__ import annotations

import pytest

from evaluation_harness.significance import (
    cohens_d,
    mcnemar_test,
    paired_bootstrap_test,
)


def test_identical_models_not_significant() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.0] * 10
    result = paired_bootstrap_test(scores, scores, seed=1)
    assert result.mean_difference == 0.0
    assert not result.significant
    assert result.p_value == 1.0


def test_clear_difference_is_significant() -> None:
    a = [1.0] * 50
    b = [0.0] * 50
    result = paired_bootstrap_test(a, b, seed=1)
    assert result.mean_difference == 1.0
    assert result.significant
    assert result.p_value < 0.05


def test_bootstrap_is_reproducible() -> None:
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    b = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    first = paired_bootstrap_test(a, b, seed=42)
    second = paired_bootstrap_test(a, b, seed=42)
    assert first == second


def test_mcnemar_clear_difference() -> None:
    a = [1.0] * 30 + [0.0] * 2
    b = [0.0] * 30 + [0.0] * 2
    result = mcnemar_test(a, b)
    assert result.significant
    assert result.test == "mcnemar"


def test_mcnemar_no_discordant_pairs() -> None:
    scores = [1.0, 0.0, 1.0]
    result = mcnemar_test(scores, scores)
    assert result.p_value == 1.0
    assert not result.significant


def test_cohens_d_zero_when_identical() -> None:
    scores = [1.0, 0.0, 1.0, 0.0]
    assert cohens_d(scores, scores) == 0.0


def test_cohens_d_sign_follows_difference() -> None:
    a = [1.0, 1.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0, 0.0]
    assert cohens_d(a, b) > 0.0


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_test([1.0, 0.0], [1.0])


def test_empty_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        mcnemar_test([], [])

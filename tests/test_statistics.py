"""Tests for bootstrap statistical aggregation."""

from __future__ import annotations

import pytest

from evaluation_harness.statistics import Aggregate, bootstrap_aggregate


def test_mean_is_exact() -> None:
    aggregate = bootstrap_aggregate([0.0, 1.0, 1.0, 1.0])
    assert aggregate.mean == pytest.approx(0.75)
    assert aggregate.count == 4


def test_interval_brackets_the_mean() -> None:
    aggregate = bootstrap_aggregate([0.0, 1.0] * 50, seed=7)
    assert aggregate.low <= aggregate.mean <= aggregate.high


def test_all_correct_has_degenerate_interval() -> None:
    aggregate = bootstrap_aggregate([1.0] * 20)
    assert aggregate.mean == 1.0
    assert aggregate.low == 1.0
    assert aggregate.high == 1.0


def test_single_observation_collapses_interval() -> None:
    aggregate = bootstrap_aggregate([0.4])
    assert aggregate.mean == pytest.approx(0.4)
    assert aggregate.low == aggregate.high == pytest.approx(0.4)


def test_is_reproducible_under_fixed_seed() -> None:
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    first = bootstrap_aggregate(scores, seed=123)
    second = bootstrap_aggregate(scores, seed=123)
    assert first == second


def test_empty_scores_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_aggregate([])


def test_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_aggregate([1.0, 0.0], confidence=1.5)


def test_format_and_margin() -> None:
    aggregate = Aggregate(mean=0.8, low=0.7, high=0.9, confidence=0.95, count=10)
    assert aggregate.margin == pytest.approx(0.1)
    assert aggregate.format() == "80.0% +/- 10.0%"
    assert aggregate.format(pct=False, digits=2) == "0.80 +/- 0.10"

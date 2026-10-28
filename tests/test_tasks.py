"""Tests for built-in tasks, the task registry, and end-to-end measurement.

The integration tests assert the framework's central promise: that it cleanly
separates a real solver from a weak baseline, with the gap reflected in the scores.
"""

from __future__ import annotations

import pytest

from evaluation_harness.core import Runner
from evaluation_harness.models import (
    ArithmeticSolverModel,
    ConstantModel,
    KeywordSentimentModel,
)
from evaluation_harness.tasks import available_tasks, load_task
from evaluation_harness.tasks.arithmetic import build_arithmetic_task
from evaluation_harness.tasks.sentiment import build_sentiment_task


def test_arithmetic_task_is_deterministic() -> None:
    first = build_arithmetic_task(size=20, seed=1)
    second = build_arithmetic_task(size=20, seed=1)
    assert [e.input for e in first] == [e.input for e in second]


def test_arithmetic_targets_are_correct() -> None:
    task = build_arithmetic_task(size=30, seed=3)
    # The arithmetic solver is an exact oracle, so every target must be reproduced.
    result = Runner().run(ArithmeticSolverModel(), task)
    assert result.accuracy().mean == 1.0


def test_sentiment_task_loads_bundled_data() -> None:
    task = build_sentiment_task()
    assert len(task) >= 30
    assert all(e.choices == ("positive", "negative") for e in task)


def test_registry_lists_and_loads_tasks() -> None:
    assert set(available_tasks()) == {"arithmetic", "sentiment"}
    assert load_task("arithmetic").name == "arithmetic"


def test_registry_rejects_unknown_task() -> None:
    with pytest.raises(KeyError, match="Unknown task"):
        load_task("nonexistent")


def test_heuristic_beats_majority_baseline_on_sentiment() -> None:
    task = load_task("sentiment")
    runner = Runner()
    baseline = runner.run(ConstantModel(answer="positive", name="majority"), task)
    heuristic = runner.run(KeywordSentimentModel(), task)
    # The lexicon heuristic should clearly beat always-predict-positive.
    assert heuristic.accuracy().mean > baseline.accuracy().mean
    assert heuristic.accuracy().mean > 0.8


def test_weak_baseline_scores_poorly_on_arithmetic() -> None:
    task = build_arithmetic_task(size=40, seed=5)
    result = Runner().run(ConstantModel(answer="0", name="zero"), task)
    assert result.accuracy().mean < 0.1

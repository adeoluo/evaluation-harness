"""Tests for concurrent execution, retries, and prediction caching."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from evaluation_harness.caching import CachingModel
from evaluation_harness.core import Example, Runner, Task
from evaluation_harness.models import ArithmeticSolverModel

from .helpers import ExactGrader


def _arith_task(n: int = 12) -> Task:
    from evaluation_harness.tasks.arithmetic import build_arithmetic_task

    return build_arithmetic_task(size=n, seed=2)


def test_concurrent_matches_sequential_order() -> None:
    task = _arith_task()
    sequential = Runner(max_workers=1).run(ArithmeticSolverModel(), task)
    concurrent = Runner(max_workers=4).run(ArithmeticSolverModel(), task)
    seq_ids = [r.example.identifier for r in sequential.example_results]
    con_ids = [r.example.identifier for r in concurrent.example_results]
    assert seq_ids == con_ids
    assert concurrent.accuracy().mean == sequential.accuracy().mean == 1.0


def test_concurrent_progress_called_for_each() -> None:
    task = _arith_task(10)
    seen: list[int] = []
    lock = threading.Lock()

    def progress(done: int, total: int) -> None:
        with lock:
            seen.append(done)

    Runner(max_workers=4).run(ArithmeticSolverModel(), task, progress=progress)
    assert len(seen) == 10
    assert max(seen) == 10


def test_runner_rejects_bad_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        Runner(max_workers=0)


def test_runner_rejects_negative_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        Runner(max_retries=-1)


class _FlakyModel:
    """Fails a fixed number of times before succeeding."""

    name = "flaky"

    def __init__(self, failures: int) -> None:
        self._remaining = failures

    def generate(self, example: Example) -> str:
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("transient")
        return example.target


def test_retries_recover_transient_failure() -> None:
    task = Task(
        name="t",
        examples=[Example(input="q", target="ok", id="e1")],
        grader=ExactGrader(),
    )
    model = _FlakyModel(failures=2)
    result = Runner(max_retries=3, retry_base_delay=0.0).run(model, task)
    assert result.accuracy().mean == 1.0


def test_persistent_failure_recorded_not_raised() -> None:
    task = Task(
        name="t",
        examples=[Example(input="q", target="ok", id="e1")],
        grader=ExactGrader(),
    )
    model = _FlakyModel(failures=99)
    result = Runner(max_retries=1, retry_base_delay=0.0).run(model, task)
    assert result.accuracy().mean == 0.0
    assert "model error" in result.example_results[0].grade.rationale  # type: ignore[operator]


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
class _CountingModel:
    name = "counter"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, example: Example) -> str:
        self.calls += 1
        return example.target


def test_cache_avoids_recomputation() -> None:
    inner = _CountingModel()
    cached = CachingModel(inner)
    example = Example(input="q", target="a", id="e1")
    assert cached.generate(example) == "a"
    assert cached.generate(example) == "a"
    assert inner.calls == 1
    assert cached.hits == 1
    assert cached.misses == 1


def test_cache_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    inner = _CountingModel()
    cached = CachingModel(inner, cache_path=path)
    example = Example(input="q", target="a", id="e1")
    cached.generate(example)
    cached.save()

    # A fresh wrapper over a fresh inner model should hit the persisted cache.
    inner2 = _CountingModel()
    reloaded = CachingModel(inner2, cache_path=path)
    assert reloaded.generate(example) == "a"
    assert inner2.calls == 0


def test_cache_save_without_path_raises() -> None:
    with pytest.raises(ValueError, match="no cache_path"):
        CachingModel(_CountingModel()).save()


def test_cache_preserves_model_name() -> None:
    assert CachingModel(_CountingModel()).name == "counter"

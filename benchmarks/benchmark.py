"""A micro-benchmark for evaluation throughput.

Measures how fast the harness evaluates a model over a task and how fast the bootstrap and
significance machinery runs, so the README can quote numbers. Run with:

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

from evaluation_harness.core import Runner
from evaluation_harness.models import build_model
from evaluation_harness.significance import paired_bootstrap_test
from evaluation_harness.statistics import bootstrap_aggregate
from evaluation_harness.tasks import load_task


def _bench_evaluation(repeats: int) -> float:
    task = load_task("sentiment")
    model = build_model("keyword-sentiment")
    runner = Runner()
    examples = task.count if hasattr(task, "count") else len(task)
    start = time.perf_counter()
    for _ in range(repeats):
        runner.run(model, task)
    elapsed = time.perf_counter() - start
    total_examples = repeats * examples
    return total_examples / elapsed if elapsed > 0 else float("inf")


def _bench_bootstrap(repeats: int) -> float:
    scores = [1.0, 0.0] * 50
    start = time.perf_counter()
    for _ in range(repeats):
        bootstrap_aggregate(scores, resamples=10_000)
    elapsed = time.perf_counter() - start
    return repeats / elapsed if elapsed > 0 else float("inf")


def _bench_significance(repeats: int) -> float:
    a = [1.0, 0.0, 1.0, 1.0] * 25
    b = [0.0, 0.0, 1.0, 0.0] * 25
    start = time.perf_counter()
    for _ in range(repeats):
        paired_bootstrap_test(a, b, resamples=10_000)
    elapsed = time.perf_counter() - start
    return repeats / elapsed if elapsed > 0 else float("inf")


def main() -> None:
    examples_per_sec = _bench_evaluation(500)
    bootstraps_per_sec = _bench_bootstrap(200)
    tests_per_sec = _bench_significance(200)
    print("Evaluation Harness micro-benchmark\n")
    print(f"  model evaluation       : {examples_per_sec:>12,.0f} examples/sec")
    print(f"  bootstrap CI (10k)     : {bootstraps_per_sec:>12,.0f} intervals/sec")
    print(f"  paired bootstrap (10k) : {tests_per_sec:>12,.0f} tests/sec")


if __name__ == "__main__":
    main()

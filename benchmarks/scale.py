"""A large-workload scale benchmark for the core evaluation loop.

Where ``benchmark.py`` quotes peak throughput and ``stress.py`` drives the HTTP
service under concurrent load, this script measures the core operation -
:meth:`Runner.run`, which generates, times, and grades every example - on
progressively larger synthetic tasks. It reports wall-clock time, throughput,
peak memory (via the standard-library :mod:`tracemalloc`), and grading accuracy
so the README can show how the harness behaves as the dataset grows.

The tasks are generated deterministically from a fixed seed (nothing is
downloaded), so the run is reproducible and fully offline. Run with:

    uv run python benchmarks/scale.py
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

from evaluation_harness.core import Runner
from evaluation_harness.models import build_model
from evaluation_harness.tasks.arithmetic import build_arithmetic_task

WORKLOAD_SIZES: tuple[int, ...] = (2_000, 8_000, 32_000, 64_000)


@dataclass(frozen=True, slots=True)
class ScaleRow:
    """One measured row of the scale sweep."""

    size: int
    seconds: float
    throughput: float
    peak_mib: float
    accuracy: float


def run_workload(size: int, *, seed: int = 0) -> ScaleRow:
    """Evaluate the arithmetic solver over ``size`` synthetic problems.

    Timing covers the core evaluation loop only; memory is the peak resident
    allocation measured across task generation and evaluation.
    """
    model = build_model("arithmetic-solver")
    runner = Runner()
    tracemalloc.start()
    task = build_arithmetic_task(size=size, seed=seed)
    start = time.perf_counter()
    result = runner.run(model, task)
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    scores = result.scores
    accuracy = sum(scores) / len(scores) if scores else 0.0
    throughput = size / elapsed if elapsed > 0 else float("inf")
    return ScaleRow(size, elapsed, throughput, peak / (1024 * 1024), accuracy)


def main() -> None:
    print("Evaluation Harness scale benchmark")
    print("Data is SYNTHETIC: arithmetic problems generated deterministically from a seed.\n")
    header = (
        f"{'examples':>10}  {'seconds':>8}  {'examples/sec':>13}  {'peak MiB':>9}  {'accuracy':>9}"
    )
    print(header)
    print("-" * len(header))
    for size in WORKLOAD_SIZES:
        row = run_workload(size)
        print(
            f"{row.size:>10,}  {row.seconds:>8.3f}  {row.throughput:>13,.0f}  "
            f"{row.peak_mib:>9.2f}  {row.accuracy:>8.1%}"
        )


if __name__ == "__main__":
    main()

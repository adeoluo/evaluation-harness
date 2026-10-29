"""A production-style stress benchmark for the evaluation service.

Drives the full evaluate handler under sustained, concurrent load, running a
model over a task and aggregating error-barred accuracy, and reports throughput
with p50, p95, and p99 latency. Deterministic and offline; run with:

    uv run python benchmarks/stress.py
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from evaluation_harness.service import EvaluationService, handle


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    service = EvaluationService.build()
    body = {"model": "arithmetic-solver", "tasks": ["arithmetic"]}
    requests = [dict(body) for _ in range(500)]

    def one(request: dict[str, object]) -> float:
        start = time.perf_counter()
        handle("POST", "/evaluate", request, service)
        return (time.perf_counter() - start) * 1000

    print(f"Requests: {len(requests)} (each evaluates a task with bootstrap aggregation)")
    for workers in (1, 4, 8):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            latencies = list(pool.map(one, requests))
        wall = time.perf_counter() - start
        print(
            f"workers={workers:<2}  "
            f"throughput={len(requests) / wall:>7.0f} req/s  "
            f"p50={statistics.median(latencies):>7.2f} ms  "
            f"p95={_percentile(latencies, 0.95):>7.2f} ms  "
            f"p99={_percentile(latencies, 0.99):>7.2f} ms"
        )


if __name__ == "__main__":
    main()

# Operations Guide - Evaluation Harness

The operational contract of the evaluation service.

## Running the service

```bash
uv run python -m evaluation_harness serve --host 0.0.0.0 --port 8000
```

A standard-library HTTP server. It evaluates built-in models on built-in tasks and compares
two models with a significance test, returning error-barred metrics.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/health` | none | status, task names, model names |
| POST | `/evaluate` | `{"model", "tasks", "confidence"?, "seed"?}` | per-task accuracy with a confidence interval |
| POST | `/compare` | `{"model_a", "model_b", "task", "test"?, "seed"?}` | both accuracies, p-value, effect size, verdict |

Errors return JSON `{"error": ...}` with a 4xx status. Model and task names must be known;
`tasks` is bounded to 20 per request.

## Service level objectives

Measured by `benchmarks/stress.py` on a developer laptop (each request aggregates a bootstrap
confidence interval over a task):

| Objective | Target | Observed |
|---|---|---|
| Evaluate latency p50 (1 worker) | < 15 ms | about 7 ms |
| Evaluate latency p99 (1 worker) | < 15 ms | about 8 ms |
| Throughput (1 worker) | > 100 evaluations/second | about 140/second |
| Throughput (4 workers) | scales with cores | about 420/second |

## Operational constraints

- **The dominant cost is the bootstrap confidence interval**, not model inference. Each
  evaluate request runs thousands of resamples; this is why latency is milliseconds, not
  microseconds, and it is the knob to tune (resample count) under load.
- **Throughput scales with threads here, unusually.** The bootstrap is a numeric workload
  that releases the global interpreter lock, so unlike the other services in this portfolio,
  adding workers increases throughput (measured: about 140/s at one worker to about 420/s at
  four). Size threads to cores for aggregation-heavy workloads.
- **Evaluation is deterministic for a fixed seed**, so results are reproducible and cacheable
  by `(model, task, seed, confidence)`.
- **Requests are bounded**: known models and tasks only, at most 20 tasks per request.
- **No persistence, no authentication.** Front the service with a gateway. With real
  model-backed evaluations, inference would dominate and concurrency would be bound by
  provider rate limits.

## Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| High latency | Large resample count or many tasks | Lower resamples; split tasks across requests |
| `400` on evaluate | Unknown model/task or bad confidence | Inspect the `error` field |
| Wide confidence intervals | Small task sample size | Expected; report the interval, do not hide it |

## Capacity planning

For aggregation-heavy evaluation, size worker threads to cores (the bootstrap parallelizes).
For real model-backed evaluation, provision for inference latency and provider limits.

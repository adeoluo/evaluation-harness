# Scaling and Design Tradeoffs - Evaluation Harness

How the reference design grows, and the tradeoffs taken.

## Reference architecture (today)

A single process evaluates models on tasks, aggregates error-barred metrics with a
bootstrap, and compares models with significance tests, served over a standard-library HTTP
service. Deterministic for a fixed seed.

## Scaling path

| Stage | Bottleneck | Change |
|---|---|---|
| One process | Bootstrap CPU (numeric) | Add worker threads; the bootstrap releases the lock |
| One host | Host cores | Horizontal replicas plus intra-process threads |
| Many tasks | Sequential task runs | Run tasks concurrently (the Runner supports it) |
| Real models | Inference latency and rate limits | Batch and cache model calls; concurrency per provider |
| Repeated evaluations | Re-running the same config | Cache by (model, task, seed, confidence) |

Unusually for this portfolio, aggregation parallelizes within a process because the bootstrap
is numeric; model-backed evaluation would shift the bottleneck to inference.

## Substantial tradeoffs

- **Bootstrap intervals vs point estimates.** Reporting a confidence interval is honest but
  costs thousands of resamples per metric; a bare point estimate is cheap but misleading.
  The harness pays the cost and exposes the resample count as a knob.
- **Significance tests vs raw differences.** A paired bootstrap or McNemar test defends the
  word "better" but adds computation and requires paired scores; a raw difference is free but
  unfalsifiable. The reference reports the test, the p-value, and an effect size.
- **Threads vs processes.** Here threads help, because the numeric bootstrap releases the
  global interpreter lock, the opposite of the pure-Python services in this portfolio. The
  right unit of scaling depends on the workload, which the stress benchmark makes visible.
- **Deterministic seeds vs fresh randomness.** Seeding makes results reproducible and
  cacheable at the cost of exploring one resampling draw; the seed is a request parameter.

## What would change under real load

Run tasks concurrently, size threads to cores for aggregation, cache by evaluation key, and,
for model-backed evaluation, place inference behind a batched, rate-limit-aware client where
it would become the dominant cost. The statistics are unchanged.

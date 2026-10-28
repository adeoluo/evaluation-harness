# ADR-0002: Bootstrap confidence intervals for reported metrics

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

Every reported metric needs an honest error bar. The usual choice is the normal
approximation of the standard error of the mean. Evaluation scores, however, are bounded
in `[0, 1]` and are frequently bimodal (a pass/fail grader yields only 0.0 and 1.0).
On the small benchmarks common in model evaluation, the normal approximation produces
intervals that are too narrow and can extend past 0 or 1.

## Decision

Report the mean with a **percentile bootstrap confidence interval**: resample the
per-example scores with replacement many times, and take percentiles of the resample
means. Resampling uses a seeded generator so intervals are exactly reproducible.

## Alternatives considered

- **Normal approximation (mean +/- 1.96 * SE).** Simple, but unreliable for bounded,
  bimodal scores on small samples, and can yield impossible bounds.
- **Wilson interval.** Good for pure pass/fail proportions, but does not generalize to
  continuous or partial-credit scores, which the `Grade` type supports.

## Consequences

- Intervals are valid for any grader that returns a score in `[0, 1]`, including
  partial-credit and rubric graders.
- A seed is required for reproducibility and is recorded on every result.
- Bootstrapping costs a few milliseconds per task, which is negligible next to generation.

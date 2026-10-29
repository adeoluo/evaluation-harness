# End-to-End Case Study - Is Model A Really Better Than Model B?

How the harness answers the question that matters, with a confidence interval and a
significance test rather than a single number.

## Scenario

A team has two models and a task. Model A scores higher on the task. The question is whether
that difference is real or noise.

## Evaluating each model

```bash
curl -s -X POST localhost:8000/evaluate -d '{"model": "arithmetic-solver", "tasks": ["arithmetic"]}'
```

```json
{"model": "arithmetic-solver",
 "results": [{"task": "arithmetic", "count": 60, "accuracy": 1.0, "ci_low": 1.0, "ci_high": 1.0}]}
```

Every accuracy comes with a bootstrap confidence interval, so a point estimate is never
reported on its own.

## Comparing with a significance test

```bash
curl -s -X POST localhost:8000/compare -d '{
  "model_a": "arithmetic-solver", "model_b": "constant-zero", "task": "arithmetic", "test": "bootstrap"
}'
```

```json
{"mean_difference": 1.0, "p_value": 0.0, "effect_size": 12.4, "significant": true, "test": "paired bootstrap"}
```

The comparison reports the mean difference, a p-value, an effect size (Cohen's d), and a
verdict. A difference is only called significant when the test says so, so "better" is a
claim the harness can actually defend.

## What an operator sees

- **Error bars, always.** No accuracy is reported without a confidence interval.
- **A real test, not eyeballing.** Paired bootstrap or McNemar turns "looks better" into a
  p-value and an effect size.
- **Reproducibility.** A fixed seed makes every number regenerate exactly, so a reported
  result can be independently checked.

## Takeaway

Treating evaluation as statistics, point estimates with intervals and comparisons with
significance tests, is what separates an honest leaderboard from a misleading one.

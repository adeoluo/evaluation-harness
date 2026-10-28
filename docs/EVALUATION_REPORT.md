# Evaluation Report - Evaluation Harness

> Reproduce everything here with `make evaluation` (or the commands below). Scores are
> the mean with a 95% percentile bootstrap confidence interval over the task's examples.

## Setup

- **Tasks:** `arithmetic` (60 seed-generated numeric problems) and `sentiment`
  (48 bundled, labeled review sentences including negation and mixed-polarity cases).
- **Metric:** accuracy. Arithmetic uses the numeric grader (value comparison within
  tolerance); sentiment uses the choice grader (positive / negative).
- **Statistics:** percentile bootstrap, 10,000 resamples, seed 0.
- **Models:** two solvers and two baselines, all deterministic and offline.
- **Hardware / cost:** runs on a laptop CPU in under a second; no API calls, zero cost.

## Headline results

| Task | Model | Accuracy (95% CI) | N | vs. baseline |
|---|---|---:|---:|---:|
| arithmetic | `arithmetic-solver` | **100.0% +/- 0.0%** | 60 | +100.0 |
| arithmetic | `constant-zero` (baseline) | 0.0% +/- 0.0% | 60 | - |
| sentiment | `keyword-sentiment` | **91.7% +/- 7.3%** | 48 | +41.7 |
| sentiment | majority-class (baseline) | 50.0% +/- 14.6% | 48 | - |

Reproduce:

```bash
uv run python -m evaluation_harness run --model arithmetic-solver --task arithmetic
uv run python -m evaluation_harness run --model keyword-sentiment --task sentiment
```

## Interpretation

- The arithmetic solver is an exact oracle (it evaluates the expression), so a perfect
  score with a zero-width interval is the correct, expected outcome and acts as a
  correctness check on the numeric grader and the task targets.
- The lexicon sentiment model clearly beats the majority-class baseline but is not
  perfect: its errors are concentrated on the negation examples ("not good at all",
  "hardly a masterpiece"), which a bag-of-words lexicon cannot handle. The non-zero
  confidence interval reflects these genuine errors.

## Error analysis (sentiment, `keyword-sentiment`)

All four errors are negation or mixed-polarity sentences where a positive lexicon word
dominates the surface form despite a negative overall meaning. This is the expected,
well-documented failure mode of lexicon classifiers and is why those examples were added
to the benchmark.

## Model-graded validation

The `model_graded` grader is exercised with a deterministic scripted judge in the test
suite. Before using a live judge model in anger, validate it against human labels and
report agreement (for example Cohen's kappa); that validation is out of scope for the
bundled offline tasks.

## Reproducibility

- Seeds: aggregation seed 0; arithmetic generation seed 0.
- Everything is offline and deterministic; the numbers above regenerate exactly.
- Full per-example records are available via `--json report.json`.

# System Card - Evaluation Harness

## Overview

- **What it is:** a framework for measuring the quality of language models and agents.
  It is a measurement tool, not a model. It executes models over tasks, grades the
  outputs, and reports accuracy with bootstrap confidence intervals.
- **Components under evaluation:** any object satisfying the `Model` protocol - the
  bundled deterministic stubs and heuristics, or the optional OpenAI / Anthropic adapters.
- **Version:** 0.1.0.

## Intended use

- **Primary use cases:** comparing models or prompting strategies on a task with honest
  error bars; regression-testing a system across versions; checking a benchmark for
  training-data contamination.
- **Primary users:** engineers and researchers evaluating language models.

## Out-of-scope / prohibited use

- Not a leaderboard or a source of competitive benchmark scores; the bundled tasks are
  demonstrations.
- Model-graded scores must not be treated as ground truth without validating the judge
  against human labels.
- Do not use the contamination scan as a guarantee of a clean benchmark; it detects
  n-gram overlap only.

## Performance

See [EVALUATION_REPORT.md](EVALUATION_REPORT.md). On the bundled tasks the framework
correctly separates solvers from baselines and reports calibrated intervals.

## Risks and limitations

- **Judge reliability:** model-graded evaluation inherits the judge's biases and errors.
- **Contamination blind spots:** paraphrased leakage shares no n-grams and is not caught.
- **Small-sample intervals:** confidence intervals widen on small tasks, as they should;
  they are not a substitute for an adequately sized evaluation set.

## Safety mitigations

- Deterministic, offline defaults make results reproducible and auditable.
- Per-example records can be exported (`--json`) for inspection and error analysis.
- Provider adapters are opt-in and isolated; nothing contacts the network by default.

## Environmental / cost footprint

The bundled evaluations run on CPU in under a second with no API calls. Cost and energy
when evaluating frontier models are entirely determined by the provider calls the user
chooses to make.

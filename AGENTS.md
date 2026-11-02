# AGENTS.md

Working notes for coding agents and contributors in **Evaluation Harness**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

A from-scratch evaluation and benchmarking framework for large language models and agents.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/evaluation_harness/__init__.py`: Evaluation Harness - a framework for evaluating language models.
- `src/evaluation_harness/caching.py`: A caching wrapper for models.
- `src/evaluation_harness/cli.py`: Command-line interface for running evaluations.
- `src/evaluation_harness/contamination.py`: N-gram contamination detection.
- `src/evaluation_harness/core/__init__.py`: Core abstractions for the evaluation harness.
- `src/evaluation_harness/core/example.py`: The :class:`Example`: a single, immutable evaluation instance.
- `src/evaluation_harness/core/grader.py`: The :class:`Grader`: scores a model prediction against an example.
- `src/evaluation_harness/core/model.py`: The :class:`Model`: any system that produces a prediction for an example.
- `src/evaluation_harness/core/result.py`: Result containers produced by a :class:`~evaluation_harness.core.runner.Runner`.
- `src/evaluation_harness/core/runner.py`: The :class:`Runner`: executes a model over a task and collects results.
- `src/evaluation_harness/core/task.py`: The :class:`Task`: a named benchmark pairing examples with a grader.
- `src/evaluation_harness/graders/__init__.py`: Built-in graders.
- `src/evaluation_harness/graders/choice.py`: Grader for multiple-choice tasks.
- `src/evaluation_harness/graders/collection.py`: Graders for set-valued and list-valued answers.
- `src/evaluation_harness/graders/composite.py`: Graders that combine other graders.
- `src/evaluation_harness/graders/exact_match.py`: Exact-match grader with configurable normalization.
- `src/evaluation_harness/graders/f1.py`: Token-level F1 grader for extractive and short-answer tasks.
- `src/evaluation_harness/graders/fuzzy.py`: Fuzzy string grader based on edit distance.
- `src/evaluation_harness/graders/includes.py`: Grader that checks whether the target appears within the prediction.
- `src/evaluation_harness/graders/json_grader.py`: Graders for structured (JSON) model output.
- `src/evaluation_harness/graders/model_graded.py`: Model-graded (rubric) grading.
- `src/evaluation_harness/graders/normalization.py`: Text normalization shared across string-based graders.
- `src/evaluation_harness/graders/numeric.py`: Grader for numeric answers with absolute or relative tolerance.
- `src/evaluation_harness/graders/regex_match.py`: Grader that matches the prediction against a regular expression.
- `src/evaluation_harness/graders/rouge.py`: ROUGE-L grader for summary and long-answer overlap.
- `src/evaluation_harness/models/__init__.py`: Built-in models and a small registry for constructing them by name.
- `src/evaluation_harness/models/entailment_heuristic.py`: A transparent heuristic for the entailment task.
- `src/evaluation_harness/models/heuristics.py`: Transparent heuristic baselines for the bundled tasks.
- `src/evaluation_harness/models/providers.py`: Optional adapters for frontier model providers.
- `src/evaluation_harness/models/stub.py`: Deterministic, offline models.
- `src/evaluation_harness/observability.py`: In-process request metrics for the HTTP service.
- `src/evaluation_harness/persistence.py`: Persisting and reloading evaluation runs.
- `src/evaluation_harness/regression.py`: Regression detection between two evaluation runs.
- `src/evaluation_harness/reporting.py`: Rendering evaluation results as text tables and JSON.
- `src/evaluation_harness/reporting_ext.py`: Markdown and HTML reports, plus error analysis.
- `src/evaluation_harness/service.py`: A standard-library HTTP service exposing the evaluation harness.
- `src/evaluation_harness/significance.py`: Statistical significance testing for model comparisons.
- `src/evaluation_harness/statistics.py`: Statistical aggregation for evaluation scores.
- `src/evaluation_harness/tasks/__init__.py`: Built-in tasks and a small registry for loading them by name.
- `src/evaluation_harness/tasks/arithmetic.py`: The bundled ``arithmetic`` task: numeric-answer word problems.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the application entrypoint |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut by the maintainer with `make cut-release`, never from CI.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

"""Command-line interface for running evaluations.

Subcommands:

- ``list`` - show the built-in tasks and models.
- ``run`` - evaluate a model on one or more tasks and print a results table, with an
  optional machine-readable JSON export.
- ``compare`` - evaluate two models on a task and report a significance test.

The CLI wires together the registries, the runner, and the reporters; it contains no
evaluation logic of its own.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from evaluation_harness import __version__
from evaluation_harness.core.result import TaskResult
from evaluation_harness.core.runner import Runner
from evaluation_harness.models import available_models, build_model
from evaluation_harness.reporting import format_results_table, results_to_json
from evaluation_harness.significance import mcnemar_test, paired_bootstrap_test
from evaluation_harness.tasks import available_tasks, load_task


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="evaluation-harness",
        description="A principled evaluation harness for language models and agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List built-in tasks and models.")
    list_parser.set_defaults(handler=_handle_list)

    run_parser = subparsers.add_parser("run", help="Evaluate a model on one or more tasks.")
    run_parser.add_argument("--model", required=True, help="Built-in model name.")
    run_parser.add_argument(
        "--task",
        required=True,
        action="append",
        dest="tasks",
        metavar="TASK",
        help="Task name (repeat to evaluate several).",
    )
    run_parser.add_argument("--seed", type=int, default=0, help="Aggregation seed.")
    run_parser.add_argument(
        "--confidence", type=float, default=0.95, help="Confidence level for intervals."
    )
    run_parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write a full JSON report to PATH.",
    )
    run_parser.set_defaults(handler=_handle_run)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two models on a task with a significance test."
    )
    compare_parser.add_argument("--model-a", required=True, help="First model name.")
    compare_parser.add_argument("--model-b", required=True, help="Second model name.")
    compare_parser.add_argument("--task", required=True, help="Task name.")
    compare_parser.add_argument(
        "--test",
        choices=("bootstrap", "mcnemar"),
        default="bootstrap",
        help="Significance test to apply.",
    )
    compare_parser.add_argument("--seed", type=int, default=0, help="Aggregation seed.")
    compare_parser.set_defaults(handler=_handle_compare)

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP evaluation service.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    serve_parser.set_defaults(handler=_handle_serve)
    return parser


def _handle_serve(args: argparse.Namespace) -> int:
    from evaluation_harness.service import serve

    serve(host=args.host, port=args.port)
    return 0


def _handle_list(_: argparse.Namespace) -> int:
    print("Tasks:")
    for name in available_tasks():
        print(f"  - {name}")
    print("Models:")
    for name in available_models():
        print(f"  - {name}")
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    try:
        model = build_model(args.model)
    except KeyError as error:
        print(str(error), file=sys.stderr)
        return 2

    runner = Runner(seed=args.seed)
    results: list[TaskResult] = []
    for task_name in args.tasks:
        try:
            task = load_task(task_name)
        except KeyError as error:
            print(str(error), file=sys.stderr)
            return 2
        results.append(runner.run(model, task))

    print(format_results_table(results, confidence=args.confidence))

    if args.json is not None:
        args.json.write_text(results_to_json(results, confidence=args.confidence), encoding="utf-8")
        print(f"\nWrote JSON report to {args.json}")
    return 0


def _handle_compare(args: argparse.Namespace) -> int:
    try:
        model_a = build_model(args.model_a)
        model_b = build_model(args.model_b)
        task = load_task(args.task)
    except KeyError as error:
        print(str(error), file=sys.stderr)
        return 2

    runner = Runner(seed=args.seed)
    result_a = runner.run(model_a, task)
    result_b = runner.run(model_b, task)
    test = mcnemar_test if args.test == "mcnemar" else paired_bootstrap_test
    comparison = test(result_a.scores, result_b.scores)

    print(f"Task: {task.name} (N={result_a.count})")
    print(f"  {model_a.name}: {result_a.accuracy().format()}")
    print(f"  {model_b.name}: {result_b.accuracy().format()}")
    print(f"  Mean difference (A - B): {comparison.mean_difference:+.4f}")
    print(f"  Test: {comparison.test}")
    print(f"  p-value: {comparison.p_value:.4f}")
    print(f"  Effect size (Cohen's d): {comparison.effect_size:+.3f}")
    verdict = "significant" if comparison.significant else "not significant"
    print(f"  Verdict: difference is {verdict} at alpha=0.05")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    exit_code: int = handler(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

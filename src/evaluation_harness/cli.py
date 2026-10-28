"""Command-line interface for running evaluations.

Two subcommands:

- ``list`` - show the built-in tasks and models.
- ``run`` - evaluate a model on one or more tasks and print a results table, with an
  optional machine-readable JSON export.

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
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    exit_code: int = handler(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

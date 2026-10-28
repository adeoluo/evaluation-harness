"""Tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation_harness.cli import main


def test_list_shows_tasks_and_models(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "arithmetic" in output
    assert "keyword-sentiment" in output


def test_run_prints_results_table(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--model", "keyword-sentiment", "--task", "sentiment"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "sentiment" in output
    assert "Accuracy" in output


def test_run_multiple_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "run",
            "--model",
            "arithmetic-solver",
            "--task",
            "arithmetic",
            "--task",
            "sentiment",
        ]
    )
    assert exit_code == 0
    assert capsys.readouterr().out.count("arithmetic") >= 1


def test_run_writes_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "report.json"
    exit_code = main(
        ["run", "--model", "arithmetic-solver", "--task", "arithmetic", "--json", str(out)]
    )
    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["results"][0]["task"] == "arithmetic"
    assert payload["results"][0]["accuracy"]["mean"] == 1.0


def test_run_unknown_model_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--model", "nope", "--task", "arithmetic"])
    assert exit_code == 2
    assert "Unknown model" in capsys.readouterr().err


def test_run_unknown_task_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--model", "arithmetic-solver", "--task", "nope"])
    assert exit_code == 2
    assert "Unknown task" in capsys.readouterr().err


def test_missing_command_errors() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_compare_command_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "compare",
            "--model-a",
            "keyword-sentiment",
            "--model-b",
            "constant-positive",
            "--task",
            "sentiment",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "p-value" in output
    assert "Effect size" in output


def test_compare_mcnemar(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "compare",
            "--model-a",
            "keyword-sentiment",
            "--model-b",
            "constant-positive",
            "--task",
            "sentiment",
            "--test",
            "mcnemar",
        ]
    )
    assert exit_code == 0
    assert "mcnemar" in capsys.readouterr().out


def test_compare_unknown_model_exits(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["compare", "--model-a", "nope", "--model-b", "echo", "--task", "sentiment"])
    assert exit_code == 2

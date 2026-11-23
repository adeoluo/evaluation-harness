"""The results table in the README is what the bundled models score today."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation_harness.cli import main

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()

ROWS = [
    ("arithmetic-solver", "arithmetic", "100.0% +/- 0.0%", 60),
    ("constant-zero", "arithmetic", "0.0% +/- 0.0%", 60),
    ("keyword-sentiment", "sentiment", "91.7% +/- 7.3%", 48),
    ("constant-positive", "sentiment", "50.0% +/- 14.6%", 48),
    ("overlap-entailment", "entailment", "50.0% +/- 16.7%", 30),
    ("oracle", "factual_qa", "100.0% +/- 0.0%", 25),
]


@pytest.mark.parametrize(("model", "task", "score", "count"), ROWS)
def test_readme_row_matches_a_fresh_run(
    model: str, task: str, score: str, count: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--model", model, "--task", task]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert score in line and f" {count} " in line, line
    row = next(r for r in README.splitlines() if r.startswith(f"| {task} | `{model}`"))
    assert score in row and row.rstrip(" |").endswith(str(count)), row


def test_readme_quotes_the_compare_effect_size(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--model-a", "keyword-sentiment", "--model-b", "constant-positive"]
    assert main(["compare", *args, "--task", "sentiment"]) == 0
    out = capsys.readouterr().out
    effect = float(out.split("Cohen's d): ")[1].split()[0])
    assert f"d = **{effect:.2f}**" in README

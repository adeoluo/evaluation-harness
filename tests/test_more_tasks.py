"""Tests for the entailment and factual-QA tasks and their heuristics."""

from __future__ import annotations

from evaluation_harness.core import Runner
from evaluation_harness.models import ConstantModel, OverlapEntailmentModel, build_model
from evaluation_harness.tasks import available_tasks, load_task
from evaluation_harness.tasks.entailment import build_entailment_task
from evaluation_harness.tasks.factual_qa import build_factual_qa_task


def test_registry_includes_new_tasks() -> None:
    assert {"arithmetic", "sentiment", "entailment", "factual_qa"} <= set(available_tasks())


def test_entailment_task_loads() -> None:
    task = build_entailment_task()
    assert len(task) >= 25
    assert all(e.choices == ("entailment", "contradiction", "neutral") for e in task)


def test_factual_qa_task_loads() -> None:
    task = build_factual_qa_task()
    assert len(task) >= 20


def test_entailment_heuristic_beats_majority() -> None:
    task = load_task("entailment")
    runner = Runner()
    heuristic = runner.run(OverlapEntailmentModel(), task)
    baseline = runner.run(ConstantModel(answer="entailment", name="majority"), task)
    assert heuristic.accuracy().mean > baseline.accuracy().mean


def test_factual_qa_f1_gives_partial_credit() -> None:
    task = load_task("factual_qa")
    # An oracle that returns the exact target scores perfectly under F1.
    from evaluation_harness.models import EchoModel

    class OracleModel:
        name = "oracle"

        def generate(self, example: object) -> str:
            return example.target  # type: ignore[attr-defined]

    result = Runner().run(OracleModel(), task)
    assert result.accuracy().mean == 1.0
    # Echo returns the question, which should rarely match the answer.
    echo_result = Runner().run(EchoModel(), task)
    assert echo_result.accuracy().mean < 0.3


def test_build_model_constructs_entailment_heuristic() -> None:
    assert build_model("overlap-entailment").name == "overlap-entailment"

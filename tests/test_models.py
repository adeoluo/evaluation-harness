"""Tests for built-in models and the model registry."""

from __future__ import annotations

import pytest

from evaluation_harness.core import Example
from evaluation_harness.models import (
    ArithmeticSolverModel,
    ConstantModel,
    EchoModel,
    KeywordSentimentModel,
    OpenAIModel,
    available_models,
    build_model,
    providers,
)


def test_constant_model_is_constant() -> None:
    model = ConstantModel(answer="7")
    assert model.generate(Example(input="anything", target="x")) == "7"


def test_echo_model_returns_input() -> None:
    assert EchoModel().generate(Example(input="hello", target="x")) == "hello"


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("What is 2 + 2?", "4"),
        ("What is 10 - 3?", "7"),
        ("What is 6 * 7?", "42"),
    ],
)
def test_arithmetic_solver(question: str, answer: str) -> None:
    assert ArithmeticSolverModel().generate(Example(input=question, target=answer)) == answer


def test_arithmetic_solver_handles_unparseable_input() -> None:
    assert ArithmeticSolverModel().generate(Example(input="no math here", target="0")) == ""


def test_keyword_sentiment_classifies() -> None:
    model = KeywordSentimentModel()
    assert model.generate(Example(input="a wonderful, brilliant film", target="positive")) == (
        "positive"
    )
    assert model.generate(Example(input="an awful, boring mess", target="negative")) == "negative"


def test_keyword_sentiment_falls_back_to_default_on_tie() -> None:
    model = KeywordSentimentModel(default="negative")
    assert model.generate(Example(input="the movie released today", target="x")) == "negative"


def test_registry_builds_known_models() -> None:
    assert "arithmetic-solver" in available_models()
    assert build_model("keyword-sentiment").name == "keyword-sentiment"


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("does-not-exist")


def test_provider_adapter_name() -> None:
    assert OpenAIModel(model="gpt-4o").name == "gpt-4o"


def test_require_raises_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="uv sync --extra openai"):
        providers._require("openai", "openai")

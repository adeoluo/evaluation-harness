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
    assert OpenAIModel(model="gpt-6.1-sol").name == "gpt-6.1-sol"


def test_require_raises_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="uv sync --extra openai"):
        providers._require("openai", "openai")


class _Recorder:
    """Stands in for an SDK client and records the keyword arguments of each call."""

    def __init__(self, reply: object) -> None:
        self.calls: list[dict[str, object]] = []
        self.reply = reply

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.reply


def _fake_sdks(monkeypatch: pytest.MonkeyPatch) -> tuple[_Recorder, _Recorder]:
    import sys
    import types

    openai_reply = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
    )
    anthropic_reply = types.SimpleNamespace(
        content=[
            types.SimpleNamespace(type="thinking"),
            types.SimpleNamespace(type="text", text="ok"),
        ]
    )
    chat, messages = _Recorder(openai_reply), _Recorder(anthropic_reply)
    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = lambda: types.SimpleNamespace(chat=types.SimpleNamespace(completions=chat))  # type: ignore[attr-defined]
    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.Anthropic = lambda: types.SimpleNamespace(messages=messages)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_mod)
    monkeypatch.setattr(providers, "_require", lambda package, extra: None)
    return chat, messages


def test_openai_request_turns_reasoning_off_for_repeatable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat, _ = _fake_sdks(monkeypatch)
    model = providers.OpenAIModel()
    assert model.generate(Example(input="p", target="t")) == "ok"
    sent = chat.calls[0]
    assert sent["model"] == "gpt-6-luna"
    assert sent["reasoning_effort"] == "none"
    assert sent["temperature"] == 0.0


def test_anthropic_request_sends_no_sampling_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    _, messages = _fake_sdks(monkeypatch)
    model = providers.AnthropicModel()
    assert model.generate(Example(input="p", target="t")) == "ok"
    sent = messages.calls[0]
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["max_tokens"] == 2048
    assert not {"temperature", "top_p", "top_k"} & sent.keys()

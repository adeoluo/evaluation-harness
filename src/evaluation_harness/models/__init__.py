"""Built-in models and a small registry for constructing them by name.

Deterministic models (``constant``, ``echo``) and transparent heuristics
(``arithmetic-solver``, ``keyword-sentiment``) run fully offline. Provider adapters
(``openai``, ``anthropic``) are available when their extra is installed.
"""

from __future__ import annotations

from collections.abc import Callable

from evaluation_harness.core.model import Model
from evaluation_harness.models.heuristics import (
    ArithmeticSolverModel,
    KeywordSentimentModel,
)
from evaluation_harness.models.providers import AnthropicModel, OpenAIModel
from evaluation_harness.models.stub import ConstantModel, EchoModel

_BUILDERS: dict[str, Callable[[], Model]] = {
    "echo": EchoModel,
    "arithmetic-solver": ArithmeticSolverModel,
    "keyword-sentiment": KeywordSentimentModel,
    "constant-positive": lambda: ConstantModel(answer="positive", name="constant-positive"),
    "constant-zero": lambda: ConstantModel(answer="0", name="constant-zero"),
}


def available_models() -> list[str]:
    """Return the names of models constructible via :func:`build_model`."""
    return sorted(_BUILDERS)


def build_model(name: str) -> Model:
    """Construct a built-in model by name.

    Raises:
        KeyError: If ``name`` is not a registered built-in model.
    """
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(available_models())}"
        ) from None
    return builder()


__all__ = [
    "AnthropicModel",
    "ArithmeticSolverModel",
    "ConstantModel",
    "EchoModel",
    "KeywordSentimentModel",
    "OpenAIModel",
    "available_models",
    "build_model",
]

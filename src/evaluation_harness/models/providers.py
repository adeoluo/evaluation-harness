"""Optional adapters for frontier model providers.

These are the only components that touch the network. They lazily import their SDK so
the framework has no hard dependency on any provider: installing an extra enables an
adapter (``uv sync --extra openai``), and nothing else in the framework changes. A
missing SDK raises a clear, actionable error rather than an opaque ``ImportError``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal

from evaluation_harness.core.example import Example


def _require(package: str, extra: str) -> None:
    if importlib.util.find_spec(package) is None:
        raise ImportError(
            f"The {package!r} package is required for this model. "
            f"Install it with: uv sync --extra {extra}"
        )


@dataclass(frozen=True)
class OpenAIModel:
    """Chat-completion adapter for OpenAI models.

    Args:
        model: The model identifier (for example ``gpt-6-luna``).
        system_prompt: Optional system message prepended to every request.
        temperature: Sampling temperature; ``0.0`` keeps evaluations near-deterministic.
    """

    model: str = "gpt-6-luna"
    system_prompt: str | None = None
    temperature: float = 0.0
    # Current OpenAI models reason by default and then reject temperature; "none" turns
    # reasoning off so temperature 0 applies.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] = "none"

    @property
    def name(self) -> str:
        return self.model

    def generate(self, example: Example) -> str:
        _require("openai", "openai")
        from openai import OpenAI

        client = OpenAI()
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": example.input})
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )
        return response.choices[0].message.content or ""


@dataclass(frozen=True)
class AnthropicModel:
    """Messages-API adapter for Anthropic Claude models."""

    model: str = "claude-sonnet-5-5"
    system_prompt: str | None = None
    # Includes adaptive thinking tokens on Claude 5 models. No temperature: the current
    # Messages API and the Claude 5 models accept only the default sampling settings.
    max_tokens: int = 2048

    @property
    def name(self) -> str:
        return self.model

    def generate(self, example: Example) -> str:
        _require("anthropic", "anthropic")
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt or "",
            messages=[{"role": "user", "content": example.input}],
        )
        parts = [
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts)

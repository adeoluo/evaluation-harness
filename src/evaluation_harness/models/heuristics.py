"""Transparent heuristic baselines for the bundled tasks.

A measurement framework is only trustworthy if it separates a real solver from a weak
baseline. These two heuristics provide exactly that contrast for the bundled tasks:
an arithmetic solver that should score near-perfectly, and a lexicon sentiment
classifier that should clearly beat a majority-class baseline without reaching 100%.
They are intentionally simple and inspectable - the interesting artifact is the
framework's measurement, not the models.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field

from evaluation_harness.core.example import Example

_SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_EXPRESSION = re.compile(r"[-+*/\d\s().]+")

_POSITIVE_WORDS = frozenset(
    {
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "love",
        "loved",
        "best",
        "brilliant",
        "fantastic",
        "enjoyable",
        "delightful",
        "superb",
        "beautiful",
        "funny",
        "charming",
        "masterpiece",
        "gripping",
        "heartwarming",
    }
)
_NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "terrible",
        "awful",
        "boring",
        "worst",
        "hate",
        "hated",
        "poor",
        "disappointing",
        "dull",
        "weak",
        "forgettable",
        "mess",
        "tedious",
        "bland",
        "lifeless",
        "painful",
        "clumsy",
        "predictable",
        "mediocre",
    }
)
_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class ArithmeticSolverModel:
    """Evaluates the arithmetic expression embedded in the question.

    Expressions are parsed into an abstract syntax tree and evaluated with a fixed set
    of safe binary operators, never with :func:`eval`.
    """

    name: str = "arithmetic-solver"

    def generate(self, example: Example) -> str:
        match = _EXPRESSION.search(example.input.replace("What is", "").replace("?", ""))
        if match is None:
            return ""
        try:
            value = _evaluate(match.group(0).strip())
        except (ValueError, SyntaxError, ZeroDivisionError):
            return ""
        if value == int(value):
            return str(int(value))
        return str(value)


@dataclass(frozen=True)
class KeywordSentimentModel:
    """Classifies sentiment by counting words from positive/negative lexicons.

    Ties (including texts with no lexicon words) fall back to ``default``, mimicking a
    majority-class prior.
    """

    default: str = "positive"
    name: str = "keyword-sentiment"
    _positive: frozenset[str] = field(default=_POSITIVE_WORDS, repr=False)
    _negative: frozenset[str] = field(default=_NEGATIVE_WORDS, repr=False)

    def generate(self, example: Example) -> str:
        words = _WORD.findall(example.input.lower())
        positive = sum(word in self._positive for word in words)
        negative = sum(word in self._negative for word in words)
        if positive > negative:
            return "positive"
        if negative > positive:
            return "negative"
        return self.default


def _evaluate(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _evaluate_node(tree.body)


def _evaluate_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate_node(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINARY_OPERATORS:
        function = _SAFE_BINARY_OPERATORS[type(node.op)]
        return float(function(_evaluate_node(node.left), _evaluate_node(node.right)))
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

"""The bundled ``arithmetic`` task: numeric-answer word problems.

Problems are generated deterministically from a seed, so the task is reproducible and
free of any contamination concern (nothing is downloaded, and the generator is public).
The task exercises the numeric grader, which must score answers by value rather than by
string form.
"""

from __future__ import annotations

import random

from evaluation_harness.core.example import Example
from evaluation_harness.core.task import Task
from evaluation_harness.graders.numeric import NumericGrader

_OPERATORS: tuple[tuple[str, str], ...] = (
    ("+", "plus"),
    ("-", "minus"),
    ("*", "times"),
)


def build_arithmetic_task(*, size: int = 60, seed: int = 0) -> Task:
    """Construct the arithmetic task.

    Args:
        size: Number of problems to generate.
        seed: Seed controlling operand and operator selection.

    Returns:
        A :class:`Task` graded by :class:`NumericGrader`.
    """
    rng = random.Random(seed)
    examples: list[Example] = []
    for index in range(size):
        symbol, _word = _OPERATORS[rng.randrange(len(_OPERATORS))]
        left = rng.randint(2, 99)
        right = rng.randint(2, 99)
        answer = _apply(symbol, left, right)
        examples.append(
            Example(
                input=f"What is {left} {symbol} {right}?",
                target=str(answer),
                id=f"arithmetic-{index:03d}",
                metadata={"operator": symbol},
            )
        )
    return Task(
        name="arithmetic",
        examples=examples,
        grader=NumericGrader(),
        description="Two-operand integer arithmetic word problems with numeric grading.",
    )


def _apply(symbol: str, left: int, right: int) -> int:
    if symbol == "+":
        return left + right
    if symbol == "-":
        return left - right
    return left * right

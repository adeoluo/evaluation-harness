"""The :class:`Model`: any system that produces a prediction for an example.

``Model`` is a :class:`~typing.Protocol`, so any object exposing a ``name`` and a
``generate`` method satisfies it - no inheritance required. This keeps the framework
open: a wrapper around a frontier API, a local model, or a deterministic stub are all
first-class citizens and are evaluated identically.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evaluation_harness.core.example import Example


@runtime_checkable
class Model(Protocol):
    """A system under evaluation.

    Implementations must be deterministic given a fixed seed to keep evaluations
    reproducible; stochastic models should expose their sampling seed through
    construction so a run can be replayed exactly.
    """

    @property
    def name(self) -> str:
        """A short, stable identifier used in reports (for example ``gpt-4o``)."""
        ...

    def generate(self, example: Example) -> str:
        """Return the model's raw prediction for ``example`` as text."""
        ...

"""The :class:`Example`: a single, immutable evaluation instance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Example:
    """One evaluation instance.

    Attributes:
        input: The prompt or question presented to the model.
        target: The reference answer, in string form. Graders compare the
            model's prediction against this value.
        id: A stable identifier, useful for joining results and for
            contamination checks. Generated from the input when omitted.
        choices: Ordered answer options for multiple-choice tasks. ``target``
            should then be the correct option's text or letter.
        metadata: Free-form, read-only annotations (category, difficulty, source).
    """

    input: str
    target: str
    id: str | None = None
    choices: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.input:
            raise ValueError("Example.input must be a non-empty string")
        if self.id is None:
            # Derive a deterministic identifier so results are reproducible.
            object.__setattr__(self, "id", _stable_id(self.input, self.target))

    @property
    def identifier(self) -> str:
        """The example id, guaranteed to be populated after construction."""
        assert self.id is not None  # established in __post_init__
        return self.id


def _stable_id(*parts: str) -> str:
    import hashlib

    digest = hashlib.sha256("\u241f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]

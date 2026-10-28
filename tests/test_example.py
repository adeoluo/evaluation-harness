"""Tests for the :class:`Example` data type."""

from __future__ import annotations

import pytest

from evaluation_harness.core import Example


def test_example_generates_stable_id() -> None:
    first = Example(input="2 + 2", target="4")
    second = Example(input="2 + 2", target="4")
    assert first.identifier == second.identifier


def test_different_inputs_get_different_ids() -> None:
    a = Example(input="2 + 2", target="4")
    b = Example(input="3 + 3", target="6")
    assert a.identifier != b.identifier


def test_explicit_id_is_preserved() -> None:
    example = Example(input="q", target="a", id="custom-1")
    assert example.identifier == "custom-1"


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Example(input="", target="a")


def test_metadata_and_choices_round_trip() -> None:
    example = Example(
        input="pick",
        target="B",
        choices=("A", "B", "C"),
        metadata={"category": "logic"},
    )
    assert example.choices == ("A", "B", "C")
    assert example.metadata["category"] == "logic"

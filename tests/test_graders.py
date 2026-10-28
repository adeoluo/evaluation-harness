"""Tests for the built-in graders."""

from __future__ import annotations

import pytest

from evaluation_harness.core import Example
from evaluation_harness.graders import (
    ChoiceGrader,
    ExactMatchGrader,
    IncludesGrader,
    NumericGrader,
    RegexGrader,
    extract_last_number,
    normalize_text,
)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_normalize_default_rules() -> None:
    assert normalize_text("  The Answer! ") == "answer"


def test_normalize_can_be_disabled() -> None:
    assert normalize_text("The Cat", strip_articles=False, strip_punctuation=False) == "the cat"


# --------------------------------------------------------------------------- #
# Exact match
# --------------------------------------------------------------------------- #
def test_exact_match_is_lenient_by_default() -> None:
    grader = ExactMatchGrader()
    example = Example(input="q", target="Paris")
    assert grader.grade(example, "  paris. ").passed


def test_exact_match_strict_distinguishes_surface_form() -> None:
    grader = ExactMatchGrader(normalize=False)
    example = Example(input="q", target="Paris")
    assert not grader.grade(example, "paris").passed
    assert grader.name == "exact_match_strict"


def test_exact_match_failure_has_rationale() -> None:
    grade = ExactMatchGrader().grade(Example(input="q", target="Paris"), "London")
    assert grade.score == 0.0
    assert grade.rationale is not None


# --------------------------------------------------------------------------- #
# Includes
# --------------------------------------------------------------------------- #
def test_includes_matches_embedded_answer() -> None:
    grader = IncludesGrader()
    example = Example(input="q", target="Paris")
    assert grader.grade(example, "The capital is Paris.").passed


def test_includes_rejects_absent_answer() -> None:
    grader = IncludesGrader()
    example = Example(input="q", target="Paris")
    assert not grader.grade(example, "The capital is London.").passed


# --------------------------------------------------------------------------- #
# Regex
# --------------------------------------------------------------------------- #
def test_regex_default_pattern() -> None:
    grader = RegexGrader(r"\b\d{3}-\d{4}\b")
    example = Example(input="q", target="")
    assert grader.grade(example, "call 555-1234 now").passed
    assert not grader.grade(example, "no number").passed


def test_regex_per_example_pattern_overrides_default() -> None:
    grader = RegexGrader()
    example = Example(input="q", target="", metadata={"pattern": r"ab+c"})
    assert grader.grade(example, "xx abbbc yy").passed


def test_regex_missing_pattern_raises() -> None:
    grader = RegexGrader()
    with pytest.raises(ValueError, match="No pattern"):
        grader.grade(Example(input="q", target=""), "text")


# --------------------------------------------------------------------------- #
# Numeric
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the answer is 42", 42.0),
        ("1,000", 1000.0),
        ("result: -3.5", -3.5),
        ("no digits", None),
        ("steps 1 then 2 then 99", 99.0),
    ],
)
def test_extract_last_number(text: str, expected: float | None) -> None:
    assert extract_last_number(text) == expected


def test_numeric_absolute_tolerance() -> None:
    grader = NumericGrader(absolute_tolerance=0.1)
    example = Example(input="q", target="10")
    assert grader.grade(example, "the answer is 10.05").passed
    assert not grader.grade(example, "the answer is 10.2").passed


def test_numeric_relative_tolerance() -> None:
    grader = NumericGrader(relative_tolerance=0.01)
    example = Example(input="q", target="1000")
    assert grader.grade(example, "1005").passed  # within 1%
    assert not grader.grade(example, "1020").passed


def test_numeric_handles_comma_formatted_answer() -> None:
    grader = NumericGrader()
    example = Example(input="q", target="1000000")
    assert grader.grade(example, "1,000,000").passed


def test_numeric_no_number_scores_zero() -> None:
    grade = NumericGrader().grade(Example(input="q", target="5"), "I am not sure")
    assert grade.score == 0.0


def test_numeric_negative_tolerance_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NumericGrader(absolute_tolerance=-1.0)


# --------------------------------------------------------------------------- #
# Choice
# --------------------------------------------------------------------------- #
def _choice_example() -> Example:
    return Example(input="Pick a capital", target="B", choices=("London", "Paris", "Berlin"))


def test_choice_accepts_letter_answer() -> None:
    grader = ChoiceGrader()
    assert grader.grade(_choice_example(), "I think the answer is B.").passed


def test_choice_accepts_text_answer() -> None:
    grader = ChoiceGrader()
    assert grader.grade(_choice_example(), "Paris").passed


def test_choice_wrong_answer_has_rationale() -> None:
    grade = ChoiceGrader().grade(_choice_example(), "A")
    assert grade.score == 0.0
    assert grade.rationale is not None


def test_choice_target_can_be_text() -> None:
    grader = ChoiceGrader()
    example = Example(input="q", target="Paris", choices=("London", "Paris", "Berlin"))
    assert grader.grade(example, "B").passed


def test_choice_requires_choices() -> None:
    grader = ChoiceGrader()
    with pytest.raises(ValueError, match="requires choices"):
        grader.grade(Example(input="q", target="A"), "A")


def test_choice_unparseable_prediction_scores_zero() -> None:
    grade = ChoiceGrader().grade(_choice_example(), "hmm, not sure")
    assert grade.score == 0.0

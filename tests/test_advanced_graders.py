"""Tests for the advanced graders: F1, fuzzy, collection, JSON, ROUGE-L, composite."""

from __future__ import annotations

import pytest

from evaluation_harness.core import Example
from evaluation_harness.graders import (
    AllGrader,
    AnyGrader,
    ExactMatchGrader,
    F1Grader,
    FuzzyMatchGrader,
    JsonFieldGrader,
    JsonValidGrader,
    ListMatchGrader,
    RougeLGrader,
    SetMatchGrader,
    WeightedGrader,
    extract_json,
    levenshtein,
    rouge_l,
    token_f1,
)


def ex(target: str) -> Example:
    return Example(input="q", target=target)


# --------------------------------------------------------------------------- #
# F1
# --------------------------------------------------------------------------- #
def test_token_f1_exact_is_one() -> None:
    assert token_f1("the quick fox", "the quick fox") == 1.0


def test_token_f1_partial() -> None:
    score = token_f1("Franklin D Roosevelt", "Franklin Delano Roosevelt")
    assert 0.0 < score < 1.0


def test_token_f1_disjoint_is_zero() -> None:
    assert token_f1("cat", "dog") == 0.0


def test_f1_grader_continuous_and_threshold() -> None:
    example = ex("Franklin Delano Roosevelt")
    assert 0 < F1Grader().grade(example, "Franklin Roosevelt").score < 1
    assert F1Grader(threshold=0.9).grade(example, "Franklin Roosevelt").score == 0.0


def test_f1_grader_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        F1Grader(threshold=2.0)


# --------------------------------------------------------------------------- #
# Fuzzy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("kitten", "sitting", 3), ("", "abc", 3), ("same", "same", 0), ("ab", "", 2)],
)
def test_levenshtein(a: str, b: str, distance: int) -> None:
    assert levenshtein(a, b) == distance


def test_fuzzy_match_tolerates_typos() -> None:
    grader = FuzzyMatchGrader(threshold=0.8)
    assert grader.grade(ex("Mississippi"), "Missisippi").passed


def test_fuzzy_match_rejects_distant() -> None:
    grader = FuzzyMatchGrader(threshold=0.8)
    assert not grader.grade(ex("Mississippi"), "Alabama").passed


def test_fuzzy_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        FuzzyMatchGrader(threshold=-0.1)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def test_set_match_order_independent() -> None:
    grader = SetMatchGrader(exact=True)
    assert grader.grade(ex("red, green, blue"), "blue, red, green").passed


def test_set_match_partial_jaccard() -> None:
    grader = SetMatchGrader()
    grade = grader.grade(ex("red, green, blue"), "red, green, yellow")
    assert 0 < grade.score < 1


def test_list_match_position_sensitive() -> None:
    grader = ListMatchGrader()
    full = grader.grade(ex("a, b, c"), "a, b, c")
    partial = grader.grade(ex("a, b, c"), "a, c, b")
    assert full.score == 1.0
    assert partial.score == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def test_extract_json_from_fence() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_embedded() -> None:
    assert extract_json('Here it is: {"x": 2} done') == {"x": 2}


def test_extract_json_none() -> None:
    assert extract_json("no json here") is None


def test_json_valid_requires_keys() -> None:
    grader = JsonValidGrader(required_keys=("name", "age"))
    assert grader.grade(ex(""), '{"name": "A", "age": 3}').passed
    assert not grader.grade(ex(""), '{"name": "A"}').passed


def test_json_field_extraction() -> None:
    grader = JsonFieldGrader("answer")
    assert grader.grade(ex("42"), '{"answer": "42", "work": "..."}').passed
    assert not grader.grade(ex("42"), '{"answer": "7"}').passed


def test_json_field_requires_object() -> None:
    assert JsonFieldGrader("a").grade(ex("x"), "[1, 2, 3]").score == 0.0


def test_json_field_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        JsonFieldGrader("")


# --------------------------------------------------------------------------- #
# ROUGE-L
# --------------------------------------------------------------------------- #
def test_rouge_l_identical() -> None:
    assert rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_subsequence_credit() -> None:
    score = rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.0 < score <= 1.0


def test_rouge_l_grader_threshold() -> None:
    example = ex("the quick brown fox")
    assert RougeLGrader(threshold=0.5).grade(example, "the quick brown fox").passed


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def test_any_grader_passes_if_one_passes() -> None:
    grader = AnyGrader([ExactMatchGrader(), F1Grader(threshold=0.5)])
    assert grader.grade(ex("new york city"), "new york").passed


def test_all_grader_requires_all() -> None:
    grader = AllGrader([ExactMatchGrader(), F1Grader(threshold=0.9)])
    assert not grader.grade(ex("new york city"), "new york").passed


def test_weighted_grader_average() -> None:
    grader = WeightedGrader([(ExactMatchGrader(), 1.0), (F1Grader(), 1.0)])
    grade = grader.grade(ex("new york city"), "new york")
    assert 0.0 < grade.score < 1.0


def test_weighted_grader_rejects_zero_weights() -> None:
    with pytest.raises(ValueError, match="not all be zero"):
        WeightedGrader([(ExactMatchGrader(), 0.0)])


def test_composite_requires_children() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AnyGrader([])

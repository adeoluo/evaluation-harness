"""Tests for n-gram contamination detection."""

from __future__ import annotations

import pytest

from evaluation_harness.contamination import (
    containment,
    detect_contamination,
    ngrams,
    tokenize,
)
from evaluation_harness.core import Example, Task

from .helpers import ExactGrader


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("The Cat, sat!") == ["the", "cat", "sat"]


def test_ngrams_basic() -> None:
    assert ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}


def test_ngrams_too_few_tokens_is_empty() -> None:
    assert ngrams(["a"], 2) == set()


def test_ngrams_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        ngrams(["a", "b"], 0)


def test_containment_full_and_partial() -> None:
    reference = ngrams(tokenize("the quick brown fox jumps"), 2)
    assert containment("the quick brown fox jumps", reference, n=2) == 1.0
    assert 0.0 < containment("the quick green turtle swims", reference, n=2) < 1.0


def _task() -> Task:
    examples = [
        Example(input="the quick brown fox jumps over", target="lazy dog", id="leaked"),
        Example(input="a completely unrelated novel sentence here", target="xyz", id="clean"),
    ]
    return Task(name="probe", examples=examples, grader=ExactGrader())


def test_detect_flags_leaked_example() -> None:
    reference = ["the quick brown fox jumps over the lazy dog"]
    report = detect_contamination(_task(), reference, n=3, threshold=0.5)
    assert not report.is_clean
    flagged_ids = {item.example_id for item in report.flagged}
    assert "leaked" in flagged_ids
    assert "clean" not in flagged_ids


def test_detect_clean_when_no_overlap() -> None:
    report = detect_contamination(_task(), ["nothing in common at all"], n=3)
    assert report.is_clean
    assert report.flagged_fraction == 0.0


def test_detect_reports_totals() -> None:
    report = detect_contamination(_task(), ["the quick brown fox jumps over"], n=3)
    assert report.total == 2
    assert 0.0 <= report.flagged_fraction <= 1.0


def test_detect_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        detect_contamination(_task(), ["x"], threshold=2.0)


def test_flagged_sorted_by_containment_descending() -> None:
    reference = ["the quick brown fox jumps over the lazy dog novel sentence"]
    report = detect_contamination(_task(), reference, n=2, threshold=0.0)
    scores = [item.containment for item in report.flagged]
    assert scores == sorted(scores, reverse=True)

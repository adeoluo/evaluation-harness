"""Built-in graders.

Each grader satisfies the :class:`~evaluation_harness.core.grader.Grader` protocol and is
independently usable. They span the common families of evaluation task:

- **String** - exact match, includes, regex, fuzzy (edit distance).
- **Numeric** - value comparison within tolerance.
- **Choice** - multiple choice by letter or text.
- **Overlap** - token F1 and ROUGE-L for partial credit on free-form answers.
- **Collection** - set and ordered-list matching.
- **Structured** - JSON validity and field extraction.
- **Model-graded** - a rubric judge model.
- **Composite** - combine any of the above (any / all / weighted).
"""

from __future__ import annotations

from evaluation_harness.graders.choice import ChoiceGrader
from evaluation_harness.graders.collection import ListMatchGrader, SetMatchGrader, parse_items
from evaluation_harness.graders.composite import AllGrader, AnyGrader, WeightedGrader
from evaluation_harness.graders.exact_match import ExactMatchGrader
from evaluation_harness.graders.f1 import F1Grader, token_f1
from evaluation_harness.graders.fuzzy import FuzzyMatchGrader, levenshtein, similarity
from evaluation_harness.graders.includes import IncludesGrader
from evaluation_harness.graders.json_grader import (
    JsonFieldGrader,
    JsonValidGrader,
    extract_json,
)
from evaluation_harness.graders.model_graded import DEFAULT_RUBRIC, ModelGradedGrader
from evaluation_harness.graders.normalization import normalize_text
from evaluation_harness.graders.numeric import NumericGrader, extract_last_number
from evaluation_harness.graders.regex_match import RegexGrader
from evaluation_harness.graders.rouge import RougeLGrader, lcs_length, rouge_l

__all__ = [
    "DEFAULT_RUBRIC",
    "AllGrader",
    # composite
    "AnyGrader",
    # choice
    "ChoiceGrader",
    # string
    "ExactMatchGrader",
    # overlap
    "F1Grader",
    "FuzzyMatchGrader",
    "IncludesGrader",
    "JsonFieldGrader",
    # structured
    "JsonValidGrader",
    "ListMatchGrader",
    # model-graded
    "ModelGradedGrader",
    # numeric
    "NumericGrader",
    "RegexGrader",
    "RougeLGrader",
    # collection
    "SetMatchGrader",
    "WeightedGrader",
    "extract_json",
    "extract_last_number",
    "lcs_length",
    "levenshtein",
    # shared
    "normalize_text",
    "parse_items",
    "rouge_l",
    "similarity",
    "token_f1",
]

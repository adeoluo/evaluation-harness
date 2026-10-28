"""Built-in graders.

Each grader satisfies the :class:`~evaluation_harness.core.grader.Grader` protocol and
is independently usable. They range from strict string comparison to numeric tolerance
to rubric-based model grading, covering the common families of evaluation task.
"""

from __future__ import annotations

from evaluation_harness.graders.choice import ChoiceGrader
from evaluation_harness.graders.exact_match import ExactMatchGrader
from evaluation_harness.graders.includes import IncludesGrader
from evaluation_harness.graders.model_graded import DEFAULT_RUBRIC, ModelGradedGrader
from evaluation_harness.graders.normalization import normalize_text
from evaluation_harness.graders.numeric import NumericGrader, extract_last_number
from evaluation_harness.graders.regex_match import RegexGrader

__all__ = [
    "DEFAULT_RUBRIC",
    "ChoiceGrader",
    "ExactMatchGrader",
    "IncludesGrader",
    "ModelGradedGrader",
    "NumericGrader",
    "RegexGrader",
    "extract_last_number",
    "normalize_text",
]

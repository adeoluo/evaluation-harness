"""Text normalization shared across string-based graders.

Normalization decisions are a frequent, silent source of measurement error: a model
that answers ``"  The Answer."`` should not be marked wrong against ``"the answer"``
merely over casing, whitespace, punctuation, or leading articles. These helpers make
the normalization explicit and configurable so a grader's leniency is a deliberate,
documented choice rather than an accident.
"""

from __future__ import annotations

import re
import string

_WHITESPACE = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_punctuation: bool = True,
    strip_articles: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Return a canonical form of ``text`` under the selected rules.

    The default settings implement the widely used "normalized exact match" from the
    open-domain question answering literature (lowercase, drop punctuation and
    articles, collapse whitespace).
    """
    result = text.strip()
    if lowercase:
        result = result.lower()
    if strip_punctuation:
        result = result.translate(_PUNCTUATION_TABLE)
    if strip_articles:
        result = _ARTICLES.sub(" ", result)
    if collapse_whitespace:
        result = _WHITESPACE.sub(" ", result).strip()
    return result

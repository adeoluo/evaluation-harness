"""A transparent heuristic for the entailment task.

Like the other heuristics, this is a deliberately simple, inspectable baseline whose only
purpose is to give the framework something above majority-class to measure. It uses word
overlap and a small set of negation cues: high overlap suggests entailment, an explicit
negation cue suggests contradiction, and otherwise neutral. It is not meant to be strong,
only honest and reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evaluation_harness.core.example import Example

_WORD = re.compile(r"[a-z]+")
_NEGATION = frozenset(
    {"not", "no", "never", "closed", "empty", "off", "lost", "crashed", "dead", "dry"}
)
_STOPWORDS = frozenset(
    {"a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of", "to", "and"}
)


@dataclass(frozen=True)
class OverlapEntailmentModel:
    """Classifies entailment by premise/hypothesis word overlap and negation cues."""

    name: str = "overlap-entailment"
    entail_threshold: float = 0.35

    def generate(self, example: Example) -> str:
        premise, hypothesis = _split(example.input)
        premise_words = _content_words(premise)
        hypothesis_words = _content_words(hypothesis)
        if not hypothesis_words:
            return "neutral"

        hypothesis_cues = set(_WORD.findall(hypothesis.lower()))
        premise_cues = set(_WORD.findall(premise.lower()))
        # A negation present on exactly one side suggests contradiction.
        if bool(hypothesis_cues & _NEGATION) ^ bool(premise_cues & _NEGATION):
            return "contradiction"

        overlap = len(premise_words & hypothesis_words) / len(hypothesis_words)
        if overlap >= self.entail_threshold:
            return "entailment"
        return "neutral"


def _split(text: str) -> tuple[str, str]:
    premise = ""
    hypothesis = ""
    for line in text.splitlines():
        lowered = line.lower()
        if lowered.startswith("premise:"):
            premise = line.split(":", 1)[1].strip()
        elif lowered.startswith("hypothesis:"):
            hypothesis = line.split(":", 1)[1].strip()
    return premise, hypothesis


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}

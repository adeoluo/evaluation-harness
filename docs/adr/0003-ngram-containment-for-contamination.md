# ADR-0003: N-gram containment for contamination detection

- **Status:** Accepted
- **Date:** 2026-01-01

## Context

A benchmark result is meaningless if the test items leaked into the model's training
data. A practical, model-agnostic check is n-gram overlap between each evaluation example
and a reference corpus (a sample of the training data). The question is how to score the
overlap.

## Decision

Score each example by **containment**: the fraction of the example's n-grams that appear
anywhere in the reference corpus. Flag the example when containment meets or exceeds a
threshold. Default to 13-grams, following common decontamination practice.

## Alternatives considered

- **Jaccard similarity** between the example's and the corpus's n-gram sets. An evaluation
  example is tiny relative to a training corpus, so the symmetric denominator makes the
  score vanishingly small even for a verbatim leak. It answers the wrong question.
- **Exact string match.** Misses examples with trivial edits (whitespace, casing) and
  examples embedded in a larger passage.

## Consequences

- An example that is verbatim-present in a large corpus scores near 1.0 and is flagged,
  which is the behavior we want.
- The reference n-gram set is held in memory; for very large corpora this is the main
  cost, and a streaming or hashed representation is a natural future optimization.
- Paraphrased leakage (no shared n-grams) is not detected; this is an accepted limitation
  of all n-gram methods and is documented in the threat model.

# Threat Model - Evaluation Harness

## Assets

- Integrity of the reported numbers (a misleading evaluation is the primary harm).
- Any API keys supplied by the user for provider adapters.

## Trust boundaries

- **Task data and model outputs** are untrusted text. They flow into graders and reports.
- **Provider adapters** are the only components that cross the network boundary.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Benchmark contamination | Test items present in training data | Inflated, misleading scores | N-gram containment scan; data card documents provenance |
| Judge manipulation | Prediction text crafted to steer a model-graded verdict | Wrong scores | Defensive verdict parsing; validate judge against human labels; prefer rule-based graders where possible |
| Unsafe expression evaluation | Arithmetic solver parsing model/task text | Code execution | Expressions are parsed to an AST and evaluated with a fixed operator allow-list; `eval` is never used |
| Secret leakage | API keys in logs or reports | Credential compromise | Keys read from environment only; never logged; `.env` is git-ignored |
| Supply-chain compromise | Malicious dependency | Code execution | Minimal dependencies (numpy core); pinned lockfile; Dependabot; CodeQL |
| Resource exhaustion | Very large task or corpus | Memory / time blowup | Deterministic, bounded bundled tasks; contamination reference set size is the documented cost |

## Residual risk

Paraphrased contamination (no shared n-grams) is undetectable by n-gram methods and
remains a known, documented limitation. Model-graded reliability ultimately depends on
the judge, which the user must validate for their setting.

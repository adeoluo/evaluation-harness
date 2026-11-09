# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.1] 2026-11-09

### Bug fixes

- Current OpenAI and Claude models, SDK majors and request parameters for the provider adapters (#11)

### Documentation

- Standard README structure and language framing (#2)
- Vertical layered architecture diagram
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Recorded terminal demo of a real run, with its vhs tape (#3)
- Remove authorship statements from docs and scripts
- Recorded demos of more commands, with their vhs tapes (#4)
- Recorded demos of the HTTP service and more commands (#5)
- Describe the supply chain accurately (#6)
- Complete citation metadata and bump it on release (#8)
- Diagrams on white in the shared theme, make diagram for every diagram, and a self-contained Definition of Done (#10)

### Build and CI

- Run CodeQL on main and on a weekly schedule only
- Build and attach the sdist and wheel to every release (#12)

### Maintenance

- Add release automation, AGENTS.md, llms.txt and contributor docs
- Classify squash merges in the changelog by the commits they contain
- Latest toolchain and CI versions (#7)
- Release script keeps hard-coded version strings in step (#9)

## [0.1.0] 2026-11-01

### Features

- **core:** Add evaluation primitives and bootstrap statistics
- **graders:** Add string, numeric, choice, and model-graded graders
- **models,tasks:** Add offline models, bundled tasks, and registries
- **cli:** Add contamination detection, reporting, and a CLI
- **graders:** Add F1, fuzzy, collection, JSON, ROUGE-L, and composite graders
- **significance:** Add paired significance testing and a compare command
- **runner:** Add concurrency, retries, and a caching model
- **persistence:** Add run records, regression detection, and rich reports
- **tasks:** Add entailment and factual-QA tasks with heuristics
- Add HTTP service, stress benchmark, and operations documentation
- Add production observability (metrics and readiness) and reword footer

### Documentation

- Write README, architecture diagram, reports, and cards
- Document the expanded capabilities and compare command
- Add evaluation benchmark and a performance section
- Use plain, factual wording (#20)

### Tests

- **sentiment:** Add negation and mixed-sentiment examples
- Validate from-scratch statistics against scipy; extend scale benchmark (#1)

### Maintenance

- Scaffold evaluation harness from standards template
- **deps:** Bump actions/upload-artifact from 4 to 7
- **deps:** Bump actions/checkout from 4 to 7
- **deps:** Bump astral-sh/setup-uv from 4 to 7
- **ci:** Bump GitHub Actions to current major versions
- Remove Dependabot configuration

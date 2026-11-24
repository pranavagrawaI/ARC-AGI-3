# ARC-AGI-3 Research

Empirical research, solver modules, and interpretability/evaluation studies for **ARC-AGI-3**.

## Featured Analysis

- **[De-Risking Cognitive-Map Measures in LLMs](analysis/cognitive-map-derisk/README.md)** ([full report](analysis/cognitive-map-derisk/summary.md)): Empirical stress test on `allenai/OLMo-2-0425-1B` evaluating whether linear probes of in-context HMM belief-state geometry (map formation and cross-alphabet structure reuse) form a valid benchmark-independent predictor of fluid intelligence. Includes full code, 21 unit tests, and compact JSON results.

## Repository Layout

| Path | Contents |
|---|---|
| [`analysis/`](analysis/) | Standalone research analyses, diagnostics, and compact result records (including [`analysis/cognitive-map-derisk/`](analysis/cognitive-map-derisk/)) |
| [`solvers/`](solvers/) | Stable, buildable solver modules (see [`solvers/README.md`](solvers/README.md)) |
| [`arc-host/`](arc-host/) | Environment-side ARC SDK boundary and persistent-process host protocol |
| [`experiments/`](experiments/) | Isolated experiment cards, precommitted contrasts, and compact outcome records |
| [`docs/`](docs/) | Architecture Decision Records (ADRs) and operational documentation |
| [`CONTEXT.md`](CONTEXT.md) | Shared domain glossary (oracle vs. admissible contestant boundaries) |
| [`AGENTS.md`](AGENTS.md) | Project engineering conventions and experiment-restoration notes |

## Notes on Artifacts & Submodules

- Compact results, experiment cards, and code are tracked in git; multi-gigabyte raw run transcripts, activation caches, and model weights are git-ignored.
- Some solvers pin external donor repositories via git submodules (see [`AGENTS.md`](AGENTS.md) for restoration commands).

# arc-agi-3

This file records project facts, quirks, and norms found during development. Add one when you discover it, beside related entries.

## Paid provider runs

Before qualifying, freezing, or launching a run that sends requests to a paid model provider (OpenRouter, DeepInfra, OpenAI API), read [docs/agents/paid-actor-runs.md](docs/agents/paid-actor-runs.md). It covers spend accounting, endpoint compatibility, deadlines, and continuation failures from earlier actor runs.

## Restoring experiments

Keep experiment cards and compact results with their experiment. Check ignored paths for cards during a backup. The F1x + REAP solver builds from two notebooks under `solvers/m2_f1/experiment/` and `analysis/hidden-draws-20261010/`. Run `python3 -B -m unittest solvers/m2_f1x_reap/tests/test_build.py` to check both notebook hashes and rebuild the merged notebook in memory.

The November 2026 cleanup removed local raw runs. Historical analyses that read full request transcripts or captured model weights need those inputs from the original run. The daily submission service reads its queue and draw log from `analysis/hidden-draws-20261010/`.

The Tycho/dfranzen solver assembles its runtime from a pinned external Git repository. After cloning, run `git submodule update --init -- analysis/m2-winners-20261001/audit/dfranzen-repo` before building that solver.

## Linear probing on synthetic sequences

When probing LLM activations on synthetic stochastic processes (e.g., HMM belief states under `analysis/cognitive-map-derisk/`), do not trust random within-sequence train/test splits or minimum-norm OLS: random splits leak strongly via temporal autocorrelation, and min-norm OLS fails catastrophically off-sequence. Use held-out-sequence splits, cross-validated Ridge regression, and a matched i.i.d. (`0-HMM`) sequence control alongside the next-token-probability baseline.

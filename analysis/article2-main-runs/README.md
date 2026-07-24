# Article 2 main-run analysis

This directory separates game-specific trace analysis from shared benchmark indexes and cross-game comparisons.

## Structure

- `games/<game-id>/`: all analysis specific to one game.
  - `data/`: derived CSV evidence owned by that game.
  - `scripts/`: extractors that produce that game's data.
  - `protocols/`: game-bound evaluation protocols, when needed.
- `cross-game/<comparison>/`: analyses whose unit is a matched pair or corpus spanning games.
  - `data/`: comparison outputs.
  - `scripts/`: comparison producers.
- `shared/`: benchmark-wide documentation and indexes used by multiple analyses.
  - `data/`: the single source for run matrices, baselines, per-level RHAE, per-game outcomes, and configuration scores.

## Game analyses

- [`cn04`](./games/cn04/): matched TWMA versus EWMA-SV trace audit, normalized ledgers, and the echo-chamber/frustration hypothesis note.
- [`ka59`](./games/ka59/): executable trace case study, prospective audits and comparison, normalized ledgers, extractor, and the KA59-bound learner protocol.
- [`ls20`](./games/ls20/): EWMA run analysis.

Create `games/<game-id>/` only when game-specific analysis exists; do not create empty folders for every archived game.

## Cross-game analysis

- [`text-vs-executable`](./cross-game/text-vs-executable/): matched TWMA versus EWMA-SV ranking and divergence corpus.

## Shared evidence

- [`rhae-overview.md`](./shared/rhae-overview.md): official reconstructed RHAE method and artifact map.
- [`archive-overview.md`](./shared/archive-overview.md): superseded completion-first archive survey, retained as diagnostic history.
- [`shared/data`](./shared/data/): benchmark-wide source tables. Game folders should reference these tables rather than duplicate them.

## Placement rule

The owner is determined by the claim's unit of analysis:

1. one game -> `games/<game-id>/`;
2. several games or matched treatment cells -> `cross-game/<comparison>/`;
3. benchmark-wide source index or scoring method -> `shared/`.

Within an owner, keep human-readable analysis at its root, generated tables in `data/`, producers in `scripts/`, and narrowly bound protocols in `protocols/`.
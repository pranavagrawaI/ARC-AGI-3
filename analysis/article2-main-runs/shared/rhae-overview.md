# RHAE reconstruction for the Article 2 main-runs archive

## Correction to the first-pass analysis

ARC-AGI-3's benchmark metric is **Relative Human Action Efficiency (RHAE)**, not binary game completion or unweighted level completion. Completion remains an input to the metric, but efficient skill acquisition is the quantity being scored.

Canonical methodology: <https://docs.arcprize.org/methodology>

For completed level $i$:

\[
s_i=\min\left(1.15,\left(\frac{H_i}{A_i}\right)^2\right)
\]

where $H_i$ is the official upper-median first-time-human action baseline and $A_i$ is the run's actions spent acquiring and completing that level. An unsolved level has score zero.

The weighted, pre-cap environment score is

\[
r_{raw}=\frac{\sum_{i=1}^{L} i\,s_i}{\sum_{i=1}^{L} i}.
\]

The toolkit also enforces the completion-weight cap

\[
c=\frac{\sum_{i\in\text{completed}} i}{\sum_{i=1}^{L} i},
\qquad
r_{game}=\min(r_{raw},c).
\]

The benchmark score for one configuration is the equal-weight mean of its 25 environment scores.

Completion fractions and win counts below are retained only as diagnostics. They must not be presented as the benchmark score.

## Derived artifacts

- [`data/run_matrix.csv`](./data/run_matrix.csv): 16 experiment configurations.
- [`data/human_baselines.csv`](./data/human_baselines.csv): current official versioned game IDs and per-level human baselines for all 25 games.
- [`data/per_level_rhae.csv`](./data/per_level_rhae.csv): 2,928 configuration/game/level records with reconstructed action intervals and RHAE contributions.
- [`data/per_game_outcomes.csv`](./data/per_game_outcomes.csv): 400 game runs with level actions, completion cap, weighted raw RHAE, and final capped RHAE.
- [`data/configuration_scores.csv`](./data/configuration_scores.csv): equal-weight aggregate RHAE for each of the 16 configurations.
- [`../games/ka59/ka59-ewma-sv-gpt-5.5-xhigh-trace-case-study.md`](../games/ka59/ka59-ewma-sv-gpt-5.5-xhigh-trace-case-study.md): full 23-turn/397-action audit of one executable-verification run, with exhaustive turn, message, action, file-change, and verifier CSVs.

The Windows archive was read in place and was not modified.

## Reconstruction method

### Human baselines

The 25 `baseline_actions` lists were retrieved from the official ARC toolkit's `Arcade.available_environments` metadata in `OperationMode.NORMAL`. The short game IDs exactly match the archive's 25 games, and every baseline length matches that game's archived `win_levels`.

The archive itself does not retain versioned public game IDs. The reconstruction therefore maps each archived short ID to the current official versioned ID listed in [`data/human_baselines.csv`](./data/human_baselines.csv). This is strongly supported by exact game-set and level-count agreement, but strict historical version identity is not independently encoded in the archive.

### Per-level agent actions

The session layout duplicates a level-completing action as both the previous level's final step and the next level's `initial_metadata.json`. Counting metadata files would therefore be wrong.

For reached level $i$, the reconstruction uses global-step boundaries:

- start = first-attempt `initial_metadata.json` global `step_index` for level $i$;
- end = next level's start boundary, or the final global step for the current/final level;
- $A_i = end-start$.

This counts supervisor resets as actions, matching the toolkit scorecard implementation: `Card.inc_reset_count()` increments both reset count and total action count. Initial game creation/reset is the zero boundary and is not charged.

For example, the previously analyzed `ewma_v1.2`, GPT-5.5 xhigh LS20 trajectory has scoring action counts:

`[23, 459, 73, 51, 268, 222, 416]`

The earlier directional-only counts omitted five reset actions; RHAE correctly charges those resets to the levels in which they occurred.

### Verification

All 400 reconstructed environment scores were independently recalculated using the installed official toolkit's `EnvironmentScoreCalculator`. The maximum absolute difference between the toolkit and the derived CSV values was approximately $5.0\times10^{-10}$ percentage points.

## Configuration-level RHAE

| Variant | Model | Effort | RHAE | Verified wins |
|---|---:|---:|---:|---:|
| `ewma_v1.2` | 5.4 | high | 33.540% | 8 |
| `ewma_v1.2` | 5.4 | xhigh | 44.717% | 9 |
| `ewma_v1.2` | 5.5 | high | 51.144% | 12 |
| `ewma_v1.2` | 5.5 | xhigh | 69.699% | 16 |
| `ewma_v1.2_s` | 5.4 | high | 30.594% | 5 |
| `ewma_v1.2_s` | 5.4 | xhigh | 53.067% | 11 |
| `ewma_v1.2_s` | 5.5 | high | 58.350% | 14 |
| `ewma_v1.2_s` | 5.5 | xhigh | 73.087% | 19 |
| `ewma_v1.2_sv` | 5.4 | high | 39.159% | 7 |
| `ewma_v1.2_sv` | 5.4 | xhigh | 53.708% | 12 |
| `ewma_v1.2_sv` | 5.5 | high | 65.631% | 18 |
| `ewma_v1.2_sv` | 5.5 | xhigh | **74.783%** | 20 |
| `twma_v1.2` | 5.4 | high | 34.159% | 6 |
| `twma_v1.2` | 5.4 | xhigh | 40.663% | 7 |
| `twma_v1.2` | 5.5 | high | 58.851% | 12 |
| `twma_v1.2` | 5.5 | xhigh | 72.494% | 19 |

The best observed cell is `ewma_v1.2_sv`, GPT-5.5 xhigh at **74.783% RHAE**, not its 92.476% mean level-completion fraction. The 17.693 percentage-point gap between those quantities is precisely the action-efficiency penalty that the completion-first summary erased.

There is one run per cell. These are observed scores, not replicated causal estimates of model, effort, simplification, or verification effects.

## RHAE-ranked matched comparisons

### 1. Fixed executable-verification harness: GPT-5.5 wins where GPT-5.4 does not

Treatment held fixed: `ewma_v1.2_sv`, xhigh. The table is filtered to GPT-5.5 verified wins versus GPT-5.4 non-wins, then ranked by RHAE difference.

| Game | GPT-5.4 result / RHAE | GPT-5.5 result / RHAE | Difference |
|---|---:|---:|---:|
| `sc25` | 0/6, 0.000% | 6/6, 70.768% | +70.768 pp |
| `sk48` | 3/8, 2.919% | 8/8, 65.167% | +62.248 pp |
| `ka59` | 5/7, 40.215% | 7/7, 100.000% | +59.785 pp |
| `wa30` | 7/9, 33.669% | 9/9, 92.699% | +59.031 pp |
| `cn04` | 2/6, 9.815% | 6/6, 67.544% | +57.728 pp |

This replaces the completion-fraction ranking. `ka59` and `wa30` become higher-priority trace pairs because their later-level completion and efficiency generate larger RHAE divergences.

A necessary counterexample is `r11l`: under the same treatment GPT-5.4 scored 100.000% and won, while GPT-5.5 scored 4.762% and stopped at 1/6. Model version does not induce a per-game total ordering.

### 2. GPT-5.5 xhigh textual versus executable-verification harness

Ranked by absolute RHAE difference:

| Game | Textual (`twma`) | Executable + verification (`ewma_sv`) | SV minus textual |
|---|---:|---:|---:|
| `r11l` | 6/6, 90.558% | 1/6, 4.762% | -85.796 pp |
| `bp35` | 6/9, 9.405% | 9/9, 66.883% | +57.477 pp |
| `wa30` | 7/9, 35.374% | 9/9, 92.699% | +57.326 pp |
| `ka59` | 7/7, 43.355% | 7/7, 100.000% | +56.645 pp |
| `vc33` | 7/7, 69.063% | 3/7, 20.552% | -48.512 pp |

`ka59` is especially valuable: both treatments solve the whole game, so its 56.645-point divergence isolates acquisition efficiency rather than merely completion. That is a better RHAE case study than a binary solve flip.

### 3. No universal-failure bucket

No game is a verified failure under all 16 configurations, and every game earns nonzero RHAE in at least one configuration. The requested universal-failure bucket is therefore empty.

The hardest games by **mean RHAE across all 16 cells** are:

| Game | Mean RHAE | Verified wins / 16 | Range |
|---|---:|---:|---:|
| `bp35` | 10.968% | 1 | 0.068–66.883% |
| `lf52` | 20.259% | 2 | 1.818–75.132% |
| `sk48` | 20.526% | 3 | 0.000–73.310% |

If two hard games are needed, `bp35` and `lf52` are the honest RHAE-based substitutes. They are **rarely solved and low-scoring**, not universally failed.

## LS20 across all configurations

| Variant | Model/effort | Result | Actions | RHAE |
|---|---|---:|---:|---:|
| `ewma_v1.2` | 5.4 high | 2/7 | 1,937 | 4.726% |
| `ewma_v1.2` | 5.4 xhigh | 3/7 | 1,874 | 10.378% |
| `ewma_v1.2` | 5.5 high | 7/7 | 2,304 | 20.918% |
| `ewma_v1.2` | 5.5 xhigh | 7/7 | 1,512 | 54.241% |
| `ewma_v1.2_s` | 5.4 high | 6/7 | 3,122 | 17.380% |
| `ewma_v1.2_s` | 5.4 xhigh | 7/7 | 759 | 100.000% |
| `ewma_v1.2_s` | 5.5 high | 5/7 | 3,926 | 9.084% |
| `ewma_v1.2_s` | 5.5 xhigh | 7/7 | 907 | 87.241% |
| `ewma_v1.2_sv` | 5.4 high | 6/7 | 2,646 | 38.316% |
| `ewma_v1.2_sv` | 5.4 xhigh | 7/7 | 2,201 | 58.436% |
| `ewma_v1.2_sv` | 5.5 high | 7/7 | 1,066 | 66.474% |
| `ewma_v1.2_sv` | 5.5 xhigh | 7/7 | 1,059 | 65.607% |
| `twma_v1.2` | 5.4 high | 1/7 | 1,515 | 3.571% |
| `twma_v1.2` | 5.4 xhigh | 6/7 | 2,646 | 30.051% |
| `twma_v1.2` | 5.5 high | 4/7 | 2,169 | 26.828% |
| `twma_v1.2` | 5.5 xhigh | 7/7 | 1,722 | 43.631% |

This makes the metric distinction concrete: seven LS20 configurations solve all seven levels, yet their RHAE scores range from **20.918% to 100.000%**. A win is not the result; it is only the prerequisite for competing on full-game acquisition efficiency.

## Recommended initial trace corpus

1. `sc25`, SV xhigh, GPT-5.4 versus GPT-5.5: maximum positive model divergence among the requested solve flips.
2. `ka59`, GPT-5.5 xhigh, TWMA versus SV: both solve, but RHAE differs by 56.645 points; this is the cleanest acquisition-efficiency comparison.
3. `r11l`, GPT-5.5 xhigh, TWMA versus SV: strongest negative executable-verification result.
4. `bp35`, GPT-5.5 xhigh, TWMA versus SV: strongest positive executable-verification result.
5. LS20 across all 16 cells: same game with large completion and efficiency variation, plus an independently understood mechanic reference.

Trace analysis should explain **where each level's action budget went**: discriminating probes, repeated probes, failed plans, resets, recovery after contradiction, generated planner actions, and execution overhead. Runtime and token use remain useful secondary costs, but they are not part of official RHAE.
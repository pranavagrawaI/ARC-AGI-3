# Article 2 main-runs archive overview — superseded completion-first draft

> **Superseded for scoring and pair selection.** This first-pass document used completion and wins as its primary ranking variables. ARC-AGI-3 is scored with RHAE, which also reconstructs per-level action efficiency relative to official human baselines and weights later levels. Use [`rhae-overview.md`](./rhae-overview.md), [`data/per_level_rhae.csv`](./data/per_level_rhae.csv), and [`data/configuration_scores.csv`](./data/configuration_scores.csv) for benchmark comparisons. The completion tables below remain only as historical diagnostics.

## Scope

Source archive:

`/mnt/c/AI/datasets/article2_main_runs/runs`

Derived indexes:

- [`data/run_matrix.csv`](./data/run_matrix.csv)
- [`data/per_game_outcomes.csv`](./data/per_game_outcomes.csv)

The archive was read in place. No source files under the Windows dataset were modified.

## Direct observations

### Run matrix

The archive contains 16 top-level configurations. Every configuration contains the same 25 game directories, for 400 game runs total.

The actual directory grammar is:

`<variant>_gpt-<model>_<effort>_run<run>`

The four observed variants are:

- `twma_v1.2`
- `ewma_v1.2`
- `ewma_v1.2_s`
- `ewma_v1.2_sv`

The models are `5.4` and `5.5`; efforts are `high` and `xhigh`; every configuration is `run1`.

The archive naming differs slightly from labels such as `ewma_sv_v1.2`: the actual variant is `ewma_v1.2_sv`.

### No standalone score summary is present

The dataset root contains only `runs/`. No shallow score, result, metric, evaluation, leaderboard, or summary file exists at the dataset, configuration, game, or game-run root.

A pruned filename search inside all 400 agent workspaces found no score-summary artifact. The only score-like filename was an unrelated image named `symmetric_states_sheet.png`.

None of the 400 `agent_runner.log` files contains a `score`, `reward`, `metric`, or `percent` field.

Every final per-game metadata record has exactly these keys:

- `action_input`
- `available_actions`
- `levels_completed`
- `state`
- `step_index`
- `win_levels`

Therefore the archive does not directly preserve an official per-game scalar score. Any score requiring hidden benchmark weights, action-efficiency normalization, or external evaluation cannot be reconstructed from a score field because that field is absent.

### Authoritative outcome source

A defensible per-game outcome can still be reconstructed from two independent sources:

1. The final metadata in `agent_run/client/session/level_N_attempt_M/` supplies `state`, `levels_completed`, `win_levels`, and global `step_index`.
2. `agent_runner.log` records the independent `session_inspector.py` result, including `solved` and `game_over`.

[`data/per_game_outcomes.csv`](./data/per_game_outcomes.csv) defines `VerifiedSolved=True` only when:

- final metadata state is `WIN`;
- `levels_completed == win_levels`; and
- the runner's independent session inspection says `solved=True`.

It also contains a derived `CompletionFraction = levels_completed / win_levels`. This is a transparent progress measure, **not the archive's official score**.

Across all 400 game runs:

- 195 ended in `WIN`;
- 203 ended `NOT_FINISHED`;
- 2 ended at `GAME_OVER` when the supervisor stopped.

### Configuration-level outcomes

| Configuration | Verified wins / 25 | Mean completion fraction |
|---|---:|---:|
| `ewma_v1.2`, GPT-5.4 high | 8 | 0.5056 |
| `ewma_v1.2`, GPT-5.4 xhigh | 9 | 0.6130 |
| `ewma_v1.2`, GPT-5.5 high | 12 | 0.7330 |
| `ewma_v1.2`, GPT-5.5 xhigh | 16 | 0.9144 |
| `ewma_v1.2_s`, GPT-5.4 high | 5 | 0.5247 |
| `ewma_v1.2_s`, GPT-5.4 xhigh | 11 | 0.6958 |
| `ewma_v1.2_s`, GPT-5.5 high | 14 | 0.8247 |
| `ewma_v1.2_s`, GPT-5.5 xhigh | 19 | 0.9294 |
| `ewma_v1.2_sv`, GPT-5.4 high | 7 | 0.5841 |
| `ewma_v1.2_sv`, GPT-5.4 xhigh | 12 | 0.7553 |
| `ewma_v1.2_sv`, GPT-5.5 high | 18 | 0.8815 |
| `ewma_v1.2_sv`, GPT-5.5 xhigh | 20 | 0.9248 |
| `twma_v1.2`, GPT-5.4 high | 6 | 0.5482 |
| `twma_v1.2`, GPT-5.4 xhigh | 7 | 0.6285 |
| `twma_v1.2`, GPT-5.5 high | 12 | 0.8123 |
| `twma_v1.2`, GPT-5.5 xhigh | 19 | 0.9201 |

These are descriptive results from one run per cell. They do not identify a causal treatment effect without replication or additional assumptions.

## Initial high-information comparison set

### 1. Same verified harness, GPT-5.5 succeeds and GPT-5.4 fails

Comparison is `ewma_v1.2_sv`, `xhigh`, holding game and run index fixed. The five largest completion divergences are:

| Game | GPT-5.4 | GPT-5.5 | Completion-fraction difference |
|---|---:|---:|---:|
| `sc25` | 0/6 | 6/6 WIN | +1.0000 |
| `cn04` | 2/6 | 6/6 WIN | +0.6667 |
| `sk48` | 3/8 | 8/8 WIN | +0.6250 |
| `su15` | 4/9 | 9/9 WIN | +0.5556 |
| `lf52` | 5/10 | 10/10 WIN | +0.5000 |

Other solve flips are `bp35`, `ka59`, `re86`, `s5i5`, and `wa30`.

Useful both-succeed efficiency comparisons under the same pair are:

| Game | GPT-5.4 steps | GPT-5.5 steps |
|---|---:|---:|
| `ls20` | 2,201 | 1,059 |
| `lp85` | 895 | 129 |
| `cd82` | 488 | 166 |
| `ar25` | 606 | 353 |
| `sb26` | 329 | 209 |

Only three games have both models fail under this exact harness/effort: `sp80`, `tn36`, and `dc22`.

### 2. GPT-5.5 textual versus executable-verification harness

The prompt files directly distinguish these treatments:

- `twma_v1.2` asks for a world model in Markdown and level reports.
- `ewma_v1.2_sv` requires an executable engine, state reconstruction, renderer, planner, replay verification against recorded frames, and planner verification.

At GPT-5.5 xhigh, the five largest absolute outcome differences are:

| Game | Textual (`twma`) | Executable + verification (`ewma_sv`) | Verification minus textual |
|---|---:|---:|---:|
| `r11l` | 6/6 WIN | 1/6 | -0.8333 |
| `sk48` | 3/8 | 8/8 WIN | +0.6250 |
| `vc33` | 7/7 WIN | 3/7 | -0.5714 |
| `lf52` | 6/10 | 10/10 WIN | +0.4000 |
| `bp35` | 6/9 | 9/9 WIN | +0.3333 |

This comparison is especially useful because the executable-verification configuration performs much better on some games and much worse on others. With one run per cell, that does not identify the causal effect of verification; it does refute treating the observed outcomes as a monotonic ordering by scaffold strength.

### 3. “Every architecture fails” bucket is empty

No game failed under all 16 configurations. Even restricting to the four GPT-5.5 xhigh variants, every game was solved by at least one variant.

The requested two universally failed games therefore do not exist in this archive. The nearest honest substitutes are the hardest games by cross-configuration win count:

- `bp35`: solved by 1 of 16 configurations;
- `sp80`: solved by 1 of 16;
- `dc22`: solved by 1 of 16.

These should be labeled **rarely solved**, not universally failed.

### 4. LS20 across all configurations

| Variant | Model/effort | Outcome | Global steps |
|---|---|---:|---:|
| `ewma_v1.2` | 5.4 high | 2/7 | 1,937 |
| `ewma_v1.2` | 5.4 xhigh | 3/7 | 1,874 |
| `ewma_v1.2` | 5.5 high | 7/7 WIN | 2,304 |
| `ewma_v1.2` | 5.5 xhigh | 7/7 WIN | 1,512 |
| `ewma_v1.2_s` | 5.4 high | 6/7 | 3,122 |
| `ewma_v1.2_s` | 5.4 xhigh | 7/7 WIN | 759 |
| `ewma_v1.2_s` | 5.5 high | 5/7 | 3,926 |
| `ewma_v1.2_s` | 5.5 xhigh | 7/7 WIN | 907 |
| `ewma_v1.2_sv` | 5.4 high | 6/7 | 2,646 |
| `ewma_v1.2_sv` | 5.4 xhigh | 7/7 WIN | 2,201 |
| `ewma_v1.2_sv` | 5.5 high | 7/7 WIN | 1,066 |
| `ewma_v1.2_sv` | 5.5 xhigh | 7/7 WIN | 1,059 |
| `twma_v1.2` | 5.4 high | 1/7 | 1,515 |
| `twma_v1.2` | 5.4 xhigh | 6/7 | 2,646 |
| `twma_v1.2` | 5.5 high | 4/7 | 2,169 |
| `twma_v1.2` | 5.5 xhigh | 7/7 WIN | 1,722 |

LS20 already shows why one run per cell must be interpreted cautiously: GPT-5.4 xhigh with simplification solved it in 759 steps, while GPT-5.5 high with the same variant stopped at 5/7 after 3,926 global steps. Model scale and nominal reasoning effort do not induce a total ordering on individual trajectories.

## Interpretation and next evidence requirement

The archive supports matched **trajectory studies**, not just aggregate-score comparisons. The strongest next unit is one game pair with:

- identical game, effort, and harness;
- different model;
- a verified outcome flip;
- enough trace structure to align hypotheses, probes, code changes, and failures.

`sc25` is the maximum-divergence model pair. `r11l` is the strongest negative result for executable verification. `sk48` is the strongest positive result for executable verification. Together they form a compact adversarial trio:

1. `sc25`: capability-scale divergence under fixed verified machinery;
2. `sk48`: verification/scaffolding success case;
3. `r11l`: verification/scaffolding failure case.

Before making a mechanism claim, each pair still needs a trace-level event table. Aggregate completion and step counts cannot distinguish better causal induction from persistence, more probes, generated-code quality, or recovery behavior.

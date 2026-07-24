# First-pass analysis: `ls20`, EWMA v1.2, GPT-5.5 xhigh, run 1

## Scope

Analyzed source run:

`/mnt/c/AI/datasets/article2_main_runs/runs/ewma_v1.2_gpt-5.5_xhigh_run1/ls20/run`

This is a read-only first pass over one game run. No raw dataset files were changed.

## Direct observations

### Run identity and terminal result

- The runner identifies the model as `gpt-5.5` with reasoning effort `xhigh`.
- The run began at `2026-07-08 18:48:26 +0000` and stopped at `2026-07-08 22:41:51 +0000`, a wall-clock duration of **3:53:25**.
- The final game metadata reports:
  - state: `WIN`
  - levels completed: `7`
  - required levels: `7`
  - final global step index: `1512`
  - final level-local action index: `227`
- The runner independently inspected the final workspace and reported `solved=True`, `game_over=False`, and stop reason `game solved`.

### Dataset shape

- Apparent size: approximately **25 MB**.
- Files under the run root: **11,256**.
- Important file classes:
  - 2,570 `.txt` files, predominantly ASCII game frames
  - 2,559 `.png` files, rendered versions of frames
  - 1,520 `.json` files, predominantly per-step metadata
  - 20 `.py` files, executable world models and planners
  - 20 `.md` files, world-model, reasoning-log, and level-report documents
  - 2 `.log` files, including the full external-agent event trace
- The high count of files without extensions mostly comes from the embedded Git workspace, not game observations.

### What each major directory/file means

- `prompts/`: the exact instructions and continuation/recovery prompts given to the agent.
- `agent.log`: the raw model/agent execution trace—messages, shell commands, command outputs, file edits, turn boundaries, and usage records.
- `agent_runner.log`: a compact supervisor-level start/stop and terminal-state record.
- `agent_run/`: the agent's working repository.
- `agent_run/client/session/level_N_attempt_M/`: observations from a specific level attempt.
- `*_metadata.json`: action, available actions, cumulative levels completed, state, and global step index.
- `*_final.txt`: settled 64×64 frame encoded as hexadecimal color indices.
- `*_final.png`: rendered version of the settled frame.
- `*_intermediate_XX.*`: animation frames produced within a single action.
- `world_model.md`: the final accumulated natural-language rule model.
- `level_N_reasoning_log.md`: compressed hypotheses, observations, corrections, and rejected explanations.
- `level_N_report.md`: end-of-level summary.
- planner `.py` files: executable models used to extract state and search candidate routes.

### Attempt history

The global step index includes five supervisor resets after `GAME_OVER`. Removing those resets leaves **1,507 directional game actions**.

| Level | Attempts | Directional actions | Failed attempts before solve | Share of directional actions |
|---:|---:|---:|---:|---:|
| 1 | 1 | 23 | 0 | 1.5% |
| 2 | 4 | 456 | 3 | 30.3% |
| 3 | 1 | 73 | 0 | 4.8% |
| 4 | 1 | 51 | 0 | 3.4% |
| 5 | 2 | 267 | 1 | 17.7% |
| 6 | 1 | 222 | 0 | 14.7% |
| 7 | 2 | 415 | 1 | 27.5% |

Terminal failures occurred at global steps `183`, `290`, `436`, `729`, and `1284`. The final `WIN` occurred at step `1512`.

Directional action totals:

- `ACTION1` (up): 462
- `ACTION2` (down): 472
- `ACTION3` (left): 266
- `ACTION4` (right): 307

The metadata stores the transition action that completed a level both as the previous attempt's last step and as the next level directory's `initial_metadata.json`. These duplicated boundary records must be deduplicated by global `step_index` when aggregating the dataset. Reset steps after failures appear as separate global indices but are not ordinary directional actions.

### Official RHAE

ARC-AGI-3 scores this run by per-level Relative Human Action Efficiency, not by the binary win or the ratio of aggregate actions. The official LS20 human baselines are `[22, 123, 73, 84, 96, 192, 186]`. Using global-step level boundaries and charging the five supervisor resets as actions, as the official toolkit scorecard does, gives agent level actions `[23, 459, 73, 51, 268, 222, 416]`.

| Level | Human baseline | Agent actions | Level RHAE |
|---:|---:|---:|---:|
| 1 | 22 | 23 | 91.493% |
| 2 | 123 | 459 | 7.181% |
| 3 | 73 | 73 | 100.000% |
| 4 | 84 | 51 | 115.000% |
| 5 | 96 | 268 | 12.831% |
| 6 | 192 | 222 | 74.799% |
| 7 | 186 | 416 | 19.991% |

After weighting levels 1–7, the official reconstructed environment score is **54.241% RHAE**. This is the relevant acquisition-efficiency result: the run wins, but expensive exploration on levels 2, 5, and especially the heavily weighted level 7 substantially reduces its benchmark score.

### Agent-compute trace

The raw trace records:

- 17 model turns
- 761 completed agent messages
- 892 completed command executions
- a final cumulative usage record of:
  - 101,323,949 input tokens
  - 98,818,560 cached input tokens
  - 550,398 output tokens
  - 265,460 reasoning-output tokens

Approximately 97.53% of reported input tokens were cached; the reported non-cached input difference is 2,505,389 tokens. These are provider/runtime usage fields preserved in the trace, not estimates derived from file size.

## What the game means

The repository's independently decoded `ls20` source says a level clears only when the controllable object reaches a goal while its **shape, color, and rotation** match the goal requirements. Attribute-changing objects cycle one component of that status. Movement alone cannot solve the game.

The run's learned world model converged on the same broad rule:

1. Move a 5×5 controllable block on a macro grid.
2. Read the current status icon and target icon.
3. Contact shape/color/rotation transformers enough times to make the status match the target.
4. Collect batteries when necessary to refill the action budget.
5. Account for walls, slide rails, moving transformers, scrolling viewports, and reserve lives.
6. Enter the target only after the status matches.

Complexity is introduced cumulatively:

- Level 1: movement, one rotation transformer, and target matching.
- Level 2: batteries, budget exhaustion, reserve lives, and repeated rotations.
- Level 3: color matching and slide/warp behavior.
- Level 4: more explicit slide rails and a shape transformer.
- Level 5: moving transformers, branch-dependent shape transitions, and a non-obvious target-entry route.
- Level 6: several action-phased moving symbols and a decoy target.
- Level 7: scrolling/revealing map, phase-sensitive contacts, multiple batteries, and a long route requiring combined symbolic and spatial planning.

## Interpretation

### What this run demonstrates

This is not merely a replay dataset. It contains three coupled evidence layers:

1. **Environment evidence**: exact actions, settled frames, animation frames, metadata, failures, and terminal state.
2. **Agent process evidence**: hypotheses, messages, tool calls, analyses, and corrections in `agent.log`.
3. **Agent-produced models**: Markdown world models plus executable Python planners.

That combination makes the run useful for studying whether the agent's stated beliefs were supported by observations, whether its executable model captured the same rules as its prose, and whether planning actually improved behavior after failures.

### Strong behavior

- It ultimately solved all seven levels.
- It maintained a cumulative model instead of restarting inference from scratch at every level.
- It wrote executable planners and used them to search routes before spending game actions.
- It retained failed-attempt evidence and changed strategy on retries.
- On level 7, it explicitly protected against an accidental extra timed-marker contact and verified that status and target matched before the final six-action entry.

### Weak behavior

- It scored **54.241% RHAE**. The earlier aggregate comparison of 1,507 directional actions to a separately verified 405-action human run is descriptive but is not the benchmark metric: RHAE uses official per-level upper-median human baselines, charges resets, squares each per-level action ratio, and weights later levels.
- Level 2 alone consumed 30.3% of all directional actions and required three failed attempts, showing that the early semantic inference was expensive.
- Levels 5 and 7 consumed another 45.2% combined, primarily because moving/phase-dependent objects and route staging created repeated false hypotheses.
- The raw trace is computationally large relative to the 25 MB artifact: the final cumulative record reports over 101 million input tokens, although most were cached.
- Several planners encode run-specific observed cycles, coordinates, routes, and target patterns. They are executable evidence of what worked in this game, but should not automatically be treated as a general ARC-AGI-3 learner.

## What verification uniquely establishes

The following are stronger than the agent's own prose claims:

- Final per-step metadata directly establishes `WIN`, `levels_completed=7`, and global step `1512`.
- The supervisor log independently confirms the solved stop condition.
- Per-attempt metadata directly establishes five `GAME_OVER` transitions and the attempt counts.
- The raw action metadata establishes the 1,507 directional actions after deduplicating level-boundary snapshots and separating five reset steps.
- The repository's source-decoding record independently confirms that `ls20` gates completion on shape, color, and rotation matching; this corroborates the learned world model.

The written reports alone do **not** establish that every inferred mechanic is universally correct. Some claims are observations from specific frames, and some planner rules are localized approximations. Generality would require replaying predictions against held-out transitions or other `ls20` runs.

## Recommended next analysis

For this one run, the highest-value next step is a compact transition table with one row per unique global step:

- level and attempt
- local/global step
- action
- state and levels completed
- whether the settled frame changed
- number of animation frames
- budget and reserve counts
- inferred player position/status/target
- nearby agent message or hypothesis
- whether the action confirmed, contradicted, or did not discriminate among hypotheses

That table would let us locate where the agent learned each mechanic, quantify wasted actions, and distinguish productive experiments from repeated or non-discriminating probes without rereading 4.4 MB of raw agent trace manually.

## Sources inspected

Raw run sources:

- `agent_runner.log`
- `agent.log`
- `prompts/main_prompt.md`
- `agent_run/world_model.md`
- `agent_run/level_7_reasoning_log.md`
- `agent_run/level_7_report.md`
- `agent_run/level7_planner.py`
- `agent_run/client/session/*/*_metadata.json`
- final level-7 settled frame and metadata

Repository corroboration:

- `teach/learning-records/0008-ls20-win-condition-decoded.md`
- `teach/learning-records/0004-ls20-completed-115pct.md`

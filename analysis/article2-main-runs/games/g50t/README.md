# `g50t`: executable modeling converts scarce game actions into falsifiable search

## Decision

Use `g50t` as the reverse-direction control for the `cn04` case study.

At matched game, model (`gpt-5.4`), reasoning effort (`xhigh`), and run index:

- **TWMA** stopped at **5/7 levels** after **3,485 global actions** and scored **29.135% RHAE**.
- **EWMA-SV** solved **7/7 levels** after **531 actions** and scored **93.883% RHAE**.

The environment-action ratio is **6.563×** and the RHAE gap is **64.748 percentage points**.

The same run is also cheaper in logged inference: EWMA-SV used **0.536×** TWMA's input tokens and **0.504×** its output tokens. Those TWMA totals require summing two cumulative-counter segments around a process recovery; the interrupted turn never completed, so they remain lower bounds. EWMA-SV nevertheless performed **1.636×** as many path-level artifact mutations and 167 verifier executions. It spent a larger share of its work on explicit model repair while using fewer turns, tokens, and irreversible environment actions.

The strongest defensible mechanism is narrower than “code reasons better”:

> EWMA-SV could turn a failed transition hypothesis into an executable counterexample, retain that counterexample across model edits, and search candidate tape stacks without paying one game action per branch. TWMA performed increasingly disciplined state-graph exploration, but became locked into expensive trajectory reconstruction: at least 565 of its 1,500 level-6 actions exactly replayed prior prefixes. It never acquired a compact transition model and planner capable of composing hidden tape phases before paying that reconstruction tax.

This reverses the outcome in `cn04`, but it also falsifies the strongest `cn04` narrative: **TWMA failure is not generally explained by frustration-driven epistemic closure.** On `g50t`, late TWMA exploration was prospective, state-aware, and explicitly targeted at untested edges. The failure was a cost-insensitive hidden-state frontier search: representational and budget-allocation related, not merely attitudinal.

## Scope and authority

The paired runs are:

| Arm | Archived run |
|---|---|
| TWMA | `twma_v1.2_gpt-5.4_xhigh_run1/g50t` |
| EWMA-SV | `ewma_v1.2_sv_gpt-5.4_xhigh_run1/g50t` |

Authoritative outcomes come from [`../../shared/data/per_game_outcomes.csv`](../../shared/data/per_game_outcomes.csv). The normalized pair is in [`data/g50t-summary.csv`](data/g50t-summary.csv).

The untouched source archive remains under:

```text
/mnt/c/AI/datasets/article2_main_runs/runs/
```

Analysis uses a WSL mirror under:

```text
/home/hipra/datasets/article2_main_runs/runs/
```

The mirror excludes PNGs and `*_intermediate_*` animation artifacts only. It retains all step metadata, initial/final text frames, logs, reports, executable model files, and Git history. [`data/g50t-source-manifest.csv`](data/g50t-source-manifest.csv) records the source and mirror paths, SHA-256 equality for ten control files, and equal metadata counts:

- TWMA: **3,491** metadata files in both source and mirror.
- EWMA-SV: **538** metadata files in both source and mirror.

## Plain-language game model

The player records movement tapes. `ACTION5` banks the current successful route as a helper and restarts the live body. On later moves, older helpers replay their tapes and can park on switches. Colored structures then alter reachability: bars unblock cells, purple structures cycle bodies, and later levels add an autonomous green body whose patrol interacts with the same switches.

A simple example:

1. Walk to a red switch.
2. Bank that path with `ACTION5`.
3. The helper replays and eventually remains on the switch.
4. The live body uses the opened route to reach another switch or the goal.

The hard part is not local movement. It is composing several recordings whose hidden replay phases align.

## Observations

### 1. The score gap is broad, not a single lucky level

[`data/g50t-level-summary.csv`](data/g50t-level-summary.csv) reconstructs every attempt and level:

| Level | TWMA attempts / actions / result | EWMA-SV attempts / actions / result |
|---:|---:|---:|
| 1 | 2 / 77 / solved | 2 / 148 / solved |
| 2 | 1 / 31 / solved | 1 / 31 / solved |
| 3 | 6 / 722 global / solved | 1 / 64 / solved |
| 4 | 2 / 168 / solved | 1 / 98 / solved |
| 5 | 8 / 987 / solved | 1 / 50 / solved |
| 6 | 12 / 1,500 / unsolved | 1 / 97 / solved |
| 7 | not reached | 1 / 43 / solved |

The largest discriminator is level 6: **1,500 versus 97 actions**, a **15.351×** action ratio.

The 2,954-action global gap is not caused by longer successful routes. TWMA's successful attempts used **220 actions collectively**, versus **400** for EWMA-SV. The gap closes exactly as:

```text
+3,051  failed-attempt actions
   +24  resets
  -180  successful-route actions
   +59  unfinished final attempt
=2,954
```

So the discriminator is failed-search cost. TWMA often found short routes after learning a level; it paid too much to reach and test those routes.

### 2. The global ledger contains three valid but scorecard-excluded calls

TWMA has two legitimate totals:

- global action archive: **3,485**;
- accepted scorecard actions: **3,482**.

The difference is exactly three client calls on level 3 after terminal metadata had already reported `GAME_OVER`. They emitted metadata but no final frame. The reconstructed vector is:

```text
[77, 31, 722, 168, 987, 1500]
```

After excluding those three terminal metadata-only calls, it matches the scorecard:

```text
[77, 31, 719, 168, 987, 1500, 0]
```

The global total is still the correct total for the archived action stream and the shared outcome table. The accepted vector is the correct scorecard total. Conflating them would create a false reconciliation failure.

Evidence: [`data/g50t-actions.csv`](data/g50t-actions.csv), [`data/g50t-summary.csv`](data/g50t-summary.csv), and [`data/g50t-level-summary.csv`](data/g50t-level-summary.csv).

### 3. TWMA became more disciplined, not more closed

By attempts 9–12 on level 6, TWMA was doing several things correctly:

- treating settled ASCII gameplay lines as stronger state keys than visually similar PNGs;
- merging branches that reached the same canonical state;
- extracting exact deterministic prefixes rather than rebuilding from memory;
- mining saved attempts into a state graph before spending more actions;
- choosing untested outgoing edges from known nodes;
- distinguishing the per-attempt `GAME_OVER` near step 130 from the external 1,500-action cross-attempt cap.

The late reasoning log records:

- a canonical zero-visible base reached through several rewind histories;
- an exact **92-action** prefix to a stronger between-reds state;
- a later **31-action** shortcut from reset to the unique lower-room state;
- explicit branch merges and blocked-edge inventories;
- a stopped attempt once the client rejected additional actions at the 1,500-action cap.

Primary evidence:

- archived `level_6_reasoning_log.md:193-305`;
- archived `level_6_report.md:3-36`;
- [`data/g50t-model-messages.csv`](data/g50t-model-messages.csv), especially the late level-6 messages around source log lines `6529-6791`;
- [`data/g50t-attempts.csv`](data/g50t-attempts.csv), TWMA level-6 rows.

This matters because it rejects a lazy diagnosis. TWMA was not simply repeating the same unexplained action sequence while claiming confidence. It was building a useful observational state graph.

But useful local information was purchased at the wrong price. Exact action and settled-frame comparison gives a conservative lower bound of **565 repeated actions**, or **37.7% of the complete level-6 budget**:

- attempt 5 repeated attempt 3 for 99 actions;
- attempts 6 and 7 each repeated the preceding 114-action prefix;
- attempt 8 repeated the preceding 119-action prefix;
- attempt 11 repeated attempt 10 for 92 actions;
- attempt 12 spent its final 27 actions rebuilding the same 31-action frontier it had just reached.

This excludes non-identical routes to already-known states, so it understates total setup cost. By attempts 6–10 the new probes mostly added one blocked edge or another merge into the canonical base. The dominant failure is therefore **reconstruction-tax lock-in**: long deterministic setup for low solve-relevant information gain.

### 4. TWMA’s graph incurred reconstruction tax and still lacked a compact generative state

The same evidence shows the limit of that discipline.

TWMA’s state key was largely observational: settled ASCII plus visible replay configuration. But the puzzle’s future also depends on latent tape contents, replay position within each tape, sticky switch memory, and autonomous-body phase. Visually equivalent or nearly equivalent frames can therefore hide different futures until enough history is understood.

TWMA partially recognized this problem, but its artifact remained a long prose accumulation of level- and attempt-specific facts. By the end:

- `world_model.md` was **643 lines**;
- the level-6 reasoning log was **305 lines**;
- 12 level-6 attempts had produced no planner capable of searching the latent tape state;
- the best next move remained another set of live probes from the 31-action frontier, requiring another paid reconstruction.

The exact error was not “no model.” It was **a model whose operational state was too implicit to execute and search, combined with no cost gate against rebuilding a 31–119 action frontier for one more local edge**.

### 5. EWMA-SV also formed wrong hypotheses

EWMA-SV did not avoid error.

On level 6 it first misread the green body as a replay-like helper, then as a position-correspondence effect. It revised the parser and transition model several times. At global step 413 it had banked two helper tapes and began a continuation that its model predicted would solve the level.

That **37-action continuation failed**: the client remained `NOT_FINISHED` at level-local step 59. The world-model verifier then localized the first discrepancy to level-6 step 26, global step 417. Because the continuation was batch-executed with only coarse terminal checks, **33 later actions (g418–g450) were still charged after the model justification had become false**. Verification caught the error post hoc, not at the live boundary. Subsequent replay checks exposed:

- the green entity follows an autonomous corridor patrol rather than a recorded tape;
- `ACTION5` resets its patrol phase;
- a variable-shadowing bug in the executable engine had overwritten the player move target.

After the world model again passed levels 1–6 replay verification—with the documented level-6 visual-override caveat—EWMA-SV executed a corrected **38-action continuation** and completed level 6 at 97 total level actions.

Evidence:

- [`data/g50t-actions.csv`](data/g50t-actions.csv), EWMA-SV level-6 global steps `392-488`;
- [`data/g50t-verifications.csv`](data/g50t-verifications.csv), especially records around source log lines `2919`, `2943`, `2962`, `2974`, and `2998`;
- archived `level_6_report.md:3-13`.

The executable artifact did not make the first theory true. It made the theory **cheap enough to contradict and repair while the attempt was still alive**. The 33-action detection lag also shows why verifier-backed execution should be stepwise at uncertain boundaries rather than blindly batched.

### 6. Level 7 is the strongest prospective success evidence

Level 7 began at global step 488 with only `initial_frame.*` and `initial_metadata.json`; no level-7 actions had been taken. The g488 model returned no plan and misparsed the untouched scene as 37 floor cells, two structures, and no autonomous body.

Before touching the client, EWMA-SV repaired three parser assumptions:

- only the actual HUD footprint should be masked, not the whole top strip;
- disconnected same-color purple networks are separate structures;
- a green patrol can be detected from its sprite instead of a level table.

The repaired parser produced 42 floor cells, four structures, and one autonomous body. The planner then emitted a **43-action** route and model-side simulation ended `LEVEL_COMPLETED` at step 43. The same unchanged sequence was executed against the real client and produced `WIN` at global step 531.

This is stronger than a post-solve replay pass: it is a prospective plan from a previously untouched level state. It does not prove general planning beyond this game family, but it directly demonstrates operational use of the executable model.

Primary evidence: archived `agent.log:3382-3451`, especially the untouched-state inventory at `3382-3386`, failed old parse/planner at `3392-3399`, repaired parse at `3427`, generated plan at `3435`, model-side completion at `3440`, and real execution at `3451`.

### 7. Verification was active, noisy, and consequential

EWMA-SV invoked recognized verifiers **167 times**. Three `pkill` process-control commands mentioned `verify_main_planner.py` but were not verifier executions and are excluded:

| Verifier outcome | Count |
|---|---:|
| world model passed | 59 |
| world model failed | 29 |
| planner passed | 50 |
| planner failed | 24 |
| planner skipped unsolved target | 5 |

A failed verifier was not always a puzzle-mechanic falsification. Failures also found:

- frame-rendering mistakes;
- parser regressions against earlier levels;
- `MemoryError` and timeout boundaries;
- recursive planner bugs;
- the level-6 engine variable-shadowing bug.

That distinction is important. “167 verifier calls” is not 167 independent scientific tests. The useful contract was narrower:

1. replay every recorded transition;
2. reject model changes that regress earlier levels;
3. require the planner to terminate in `LEVEL_COMPLETED` in the model;
4. compare real execution against the predicted trajectory;
5. localize and retain the first divergence after execution.

The level-6 batch did not stop at the first divergence, so the fifth item describes the replay contract, not perfect online enforcement.

The final archived state reports:

- world-model verification passed levels **1–7**;
- planner verification passed levels **1–7**;
- the real environment reported **WIN**.

Evidence: final rows of [`data/g50t-verifications.csv`](data/g50t-verifications.csv).

### 8. The final executable model is not cleanly universal

The final model should not be over-credited.

Its provenance is also weaker than the checkpoint table alone suggests. The final Git checkpoint is `d54e687…` at global step 488, immediately after level 6. There is no g531 checkpoint. The level-7 parser, planner, reports, and winning model remain in the archived **uncommitted working tree**, with substantial post-HEAD changes to the core model files. The real environment trace still establishes the win, but reproducing the exact winning source requires preserving that working tree rather than checking out the final commit.

The verifier source itself is unchanged from the checkpoint, which reduces—but does not eliminate—the risk of post-hoc verifier weakening.

Its core mechanics are compact and shared:

- tape stack and common replay clock;
- sticky and non-sticky switches;
- `block` and `cycle` movement edits;
- autonomous corridor bodies;
- tape-stack planning over goal and switch landmarks.

But exact frame replay for level 6 still uses a narrow `apply_render_overrides(...)` table. The verifier itself warns:

> treat this as temporary and as a clue to a missing puzzle mechanic

The table is described as visual-only debt and the planner does not depend on it. Even so, exact frame verification is partly post-hoc for those intervals. Therefore:

- **supported:** the transition model and planner were operationally sufficient to solve levels 6 and 7;
- **not supported:** the final renderer is a fully general causal account of every pixel transition;
- **still open:** whether the override hides a mechanic that would matter on a new level.

Primary evidence: archived `world_model.md:16-20`, `verify_world_model.py:31-37`, and `world_model_state_io.py:115-127`.

### 9. Environment efficiency did not require more logged inference

From [`data/g50t-summary.csv`](data/g50t-summary.csv):

| Resource | TWMA | EWMA-SV | EWMA-SV / TWMA |
|---|---:|---:|---:|
| Global actions | 3,485 | 531 | 0.152× |
| Logged input tokens | 272,145,567 | 145,850,946 | 0.536× |
| Logged output tokens | 1,695,693 | 854,154 | 0.504× |
| Turns started (completed) | 57 (56) | 33 (33) | 0.579× started |
| Path-level file mutations | 162 | 265 | 1.636× |
| Surviving artifact lines | 1,508 | 1,495 | 0.991× |

TWMA's token counters reset when the master process exited and recovery started. The extractor now detects that decrease and sums the two process endpoints rather than taking only the final one. Because the interrupted turn has no `turn.completed` usage record, the TWMA total is a lower bound. This does not threaten the direction: even the lower bound is **1.866×** EWMA-SV's input and **1.985×** its output.

EWMA-SV did not merely write more final code: surviving artifact size was almost equal. It performed more path mutations and verifier iterations, then compressed back to a similarly sized final artifact while using fewer completed turns and fewer logged tokens.

The decision implication is concrete:

> On this trace, executable verification dominates both environment-action and logged-token accounting. That is stronger than an action-for-compute trade, but still not a treatment estimate: a second seed or isolated scaffold ablation could change the inference comparison.

## Interpretation

### Observation

TWMA built an increasingly careful observational state graph but spent at least 37.7% of the level-6 budget exactly reconstructing old prefixes. EWMA-SV built an executable transition system and planner, executed one wrong prospective continuation, localized the contradiction after a 33-action batch lag, repaired the model, and solved within the same attempt. On untouched level 7, its repaired parser and planner produced the exact 43-action winning route before any live action.

### Interpretation

The useful difference is **where uncertainty was allowed to live**:

- TWMA kept latent mechanics in prose and resolved them through live probes reached by increasingly expensive deterministic setup.
- EWMA-SV had to encode latent mechanics as state variables and transitions that could be replayed, searched, and contradicted offline.

A useful analogy is a subway map versus a simulator. TWMA eventually had a good map of many stations and observed connections. EWMA-SV had a flawed train simulator. The simulator was initially wrong, but once a train arrived at the wrong station, the exact transition could be fixed and every old route rerun. The map alone did not answer which sequence of timed transfers would work.

### Mechanism hypothesis

The candidate mechanism has three parts:

1. **Queryable compression:** latent tape phase, switch memory, and autonomous-body phase become explicit machine state rather than prose context.
2. **Counterexample retention:** a mismatch becomes a regression test that survives later edits.
3. **Offline branch multiplication:** candidate tape stacks can be searched without consuming environment actions, then checked prospectively against the real client.
4. **Reconstruction-tax avoidance:** known prefixes are simulated rather than repeatedly repurchased, and expensive live frontiers can be gated by explicit model predictions.

The observed action savings are consistent with this mechanism.

### What would falsify it

The mechanism would weaken materially if any of the following held:

- EWMA-SV’s winning route depended on level-specific scripts or frame overrides in the planner rather than shared transition mechanics;
- removing replay verification while retaining the executable planner did not increase live-action waste;
- giving TWMA the same explicit latent-state ledger and offline search failed to reduce level-6 actions;
- repeated seeds showed the pair was dominated by run variance;
- inference-normalized evaluation reversed the practical preference.

The current archive cannot resolve those counterfactuals.

## What `g50t` changes about the `cn04` conclusion

`cn04` supported a mechanism in which executable mismatch localization resisted a failing textual narrative. `g50t` preserves part of that conclusion but rejects its strongest generalization.

### Preserved

- Executable mismatch boundaries can expose an incorrect mechanism before many more live actions are spent.
- Model regression checks can keep later edits from silently breaking earlier explanations.
- Planner execution forces hypotheses to become operational commitments.

### Rejected as a general explanation

- TWMA failure is not always frustration-driven repetition.
- Textual analysis is not necessarily non-prospective.
- Exact state comparison and branch discipline are not unique to executable artifacts.

### Strengthened replacement

The cross-game claim should be:

> Executable modeling is most valuable when the bottleneck is **compositional latent state with expensive frontier reconstruction**: the agent has many correct local observations but must search interactions among hidden phases, memories, or concurrent entities, and each additional live branch requires replaying a long setup. Its advantage is smaller when the decisive uncertainty is a single local mechanic that can be isolated cheaply in text.

That is a more precise and falsifiable deployment rule than “EWMA reasons better.”

## Causal limits

This is one matched pair, not an ablation.

Controlled:

- game;
- model name;
- reasoning effort;
- run index;
- benchmark outcome and action accounting.

Not isolated:

- complete scaffold and prompt;
- executable tooling and verifier availability;
- policy effects induced by artifact requirements;
- total inference tokens and wall time;
- run-to-run stochasticity;
- post-solve artifact refinement.

Therefore the trace establishes a **mechanism-consistent case study**, not an average treatment effect.

The strongest causal next test is not another retrospective narrative. It is a level-6 ablation with frozen model and prompt budget:

1. EWMA-SV with replay verifier and planner;
2. executable planner with verifier disabled;
3. TWMA plus an explicit latent-state schema and state-graph search tool;
4. ordinary TWMA.

Measure environment actions, model tokens, first-divergence detection latency, and whether a failed prospective route becomes a retained regression case.

## Artifacts

| Artifact | Purpose |
|---|---|
| [`data/g50t-source-manifest.csv`](data/g50t-source-manifest.csv) | Source/mirror paths, hashes, metadata counts, mirror policy |
| [`data/g50t-summary.csv`](data/g50t-summary.csv) | Outcome, resource, scorecard, and artifact summary |
| [`data/g50t-level-summary.csv`](data/g50t-level-summary.csv) | Per-level attempts, actions, deaths, and success cost |
| [`data/g50t-attempts.csv`](data/g50t-attempts.csv) | Attempt boundaries and terminal states |
| [`data/g50t-actions.csv`](data/g50t-actions.csv) | Per-action normalized ledger and frame deltas |
| [`data/g50t-turns.csv`](data/g50t-turns.csv) | Model turns, action ranges, and token counters |
| [`data/g50t-model-messages.csv`](data/g50t-model-messages.csv) | Prospective claims and trace-local decisions |
| [`data/g50t-file-changes.csv`](data/g50t-file-changes.csv) | Artifact mutation events |
| [`data/g50t-verifications.csv`](data/g50t-verifications.csv) | Verifier outcomes and localized mismatches |
| [`data/g50t-inspections.csv`](data/g50t-inspections.csv) | Supervisor inspection state |
| [`data/g50t-checkpoints.csv`](data/g50t-checkpoints.csv) | Archived Git checkpoint lineage |
| [`scripts/extract_g50t_matched_trace.py`](scripts/extract_g50t_matched_trace.py) | Deterministic extractor and reconciliation assertions |

Regenerate with:

```bash
uv run python analysis/article2-main-runs/games/g50t/scripts/extract_g50t_matched_trace.py
```

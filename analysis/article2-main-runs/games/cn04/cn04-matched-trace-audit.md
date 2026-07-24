# `cn04` matched trace audit: textual vs executable world model

## Decision

Start with **`cn04`, GPT-5.4, `xhigh`, run 1**.

This is a clean negative case for the executable scaffold:

| arm | terminal state | levels | reconstructed global-level counts | reconstructed global count | reconstructed RHAE |
|---|---|---:|---|---:|---:|
| TWMA textual | `WIN` | 6/6 | `[16,236,31,34,77,249]` | 643¹ | 77.573338270% |
| EWMA-SV executable | `NOT_FINISHED` | 2/6 | `[14,80,1500]` | 1,594 | 9.815476190% |

¹ The scorecard records 640 accepted actions; the official reconstruction includes three post-terminal calls. The reconciliation below preserves both ledgers.

The gap is **67.757862080 percentage points**. Both traces identify the model as `gpt-5.4` with `xhigh` reasoning (`T/agent.log:1`, `E/agent.log:1`), and their level 1–3 initial frames are byte-identical. Content was copied from the untouched Windows archive and checksum-verified.

Aliases used below:

- `T` = `/home/hipra/datasets/article2_main_runs/runs/twma_v1.2_gpt-5.4_xhigh_run1/cn04/run`
- `E` = `/home/hipra/datasets/article2_main_runs/runs/ewma_v1.2_sv_gpt-5.4_xhigh_run1/cn04/run`

## Ledger reconciliation

Two internally valid ledgers disagree for TWMA:

| ledger | total | level 6 | RHAE | what it counts |
|---|---:|---:|---:|---|
| archived scorecard | 640 | 246 | 77.717731353% | accepted game actions, including resets |
| official reconstruction used in [`../../shared/data/per_game_outcomes.csv`](../../shared/data/per_game_outcomes.csv) | 643 | 249 | 77.573338270% | the global action index, including three post-`GAME_OVER` calls |

The residual is not reset bookkeeping. Level-6 attempt 1 died at local action 200/global 594, but an unguarded shell batch still issued local calls 201–203/global 595–597. Those calls produced only `GAME_OVER` metadata—no new frames—before reset at global 598 (`T/agent.log:1506,1529,1567-1569`; `T/agent_run/client/session/level_06_attempt_01/step_0201_metadata.json` through `step_0203_metadata.json`). The scorecard excludes that dead-state tail; the global counter and reconstructed benchmark include it.

**Decision:** retain 643 and 77.573338270% as the requested official reconstructed outcome, but call it reconstructed global-action accounting rather than physical environment transitions. For behavioral efficiency, report the scorecard's 640 accepted actions alongside it. EWMA-SV has no analogous discrepancy: scorecard, history, global counter, and reconstructed total all equal 1,594.

## Game primer

The board contains rigid wire-like pieces on a coarse grid. `ACTION1..4` translate the selected piece, `ACTION5` rotates or advances a piece-specific shape state, and `ACTION6(x,y)` selects a piece by clicking it. Red cells are exposed connectors; gray cells mark overlaps. A top bar advances with actions and eventually kills an unresolved attempt.

Level 3 introduces partial observability: only the selected piece is bright and exposes its red endpoints. The other two pieces remain present but are dark and hide their endpoints. Deselecting a piece therefore **deletes connector pixels from the observation, not connectors from the latent environment**. This distinction is the center of the run.

## What happened

### Levels 1–2: the executable arm was better

EWMA-SV solved level 1 in 14 actions versus TWMA's 16, then solved level 2 on its first 80-action attempt. TWMA needed three level-2 attempts and 236 reconstructed actions:

1. It treated blue/green as fixed targets and failed after 100 actions.
2. It corrected to multiple selectable pieces but tested a triple-overlap assembly and failed after another 100.
3. It deleted the triple-overlap rule, searched for clean pairwise docks, and solved in 34 ordinary actions.

This matters: the result is not “executable models are generally bad.” With fully visible, stable mechanics, EWMA-SV converted observations into a reusable transition model and planner more efficiently.

### The first meaningful divergence: level-3 action 6 versus action 509

TWMA reached level 3 and, at **local step 6**, explicitly asked whether the gray structures were inactive selectable pieces rather than fixed scenery. It clicked the upper gray structure, then the lower one at step 7, because that single probe chose between two different ontologies (`T/agent.log:793-803`). Both clicks produced large frame changes. With all three endpoint sets observed, it searched a three-piece cycle and solved at local step 31 (`T/agent.log:806-819`; `T/agent_run/level_3_reasoning_log.md:3-14`).

EWMA-SV entered level 3 with a one-movable-piece/fixed-gray ontology. Its post-attempt-1 checkpoint says the gray shapes are fixed conductors and the source alone moves (`git show cd2846c:world_model.md`). It did not click a hidden gray piece until **attempt 5, local step 5**: charged level-3 action 509. That is 503 charged actions later than TWMA's probe. The click immediately falsified the fixed-gray ontology; the agent then clicked the lower wire and transferred control among all three (`E/agent.log:3197-3231`).

The divergence was therefore not search quality inside a shared state space. It was **when each arm tested which objects belonged in the state space**.

## Where TWMA failed, and why it recovered

| location | observed failure | interpretation | recovery |
|---|---|---|---|
| level 2 attempt 1 | fixed-target plan reached `GAME_OVER` | incorrect object affordance: blue was selectable | add multiple selectable pieces |
| level 2 attempt 2 | triple assembly reached `GAME_OVER` | incorrect completion rule: triple stacking is not clean pairwise docking | delete triple-overlap completion; search pairwise contacts |
| level 4 | a 28-action cycle plan remained nonterminal | ordinary rotation had been mislabeled clockwise | re-read footprints, correct to counterclockwise, solve with three actions |
| level 6 attempt 1 | 200 accepted actions plus 3 dead-state calls; `GAME_OVER` | correct high-level growth idea was corrupted by execution against shifted geometry | reset environment but retain learned model; execute from clean state |

The level-6 reset is the cleanest addition/deletion example. Resetting deleted every player-created board arrangement. It did **not** delete the learned contextual `Y5` growth state. The second attempt therefore combined a clean environment with a richer model and solved in 45 ordinary actions (`T/agent_run/level_6_reasoning_log.md:30-45`).

TWMA also updated its model when the environment added geometry. On level 5, repeated `ACTION5` calls did not merely rotate the central P-shaped piece; they exposed additional connector states. The model deleted its rigid-rotation-only assumption, added finite growth states, and solved in 77 actions (`T/agent_run/level_5_reasoning_log.md:3-24`).

TWMA's metacontrol was materially below its demonstrated ceiling. In level-6 attempt 1 it deliberately spent 80 actions on supposedly blocked/budget-burning moves even after the board began drifting (`T/agent.log:1413-1438`), then failed to stop its final shell batch after `GAME_OVER`. The win came from persistence and repair, not consistently disciplined action selection.

## Where EWMA-SV succeeded

EWMA-SV's executable machinery did three useful things:

1. **Exact replay exposed rendering/dynamics mismatches.** The world-model verifier compares the predicted status and complete rendered frame against archived steps (`E/agent_run/verify_world_model.py:72-115`). Several model revisions were legitimately driven by concrete mismatches rather than prose intuition.
2. **The planner was effective under an identified model.** Levels 1 and 2 completed in 14 and 80 actions. Their successful traces became planner/regression evidence.
3. **Failure hypotheses became progressively narrower.** After discovering the hidden pieces, attempts 6–11 tested all-end overlap, clean pairwise overlap, deselection commit, alternate selected views, and a revised geometry. The agent did not merely repeat one identical plan.

That is real success. The failure is that these strengths were applied after a 503-action delay and against an underidentified completion model.

## Where EWMA-SV failed

### 1. It optimized inside the wrong ontology before testing the ontology

Attempts 1–4 each consumed a full 125-action bar while treating gray pieces as fixed or treating only the source click state as relevant. The high-information test was cheap—click a gray structure once—and prior level 2 had already established multiple selectable pieces. Yet the executable arm prioritized reconstructing, simplifying, replaying, and planning within the fixed-gray model.

This is the exact reasoning error: **it treated uncertainty over parameters and timing as primary while leaving uncertainty over object affordances untested**.

A falsifier would be an early trace showing that clicking gray had already been attempted and produced no evidence. It does not exist. The first such click is attempt 5.

### 2. Simplification removed evidence faster than exploration replenished it

After attempt 1, the supervisory flow repeatedly demanded aggressive generalization and deletion of distinctions (`E/agent.log:1653`, `1700`, and later analogous prompt bodies). The model reported:

- gray conductors compressed into fixed pieces with latent endpoints (`E/agent.log:1651`);
- explicit latent-terminal storage deleted;
- hidden terminals replaced by “visible red cells, or degree-1 endpoints of the body” (`E/agent.log:1698`);
- additional state and rendering distinctions removed while verification stayed green (`E/agent.log:1737`).

That compression was not absurd. It fit the observed trajectories. But it was underidentified: selected-only views do not reveal every hidden terminal, so geometry-derived endpoints were not uniquely established by replay.

The most damaging deletion was therefore semantic, not filesystem-level: **explicit terminal identity was replaced by inferred wire topology**. Attempt 6 could reproduce a closed-looking selected view while still leaving hidden endpoints wrong. Attempts 9–11 rediscovered this through alternate views, but only after further full-bar executions.

### 3. Green verification certified replay, not task competence

EWMA-SV invoked its two verifier scripts 191 times; 155 exited zero. Of the main-planner invocations, 68 explicitly said:

> `No completed level-3 attempt found, we skip this level`

Examples appear at `E/agent.log:1644-1647`, `1680-1683`, and `1720-1723`.

Meanwhile `verify_world_model.py` could report “level 3 verified” because it postdicts recorded actions from recorded initial frames and checks frame/status equality. When the engine intentionally sets level-3 completion to `None`, replaying `RUNNING` or `GAME_OVER` is sufficient (`E/agent_run/world_model_engine.py:48-50,305-308`; `E/agent_run/verify_world_model.py:72-115`).

So the green checks established:

- implementation consistency;
- exact replay on the observed support;
- no regression on solved levels 1–2.

They did **not** establish:

- correct hidden endpoints;
- correct counterfactual outcomes for unseen assemblies;
- a level-3 completion rule;
- a planner capable of solving level 3.

The verification system did reject real rendering mistakes. But once completion was declared unknown and planner verification skipped the unsolved level, it could no longer veto strategically useless models. The checks became regression guards around the known prefix, not discriminators among competing success hypotheses.

They were also too late to govern some live batches. In attempt 3, the first replay mismatch was at local step 4/global 350, but another 121 actions executed before correction. In attempt 6, a mismatch at local step 68/global 792 was followed by 57 more live actions (`E/agent.log:2476-2522,3795-3802`). Exact post-hoc replay is not online protection when the execution path bypasses per-step comparison.

### 4. Exact-deadline execution amplified each model error

EWMA-SV used 887 click actions. In level 3, 832 actions were explicit timer padding in two recognizable patterns:

- 122 repeated clicks on the active source during attempt 3;
- 710 background clicks at `(63,63)` during attempts 5–12.

That is 55.467% of the entire 1,500-action level-3 budget. Across the run, 713 actions produced no frame change. Padding was rational conditional on each hypothesis: many hypotheses required holding a candidate geometry until the bar checkpoint. But it made weak hypotheses expensive. The scaffold lacked a rule like “do not spend a full deadline on a hypothesis while a cheaper ontology-discriminating probe remains.”

Attempt 12 finally began an explicit multi-view check, selecting each wire to validate hidden endpoints before commitment. The global level cap stopped the run at local step 114, before the hypothesis reached its checkpoint (`E/agent_run/level_3_reasoning_log.md:61-64`). It may have been closer, but the trace does not establish that it would have solved.

## Additions and deletions: what helped and what hurt

### Environment and observation

| change | latent or visual? | effect |
|---|---|---|
| selecting a gray wire makes its body bright and red endpoints appear | visual reveal; the wire already existed | high-information addition; TWMA used it at step 6, EWMA-SV at charged step 509 |
| deselecting a wire hides its endpoints | visual deletion, not object deletion | created partial observability; encouraged EWMA-SV's wrong geometry-derived endpoint compression |
| moving/rotating pieces removes cells at old coordinates and adds them at new ones | latent state transition plus visual relocation | both arms modeled this well on early levels |
| special `ACTION5` growth on levels 5–6 | real geometry/state addition | TWMA expanded its state model and solved later levels |
| reset | deletes accumulated board state and restores initial state | useful when model learning survives reset; TWMA level 6 is the positive case |
| timer padding | mostly adds bar pixels while preserving board geometry | lets a candidate reach the checkpoint, but turns every weak hypothesis into a 125-action test |

### Model and artifact

| metric | TWMA | EWMA-SV |
|---|---:|---:|
| model-file change events | 40 | 392 |
| add/update/delete events | 17 / 23 / 0 | 40 / 344 / 8 |
| touched surviving artifacts | 13 | 17 |
| surviving lines in touched artifacts | 558 | 1,773 |
| Git line additions/deletions in touched artifacts | 634 / 111 | 4,071 / 2,463 |
| input/output tokens | 60,818,513 / 486,945 | 240,920,931 / 1,223,147 |

EWMA-SV produced 8.77 times more model-artifact line churn and 3.96 times the input tokens. Churn is not itself a cause, but its timing matters: repeated simplification/verification cycles occurred while the decisive object-affordance question remained untested.

Some deletions helped: removing disproven source-toggle, fixed-wire, and triple-overlap rules narrowed the problem. Others hurt: deleting explicit latent-contact distinctions in favor of a topology-derived default made the model simpler but less decision-sufficient under partial observability.

## Causal assessment

### Observations

- Same game, nominal model, effort, and run index.
- EWMA-SV wins levels 1–2 more efficiently.
- TWMA probes hidden-piece selectability at level-3 step 6 and solves at step 31.
- EWMA-SV performs the equivalent hidden-piece probe at reconstructed level-3 action 509.
- EWMA-SV then spends 832 level-3 actions on explicit checkpoint padding.
- Its world-model verifier can pass level 3 while completion is intentionally undefined, and planner verification skips level 3.
- EWMA-SV performs much more model/artifact churn.

### Best-supported interpretation

The executable scaffold helped when the object inventory and transition rules were already identified. It hurt at the partial-observability shift because it encouraged **premature artifact closure**: the model committed a wrong object inventory, simplified it, and verified replay before asking the cheap question that would falsify the inventory. After correction, replay verification still lacked authority over the unsolved completion condition, while exact-deadline tests made every candidate costly.

### What this trace does not establish

- It does not prove executable world models are generally worse.
- It does not isolate code generation, verification, supervisor prompts, and token overhead from one another; they are bundled in the EWMA-SV treatment.
- TWMA also used offline Python search. The contrast is not “code versus no code”; it is maintained executable model plus required verification/simplification workflow versus a lighter textual model with disposable search.
- TWMA was not action-disciplined throughout: it burned 80 level-6 actions and emitted three post-terminal calls. Its advantage here is faster ontology correction and retained knowledge across reset, not clean execution.
- It does not prove EWMA-SV would never solve level 3 with more than 1,500 actions.
- The textual successful trace is used only as a matched comparator and falsifier of “level 3 was intrinsically too hard,” not as information available prospectively to EWMA-SV.

## Bottom line

The decisive performance effect in `cn04` was not that EWMA-SV could not write a simulator. It wrote and repeatedly verified one. The problem was that **replay correctness became a substitute for information acquisition**.

TWMA spent two actions revealing the hidden pieces, retained their explicit connectors, and searched the right state space. EWMA-SV spent four full attempts optimizing a wrong state space, then compressed hidden connector identity into inferred geometry and used most of the remaining budget to run exact-deadline hypothesis tests. Verification prevented implementation drift but could not reject a model that deliberately had no level-3 win rule.

The deployment correction suggested by this trace is concrete: before accepting or simplifying an executable model, require a **decision-sufficiency gate**:

1. list unresolved object/affordance hypotheses;
2. choose the cheapest action that separates them;
3. forbid full-deadline execution while a cheaper discriminator remains;
4. require unsolved-level verification to fail closed—no green planner result when the level is skipped or the completion predicate is undefined;
5. preserve explicit observed latent distinctions until counterfactual tests justify deleting them.

That mechanism is the artifact this audit points to. `g50t`, where EWMA-SV wins, should be audited next to test whether the same gate would preserve its advantages rather than merely biasing the process toward TWMA behavior.

Related research note: [`cn04-echo-chamber-and-frustration-note.md`](./cn04-echo-chamber-and-frustration-note.md).

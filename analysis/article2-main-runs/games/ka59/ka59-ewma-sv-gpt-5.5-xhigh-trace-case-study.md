# KA59 trace case study: GPT-5.5 xhigh with executable modeling and verification

## Decision map

**Artifact.** A turn-level and action-level reconstruction of one complete ARC-AGI-3 run:

- game: `ka59-38d34dbb`
- archive directory: `ewma_v1.2_sv_gpt-5.5_xhigh_run1`
- model: GPT-5.5, xhigh reasoning effort
- harness: executable world model, simplification, and verification (`ewma_v1.2_sv`)

**Objective.** Determine what the model constructed, which constructions survived real evidence, which failed, and how the resulting exploration affected Relative Human Action Efficiency (RHAE).

**Direct evidence.** The objective session metadata and frames, the complete 4,337-line `agent.log`, 23 model API turns, 1,092 archived model messages, 258 file-change events, 285 verifier invocations, five git checkpoints, final generated source, and official human baselines.

**Verification route.** Actions are keyed by objective global `step_index`. Every action is linked to the model turn and command that issued it. The final archived executable model was independently replayed over the entire trajectory: all 390 comparable nonterminal settled frames matched exactly, and the model predicted all seven level-completing transitions. The official RHAE values come from the separately verified archive-wide reconstruction.

## Bottom line

This run did not succeed because the model guessed a static solution. It incrementally constructed an object-centric simulator, frame parser/renderer, replay verifier, and planner. Three structures did most of the useful work:

1. **footprint-based pieces and targets**, generalized from 1x1 sprites to rectangles and arbitrary plus shapes;
2. **one shared forced-movement mechanism**, reused for shoves, field slides, and eventually phase waves;
3. **closed-loop execution**, where plans were simulated, sent in bounded batches, replayed against real frames, and repaired at the first contradiction.

The same process also built and discarded consequential errors:

- `5` as a finish tile;
- a fixed five-tile shove with no continued field slide;
- a destructible phase source, an unrecognized second emitter, and an incorrect target slot;
- separate, over-specific phase-source/indicator ontologies;
- a phase wave that ignored blocked landings;
- selectable solid pieces;
- a supposedly generic planner that still needed a hand-written level-7 staged plan.

The run completed all seven levels in one attempt each, with no reset or `GAME_OVER`, using 397 actions. Its official environment RHAE is **100%**, but that aggregate hides poor acquisition efficiency on levels 1 and 4. Before the official environment cap, its weighted level score is 105.896%; later 115%-capped levels offset the early losses, and the completed-environment cap reduces the reported score to 100%.

## Trace inventory

| Evidence layer | Count | What it establishes |
|---|---:|---|
| Model API turns | 23 | The large continuation-level units delimited by `turn.started` / `turn.completed` |
| Archived model messages | 1,092 | Every stated observation, hypothesis, implementation intent, and execution intent |
| Command executions | 1,275 | Local analysis, planners, verifiers, and real-client commands |
| File-change events | 258 | The construction and repeated revision of executable artifacts |
| Verifier invocations | 285 | 213 world-model and 72 main-planner checks, including per-action checks inside execution wrappers |
| Real environment actions | 397 | The complete objective action trajectory |
| Attempt directories | 7 | Exactly one attempt for each of seven levels |
| Objective terminal states | 396 `NOT_FINISHED`, 1 `WIN` | No reset and no `GAME_OVER` occurred |
| Intermediate animation frames | 306 across 65 actions | Extra evidence for slide and phase-wave mechanics |

Only turns **1, 3, 6, 8, 13, 18, and 23** spent environment actions. Each corresponds to one level. The other sixteen turns used no benchmark actions; they were implementation, replay, simplification, and planning work.

Verifier output contains 214 passes, three intentional skips, 59 failures, and nine outputs without a terminal pass/fail marker. The mechanical failure taxonomy records 48 observed-frame mismatches, one terminal-status mismatch, three world-model implementation failures, six planner failures, and ten tool/unclassified failures. An observed-frame mismatch is deliberately not called a dynamics failure until frame inspection distinguishes dynamics from rendering.

The exhaustive derived artifacts are:

- [`data/ka59-actions.csv`](./data/ka59-actions.csv) — one row for each of 397 environment actions;
- [`data/ka59-model-messages.csv`](./data/ka59-model-messages.csv) — all 1,092 model messages;
- [`data/ka59-turns.csv`](./data/ka59-turns.csv) — one row for each of 23 model turns;
- [`data/ka59-file-changes.csv`](./data/ka59-file-changes.csv) — all 258 completed file-change events;
- [`data/ka59-verification-events.csv`](./data/ka59-verification-events.csv) — all 285 verifier invocations, with outcome and failure-class fields;
- `scripts/extract_ka59_trace.py` — reproducible extractor and final-model replay.

## Official action efficiency

| Level | Human baseline $H$ | Real actions $A$ | Level RHAE | Final simulator-verified planner length | Real minus post-hoc plan |
|---:|---:|---:|---:|---:|---:|
| 1 | 28 | 44 | 40.496% | 11 | +33 |
| 2 | 109 | 49 | 115.000% | 43 | +6 |
| 3 | 51 | 35 | 115.000% | 33 | +2 |
| 4 | 51 | 61 | 69.901% | 38 | +23 |
| 5 | 33 | 25 | 115.000% | 20 | +5 |
| 6 | 132 | 87 | 115.000% | 46 | +41 |
| 7 | 326 | 96 | 115.000% | 91 | +5 |
| **Total** | — | **397** | **100% official environment score** | **282** | **+115** |

The post-hoc planner column is diagnostic, not an alternative benchmark score. Those plans use knowledge learned from the real trajectory and were verified in the final simulator, not rerun from each real initial state. The 115-action difference combines necessary acquisition, wrong hypotheses, interrupted plans, and residual plan suboptimality.

Two levels account for all visible score loss after per-level bonuses:

- **Level 1:** broad exploration before the target-covering objective was understood; 44 actions versus $H=28$.
- **Level 4:** 37 actions followed a model missing continued field sliding; a corrected 24-action suffix then solved the already-reached state.

Levels 6 and 7 contain real wrong-model costs too, but their human baselines are large enough that the final action counts remain above the 115% cap.

For controlled context, the same GPT-5.5 xhigh model under the textual `twma_v1.2` harness also completed 7/7 but scored only **43.355% RHAE**. Its level action counts were `[12, 47, 171, 50, 50, 268, 1089]`, compared with `[44, 49, 35, 61, 25, 87, 96]` here. This is a matched observation, not a replicated causal estimate, but the divergence is concentrated in later levels where reusable executable mechanics matter most.

## What the game turned out to be

The final model represents the game as pieces with tile footprints:

- framed pieces can be selected with `ACTION6`;
- solid and phase pieces can be displaced but are not directly selectable;
- directional input moves the active piece one tile unless blocked by hard terrain, fields, or another piece;
- contacting one piece shoves it five tiles, followed by continued movement while its footprint remains on an `F` field;
- directional actions advance a shared phase;
- when a phase piece wraps, its wave applies blocker-aware forced movement to every other piece in its path;
- the level completes when every outlined target is covered by a distinct piece with the matching footprint and position.

This is materially smaller than the sequence of hypotheses used to discover it. The final model deliberately separates dynamics from rendering; the lower screen row is treated as a visual spent-action clock rather than game state.

## Every model turn

Line ranges below refer to the archived `agent.log`. Exact per-turn counts, token usage, action ranges, file changes, and verifier outcomes are in [`data/ka59-turns.csv`](./data/ka59-turns.csv).

| Turn | Log lines | Real actions | What the model did | Result and judgment |
|---:|---:|---:|---|---|
| 1 | 16–917 | 1–44 | Read the harness; parsed 3x3 tiles; initially modeled `0` as player and `5` as finish; built engine, renderer, BFS planner, auxiliary planner, reports, and exact replay checks; repeatedly revised all three core modules while exploring level 1. | **Mixed.** The finish-tile model passed a one-step replay, then action 3 falsified it. Action 6 established that `F` blocked voluntary entry, while actions 39–42 established that the hollow `4` outlines were enterable. The instrumentation and object parser survived; broad endpoint and visual-clock probing made this the least efficient acquisition phase. |
| 2 | 934–987 | — | Replaced the trajectory-specific visual-marker cases with a deterministic clock and compressed the model by 27 insertions and 84 deletions. | **Useful but action-neutral.** Replay and planner checks passed; explanatory simplicity improved, but the 44 level-1 actions were already spent. |
| 3 | 990–1365 | 45–93 | Generalized 1x1 entities to rectangular footprints and matching outlined targets; executed a planned route behind per-action replay guards. | **Worked.** Guards halted at actions 48, 58, 79, and 80 for two visual and two dynamics corrections, preventing stale suffixes from running. Level 2 still finished in 49 actions versus $H=109$. |
| 4 | 1380–1434 | — | Replaced field-specific push machinery with a fixed five-tile shove. | **Temporarily attractive, later false.** It compressed levels 1–2 evidence and passed existing checks, but level 4 later supplied the missing counterexample. |
| 5 | 1437–1496 | — | Removed more special push state and aligned auxiliary planners. | **Action-neutral compression.** It reduced code concepts but strengthened the same incomplete fixed-shove theory. |
| 6 | 1499–1842 | 94–128 | Generalized reconstruction to inferred grid origin and arbitrary shapes; represented solid plus pieces; built and executed a push-only level-3 plan. | **Strong success.** All 35 actions followed a simulated plan; no real exploratory branch was reported. Post-completion replay found only visual clock/contact mismatches, not a wrong route or transition law. |
| 7 | 1845–1965 | — | Replaced level-specific planner delegation with shared best-first macro search over footprint states; discarded slow prototypes and repaired one introduced `IndentationError`. | **Reusable success with limits.** The shared planner covered solved static levels, but the construction was not frictionless and later needed primitive phase search and a staged level-7 exception. |
| 8 | 1968–2138 | 129–189 | Ran a 37-action level-4 plan under the fixed-shove model; replay found a mismatch at local step 18; inferred continued sliding while touching `F`; executed a corrected 24-action suffix. | **Important failure and repair.** Nineteen actions ran after the hidden divergence. The dynamics change also made generic level-2 search too slow, temporarily forcing an auxiliary fallback; later planner work removed it. Corrected forced movement survived. |
| 9 | 2153–2266 | — | Collapsed shove behavior into “five-tile impulse, then continue while on field”; tested whether every piece was selectable. | **Good simplification and restraint.** All-selectable fit prior frames but would make untested solids actionable, so it was rejected before real execution; `touched_field` was safely removed. |
| 10 | 2269–2384 | — | Removed `activated`, stored rectangle size, and duplicated geometry; derived behavior from shape and stored value. | **Useful compression.** Reduced representational drift without real actions. |
| 11 | 2387–2417 | — | Rewrote the prose model around footprint coverage and shared movement rules. | **Documentation alignment.** No new environmental evidence. |
| 12 | 2420–2563 | — | Removed dynamic level-specific planner fallback; after several high-branching waypoint designs stalled, kept only the best reachable progress position. | **Worked on existing evidence.** One shared best-first planner then solved levels 1–4 without fallback, although the later level-7 script shows that generality remained bounded. |
| 13 | 2566–2813 | 190–214 | Investigated `C`/`D` patterns and built phase-aware engine, parser, renderer, planner, and reports. | **Efficient but failure-rich.** Replay disproved “source destruction” at action 198; a 12-action plan (199–210) exposed the target-side `DCC` as a second emitter; action 211 disproved completion at `(15,9)`; the corrected three-action suffix completed. The level still used only 25 actions versus $H=33$. |
| 14 | 2828–3015 | — | Replaced “large phase source plus special indicator” with parameterized phase emitters. | **Correct direction.** One shared phase concept replaced level-specific visual roles. |
| 15 | 3018–3149 | — | Derived period, direction, wave distance, and pixel size instead of storing redundant phase fields. | **Useful compression**, although later all-`D` direction inference remained heuristic. |
| 16 | 3152–3231 | — | Unified phase emitters and ordinary pieces under one `entities` ontology. | **Durable success.** This made shoves and waves operate on the same footprints. |
| 17 | 3234–3306 | — | Consolidated planner loops while retaining macro successors for static levels and primitive successors for phase levels. | **Practical compromise.** Full primitive search was too expensive for every level. |
| 18 | 3309–3706 | 215–301 | Solved level 6 under a sequence of phase-wave hypotheses. Prior replay rejected an `F`-passable 45-action route before any real action; actions 215–220 probed orientation/waves; a planned experiment at 221–232 falsified voluntary travel while already on `F`; local step 54 then showed a wall-blocked wave landing. | **Mixed but disciplined.** One false route cost no actions because replay caught it; two prospective theories did cost actions. The final blocker-aware forced-movement repair survived level 7, and $H=132$ kept the level at 115%. |
| 19 | 3721–3832 | — | Tried and rejected equating wave reach with shove distance; then removed `target_counted`, replaced phase axis/fill fields with `phase_dir`, and unified forced landing. | **Useful conceptual cleanup.** Replay rejected the false simplification with no action cost. |
| 20 | 3835–3917 | — | Tested removing solid/phase appearance categories; replay showed that this broke level-3 reconstruction and would expose solids to selection, so solid appearance was restored. | **Mostly maintenance, with a useful negative test.** It passed replay after correction but added substantial token/tool cost outside RHAE. |
| 21 | 3920–3955 | — | Reframed shove and wave as the same forced-movement family. | **Strong abstraction.** This framing directly enabled the successful all-piece wave correction in level 7. |
| 22 | 3958–4005 | — | Simplified best-first search and delegated completion to the engine. | **Good interface cleanup**, but did not make level-7 search tractable by itself. |
| 23 | 4008–4334 | 302–397 | Explored level 7 in the model; generalized phase waves to all pieces; temporarily made solid pieces selectable; executed 38 verified setup actions; tested the solid-selection branch for six actions; reverted it after replay mismatch; found a 52-action corrected suffix; added a 91-action staged initial-state plan; ran final verifiers. | **Best example of both success and failure.** Segment verification contained the false branch to six actions. The all-piece wave rule worked; solid selection did not. Generic search gave way to a level-specific staged plan. |

## Every environment action

[`data/ka59-actions.csv`](./data/ka59-actions.csv) is the canonical one-row-per-action record. Each row contains:

- objective global step, level, attempt, and level-local action index;
- exact action and coordinates;
- observed state, completion transition, frame differences, and animation-frame count;
- issuing model turn, `agent.log` line, direct/batch execution mode, and preceding model message;
- audit-classified action intent and any concrete evidentiary outcome;
- final-model predicted status and exact settled-frame comparison;
- active entity and position before/after;
- every moved entity, collision candidate, phase before/after, and target coverage;
- a semantic effect label derived from the final verified model.

The semantic labels cover all 397 actions:

| Final-model interpretation | Count |
|---|---:|
| Active one-tile movement, including seven completing moves | 329 |
| Shove, including one shove coincident with a phase wave | 23 |
| Successful coordinate selection | 20 |
| Coordinate no-op | 2 |
| Phase wave moved the active piece more than one tile | 8 |
| Directional input advanced phase without changing a position | 10 |
| Directional no-op with no phase state | 5 |
| **Total** | **397** |

These labels are **interpretations**, not raw metadata fields. Their warrant is unusually strong for an inferred model: its renderer matched all 390 nonterminal settled frames exactly and its engine predicted all seven terminal transitions. That still does not prove its behavior on unobserved counterfactual actions.

Intent is kept separate from outcome. It is classified from the nearest preceding model message and, for batched execution, the first hidden replay divergence:

| Stated/audited intent | Actions | Meaning |
|---|---:|---|
| Planned route | 195 | Executed under a current plan or explicit route |
| Planned route after correction | 104 | Replanned from the exact observed post-mismatch state |
| Planned setup for a probe | 52 | Route whose purpose was to reach a discriminating condition |
| Planned but no longer validated | 32 | Batch continuation after the first hidden divergence invalidated its original model justification |
| Exploratory probe | 14 | Immediate action chosen to distinguish mechanics, selection, passability, or termination |
| **Total** | **397** | — |

The 32 “no longer validated” actions are not labeled random or inherently wasted. They were rational under the pre-batch model, but their justification had lapsed before execution completed: 19 after the level-4 divergence, eight after the level-6 divergence, and five after the level-7 selection no-op.

### Action phases by level

#### Level 1 — discovery dominated

- **1–3:** tested movement toward `5`; action 3 shoved `5` through the field instead of completing, falsifying the finish-tile model.
- **4:** selected the moved `5`, establishing selectable entities.
- **5–16:** moved that entity through the right room and tested endpoints; action 13 was a blocked no-op.
- **17:** switched back to the original framed entity.
- **18–27:** explored lower-left endpoints and boundaries; actions 26–27 were ineffective boundary probes.
- **28–40:** switched entities again, explored the lower/right room, tested obstacles, and formed the target-outline hypothesis.
- **41–44:** selected the remaining piece and placed it on the matching outline; action 44 completed the level.

The final model’s initial-state plan uses 11 actions. The real trace used 44. The 33-action gap is the clearest acquisition cost in the run and yielded only 40.496% level RHAE.

#### Level 2 — model reuse with corrections

- **45–93:** 49 actions arranged four rectangular footprint pieces using 37 active moves, eight selections, and three shoves.
- Replay guards stopped after **48** (visual clock), **58** (long-slide endpoint), **79** (contact rendering), and **80** (vertical shove endpoint). No stale suffix continued after those disagreements.
- The durable generalization was from sprite identity to `(shape, position)` matching.
- The final shared planner uses 43 simulated actions; the real trace used six more but remained far below $H=109$.

#### Level 3 — clean planned transfer

- **94–128:** one 35-action scripted route: `R4, D8, L1, D3, L10, U6, R3`.
- It used three shoves and no selections; the framed piece pushed nonselectable solid plus pieces.
- The run’s own report records no exploration actions. The final shared planner later found a 33-action route.
- Post-completion replay did find visual clock/contact-rendering debt, but it did not contradict the successful route or transition dynamics.

This is the strongest evidence that arbitrary footprint parsing plus shared forced movement transferred successfully.

#### Level 4 — wrong plan, decisive counterexample

- **129–165:** 37 actions from the fixed-five-tile-shove model. The real level did not complete.
- **Local step 18 / global step 146:** first replay counterexample. The real plus landed two tiles farther than predicted because it remained on `F` after the five-tile impulse.
- **147–165:** 19 additional actions had already run inside the same batch after the hidden divergence; their original simulated justification was no longer valid.
- **166–189:** corrected 24-action suffix from the actual state completed the level.

The final initial-state planner uses 38 actions; the real run used 61. This 23-action post-hoc gap and the squared efficiency penalty reduced level RHAE to 69.901%.

#### Level 5 — cheap phase acquisition

- **190–198:** a nine-action controlled acquisition segment. The CSV separates the immediate probes at 190 and 198 from the seven setup moves between them. Action 190 falsified direct selection; 194 exposed a phase wave; 198 showed that the patterned 2x2 was shoved through `F`, not destroyed.
- **199–210:** a simulated 12-action route failed in reality because the target-side `DCC` was a second horizontal emitter, not a passive display.
- **211:** a one-action simulated completion reached `(15,9)` but remained nonterminal, restoring the real target to `(16,9)` and forcing phase-aware primitive planning.
- **212–214:** the corrected `up, down, right` suffix completed.

Twenty-five actions were enough to learn the phase mechanism and solve, versus $H=33$.

#### Level 6 — wave generalization and blocked landing

- Before action 215, historical replay rejected a 45-action route under the false claim that `F` allowed voluntary entry; no RHAE actions were spent on that route.
- **215–220:** a six-action orientation/wave experiment; action 220 showed an upward wave carrying the actor across `F`. The action taxonomy calls 215 the immediate probe and 216–220 planned setup for that probe.
- **221–232:** a planned prospective experiment tested whether a piece already carried onto `F` could move through it voluntarily; replay rejected that theory.
- **233–276:** plan execution under an evolving wave-landing theory. Waves carried the active entity at global steps 244 and 262.
- **Local step 54 / global step 268:** the key counterexample to broad corridor sweep. A predicted landing was behind a hard wall; the real piece stayed below the field.
- **269–276:** eight more actions completed within the already-issued batch after that hidden divergence.
- **277–301:** a 25-action corrected suffix completed the level.

The final initial-state planner uses 46 actions, whereas the real trace used 87. The 41-action difference is substantial acquisition/repair overhead, but $H=132$ means the level still receives the 115% cap.

#### Level 7 — bounded falsification

- **302–339:** a 38-action planned segment relocated the upper phase piece and used its wave to carry the solid plus through the field. Replay matched after correcting only the visual clock budget.
- **340–345:** six-action false branch. The click on the solid plus was the discriminating probe and was ignored; the five following down inputs were already batched and moved or phase-advanced the still-active framed piece after the plan had diverged. This falsified solid selectability.
- **346–397:** a 52-action corrected suffix used framed pieces to shove the plus to its target and then placed the two dominoes. Action 397 produced `WIN`.

The clean final staged plan is 91 simulated actions. The real trace used 96. Segmenting the experiment prevented the false selectability hypothesis from contaminating the remaining 52-action suffix.

## What the model built that worked

### 1. Object-centric footprint reconstruction

`world_model_state_io.py` infers grid origin, extracts framed and solid connected components, converts pixels to tile footprints, finds outlined targets, and reconstructs phase pieces. This representation generalized in stages:

- 1x1 framed pieces in level 1;
- rectangles in level 2;
- shifted grid origin and arbitrary plus shapes in level 3;
- phase pieces in levels 5–7.

The practical success signal is level 3: the model took no reported exploration branch and executed a complete 35-action plan immediately.

### 2. Shared forced movement

The final engine separates voluntary movement from forced movement. Shoves and phase waves both use blocker-aware landing and field-slide logic. This abstraction was not present initially; it emerged from two failures:

- continued field slide in level 4;
- blocked wave landing in level 6.

Once corrected, it transferred to level 7, where a phase wave moved the solid plus through a field even though the plus was neither active nor selectable.

### 3. Exact replay as contradiction detector

The verifier did more than produce a pass/fail badge. It localized the first mismatch and archived simulated frames, real frames, and mismatch images. Important corrections followed concrete replay failures:

- level 1 adjacency rendering and later the false finish-tile dynamics;
- level 4 continued field sliding;
- level 6 wall-blocked wave landing;
- level 7 visual clock versus dynamics;
- level 7 false solid selection.

The most important behavior was separating visual mismatches from dynamic mismatches. At level 7 step 1, replay differed by one bottom-row pixel; the model correctly repaired the clock instead of changing movement rules.

The two verifier types support different claims. `verify_world_model.py` reconstructs recorded attempts and compares observed status plus every nonterminal rendered frame; its final pass supports in-sample trajectory fidelity. `verify_main_planner.py` generates a plan and simulates it in that same learned engine; it supports self-model solvability, not real execution of the post-hoc plans.

### 4. Planner-engine coupling and bounded execution

The planner generates successors through the same engine used for replay. Real execution shifted from 44 individually issued level-1 actions to scripted batches:

- 56 actions were issued directly;
- 341 were issued in scripted batches.

Batches included stop conditions for terminal states and were followed by replay checks. This reduced conversational overhead while retaining contradiction checkpoints.

### 5. Explicit falsification and reversion

The solid-selectability episode is a good recovery pattern:

1. add the simpler candidate rule to engine, planner, and model notes;
2. verify that prior traces remain compatible;
3. execute a small discriminating real segment;
4. observe that the click was ignored;
5. revert all three artifacts;
6. replay the now-recorded counterexample;
7. replan from the exact current state.

Prior replay could not reject the hypothesis because earlier traces never clicked a solid piece. The real action was the unique discriminating test.

## What failed or did not generalize

### 1. A weakly verified finish-tile model

After one right move, the first renderer and BFS passed replay and predicted two more right moves to completion. Action 3 showed that `5` was shoved, not entered as a goal. The early verifier proved only one-step visual consistency, not the terminal mechanism. This is the clearest example of a polished, executable model passing a weak test.

### 2. Fixed five-tile shove

Turns 4–5 simplified all prior evidence to a fixed five-tile shove. That theory passed levels 1–3 but failed in level 4 where the piece remained on a field after the impulse. The correction—continue until clear of `F`—was durable, but the failed model consumed 37 real level-4 actions before the level was recognized as incomplete.

### 3. Destructible source, missed emitter, and over-specific phase ontology

The first level-5 implementation claimed the large pattern was a source that disappeared on contact. Replay at local step 9 showed it had instead been shoved to the top. The next planner treated the target-side `DCC` as a passive indicator, so its 12-action route was pushed back by an unmodeled second emitter. A third model predicted completion one tile early. Turns 14–16 then deleted the source/indicator distinction and represented all such components as entities with optional phase behavior. The initial ontology encoded visual roles, not the reusable mechanic.

### 4. Broad wave landing that ignored walls

Early level-6 evidence showed successful cross-field carries. The model overgeneralized this into a corridor sweep and predicted a landing behind a wall. Local step 54 disproved it. The repaired mechanism delegated waves to the same blocker-aware forced landing used by shoves.

### 5. Solid-piece selection

The level-7 search became easy if the solid plus could be selected. The hypothesis was added to code and passed all prior replay because no earlier action tested it. Real global step 340 was a no-op. Five subsequent actions operated on the still-active framed piece. The branch cost six actions and had to be removed from engine, planner, and documentation.

### 6. Generic planner ceiling

The planner was repeatedly simplified into one best-first framework, but exact auxiliary programs remained for levels 2 and 3, and level 7 still required `_level_7_staged_plan`: a hard-coded sequence of phases, literal actions, and entity indices duplicated in `level_7_planner_1.py`. Levels 4–6 used the shared planner. The scripts solve and verify archived layouts; they are not evidence of a reusable general solver. The reusable lesson is hierarchical waypoint or subgoal search, not the literal programs.

### 7. Exact visual verification spent effort on a non-causal clock

The bottom row is a spent-action display, not game dynamics. The first level-1 engine overfit it with branches on entity values, positions, directions, and marker thresholds; turn 2 deleted most of those cases. Exact frame replay still required per-level clock budgets in `LEVEL_STEP_BUDGETS`. This did not consume extra environment actions by itself, but it consumed model turns, edits, verifier cycles, and context. A production runtime should compare causal state separately from presentation pixels while retaining a diagnostic visual channel.

### 8. Remaining brittle reconstruction heuristics

The final phase-direction inference for an all-`D` component falls back to which half of the board contains it. It fits this trace but is not strongly identified by the observed pixels. That heuristic should be treated as a hypothesis requiring active discrimination, not a generally valid parser rule.

## Artifact evolution

The agent checkpointed after selected level transitions. Core source size grew despite repeated simplification passes:

| Snapshot | Engine | State I/O | Main planner | Model notes |
|---|---:|---:|---:|---:|
| After level 1 (`1619a7a`) | 182 | 191 | 57 | 11 |
| After level 2 (`7f5a172`) | 180 | 274 | 70 | 15 |
| After level 4 (`67d6556`) | 169 | 438 | 227 | 25 |
| After level 5 (`17a57f7`) | 228 | 560 | 278 | 29 |
| After level 6 (`4af6406`) | 281 | 588 | 251 | 29 |
| Final working tree | 294 | 591 | 333 | 29 |

The final 1,218 lines of core executable code are not a tiny solver. Most complexity lies in parsing/rendering and search, and 54 lines of the final planner are a staged level-7 exception. The stable extractable kernel is smaller:

- footprint/entity state;
- target signatures;
- voluntary versus forced movement;
- blocker-aware field landing;
- shared phase clock and wave trigger;
- exact replay mismatch reporting;
- generic state keys and successor generation.

Checkpoint history makes the boundary concrete: levels 2, 3, and 7 retain exact auxiliary programs, while levels 4–6 do not. Conversely, the final renderer's `apply_render_overrides` is a no-op, so the final 390/390 replay fit did not rely on per-recorded-frame pixel patches.

## Compute outside RHAE

The run consumed:

- 156,778,338 input tokens;
- 152,674,304 cached input tokens;
- 608,563 output tokens;
- 259,473 reasoning-output tokens.

The sixteen turns with no environment action still consumed 50,770,258 input tokens, 179,922 output tokens, 95 file-change events, and 81 verifier invocations. Official RHAE intentionally prices environment actions, not this local compute. For a smaller local solver, these numbers are a deployment problem even though they do not reduce the benchmark score.

## What can be concluded

### Directly observed

- All seven levels completed in one attempt each with 397 actions and no reset or `GAME_OVER`.
- The action sequence, frames, model turns, file-change events, and verifier outputs are archived.
- The final model exactly replays every comparable observed settled frame and all level transitions.
- Levels 1 and 4 were inefficient relative to human baselines; levels 2, 3, 5, 6, and 7 received the 115% level cap.
- The official environment score is capped at 100%.

### Strong interpretation

- Executable state and replay made contradictions local and repairable.
- Reusing footprint and forced-movement abstractions materially reduced later-level action counts.
- Segment-level verification limited the cost of the level-7 solid-selection error.
- The same-model textual run’s later-level action explosion is consistent with executable artifacts improving persistent causal state and planning.

### Not established

- Exact replay does not prove the model on unobserved actions.
- Simulator verification of the post-hoc 282-action clean plan does not prove that exact plan would execute identically in a fresh real run.
- One matched run per harness does not identify a replicated causal harness effect.
- The final parser heuristics and staged level-7 plan do not yet constitute a game-general local runtime.

## Next falsifying test

The highest-value next test is to take the extracted stable kernel—without per-level clock budgets, auxiliary scripts, or `_level_7_staged_plan`—and run it on a held-out `ka59` trace prefix or another game with footprint/phase mechanics. Freeze the model before the suffix, ask it to predict held-out transitions and produce a plan, then compare:

1. exact causal-state prediction, excluding presentation-only clock pixels;
2. real action count and RHAE;
3. whether hierarchical subgoals can replace the level-7 script;
4. whether the all-`D` phase-direction heuristic survives an active discriminating probe.

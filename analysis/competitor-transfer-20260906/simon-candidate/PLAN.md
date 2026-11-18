# Revised solver candidate: measurements, experiments and bounded execution

Status: design and local source qualification only. This document supersedes the implementation order in ../CURRENT-STATE-UPDATE.md, not its historical run evidence. No solver payload changed and no model or GPU run was performed. The separate serving investigation remains independent.

## Judgment

Bring exact scene measurement and one bounded planner ahead of automatic repair generation. Connect them through an actor-selected pre-completion experiment. Retain regression replay as protection for demonstrated behavior. These are bets against observed failure patterns, not an established explanation of the zero-level result.

## Verified starting point

Base: deployment/kaggle/q6-summary-boundary-20260906, contract 24c291bc6956a96dce361a52a713c0b3c4e4208da7e464d735d2b310834976a9. reproduce.py verifies every runtime file against CHASSIS.json. Do not edit the archived base; create separately identified comparison packages after implementation scope is selected.

The actor currently has grid_array(), latest_state(), trajectory(), prediction_feedback(), mutable helper files and submit_actions(). latest_state() already attaches prediction feedback, while format_game_context() renders status plus separate helper/stall/knowledge/scribe lines. There is no wired segmenter or bounded planner. The worker queues a submitted batch; observe() clears its suffix only when the existing evidence path requests a stop. Actor-side feedback is not itself a synchronous worker interruption.

The archived segmenter preserves exact geometry via row runs and optional cells; its bbox and summary agree in a deliberately off-diagonal fixture. It defaults to majority-color background and fixed single-color four-connectivity. Those defaults need explicit presentation and alternative grouping in the new interface.

The archived BFS has a concrete semantic defect for this proposal: it catches transition exceptions, skips those edges, and can then return no_plan_exhausted. reproduce.py demonstrates this using the actual extracted definition with a failing transition stub. It has a node cap but does not establish a hard bound on callback execution. Do not port its status semantics unchanged.

The existing dynamics.key_counterexample() explores a candidate model from two key-equal states. It does not search observed histories for same-key, same-action, different-outcome examples. Keep those operations distinguishable.

## Implementation sequence and ownership

| Slice | Change and placement | Keep or remove | Local acceptance |
|---|---|---|---|
| R: repaired baseline | Wire effective request configuration at worker/client/final-request boundaries; integrate manifest validator; add per-action committed-check interruption through the solver computation boundary | Keep one actor, fixed checkpoint, summarization, notes and exploratory single actions. Preserve original run status. | Actual child environment and assembled request prove limits; retries counted; saved validator replay; mismatch after action one cancels suffix before action two. Checker error is separate. |
| M: measurements and packet | New pure-compute scene module; thin discoverable CodeAct methods; one packet assembled from current state, existing records and exact scene inventory | Replace duplicate context rendering after field-by-field parity check. Keep full grid/history access; no semantic object labels assigned by host. | Awkward mask/geometry cases plus actual packaged CodeAct call, intelligible result and accepted action; static objects remain visible. |
| E: experimental planning | One bounded BFS service plus selected-experiment record and short operational skill | Replace repeated generated queue/visited/predecessor code when the actor selects search. Remove mandatory repair wording; do not require an experiment every turn. | Explicit result classes, callback kill/reap, unknown edge semantics, reached/unreached/unknown condition distinctions and pre-completion updates. |
| P: retention | Old-versus-proposed helper replay and observed-history collision query | Retain incumbent identity and observable checks. Defer automatic extra candidate generation and persistent ensembles. | New regression, repaired case, remaining mismatch, checker error and changed measurement are distinct; no later observations leak into decision inputs. |

The repair baseline is behavior-changing: do not claim its results are directly comparable with the historical 107615-token trajectory. Establish declared request, action, wall-time and sampling limits for all subsequent arms. Full-context enforcement must account for template overhead and output reserve, with compression/rejection before dispatch; setting a truncation field on system blocks is insufficient.

## Proposed actor surface

Names below are proposed interfaces, not currently callable methods.

- scene(grid, background, connectivity, grouping): return settings, frame identity, frame-local component IDs, exact colored local masks, origin, bbox, area and adjacency. Background may explicitly be absent; majority-color selection is an opt-in hypothesis. Grouping permits single-color components and explicitly selected composites. Pixel coordinates remain the default; no inferred tile scale.
- compare_patterns(left, right, transforms): enumerate compatible translations/rotations/reflections and residual mismatches. Preserve colors unless a requested comparison explicitly ignores them. Canonical family identity is separate from orientation-sensitive representation.
- decision_packet(): return current observation identity and budgets, scene index, latest delta, last executed result, committed-check status/version, active helper hash, and optional selected experiment. Facts and actor assertions occupy distinct fields. Exact masks/grids remain retrievable; any truncated inventory reports omissions. Cached views are keyed by observation and measurement settings and invalidated on reset or changed settings.
- plan_bfs(start, actions, successor, key, target, limits): shortest unit-cost modeled plan with node, depth, frontier-memory and elapsed-time limits. Return found, budget_exhausted, unknown_dynamics, space_exhausted or callback_error plus counts and provenance. Unknown successors must not become blocked cells. A found path can still be returned if its own edges are known, while unexplored/unknown branches remain reported. Depth and action enumeration caps mean budget exhaustion, not proof of impossibility.
- select_experiment(record): one optional selected claim, its scope, measurement/version, expected observable outcome and timing, proposed plan/action, and evidence references. Completing an experiment does not require winning a level.

Arbitrary Python callbacks require a killable solver-side subprocess with explicit serialization of the model/helper snapshot, start state and arguments. Merely adding a timeout between node expansions or relying on the in-process CodeAct cell timeout is inadequate. If a callback cannot be serialized, return an interface error rather than silently run without a hard bound. Resolve the transport against the actual packaged runtime before exposing the planner.

Do not import actor-authored callbacks into the official game process. Per-action checking needs a bounded request to the solver-side evaluator before another queued action is released; timeout/error cancels that committed suffix and returns the observation without asserting a model contradiction. Record the extra checking latency.

## Pre-completion experiment outcomes

A claim such as "matching A to B is sufficient for completion" requires a measured test condition and a declared outcome window. If the match was not reached, the objective remains untested. If the measurement is unavailable, report unknown. If the match was established and the declared window elapsed without completion, record a counterexample to that scoped sufficiency claim. Completion provides supporting evidence, not proof of necessity or universal sufficiency. Reset and level boundaries terminate or explicitly revise the experiment scope.

Measurements and checkers are versioned with the commitment. Editing an extractor does not retroactively change the meaning of a prior check. A constant encoder and a background-only predictor must report limited/vacuous evidence rather than certify a plan. One exploratory action remains available when a useful commitment cannot be made.

## Retention without automatic repair generation

Replay identical selected raw transitions through the incumbent and proposed helper and common observable checks. Return repaired cases, newly broken cases, unresolved cases, errors and first counterexample. Preserve both hashes. An actor can reject an edit or take a fresh exploratory action without fitting every historical case.

The observed state-key query groups aligned records by declared key, action parameters and scope, retaining reset/level/history provenance. Differing checked consequences expose a representation, alignment or determinism problem; they do not uniquely prove which hidden variable is absent. Unknown checks are not contradictions. Model-generated distinguishing suffixes remain a separate optional diagnostic.

## Comparisons and acceptance

Use R as the repaired baseline, R+M for the measurement/packet treatment, and R+M+E for the experimental-planning path. Keep checkpoint, sampling, full-request limits, action and wall-time budgets equal. Add replay retention after the core path is locally callable, before automatic repair generation. No requirement that each complementary component independently win a level.

First qualify the actual packaged CodeAct surface with a scripted model endpoint: method discovery/signature, call, structured result, continued actor execution and accepted action. This proves wiring and lifecycle only. It cannot establish that Qwen discovers or chooses to use the operation.

Then compare prospective game behavior under those declared limits: completion first, followed by correctly selected objects, achieved test conditions, appropriate belief revision and calls spent reconstructing state. Use a small held-out set of decision cases with later observations hidden; keep intervention-authoring cases separate. More hypotheses, more passing helper tests, and noncompletion predictions are not success criteria.

Required fixtures include distant same-color objects; adjacent multicolor composites; same-area/same-bbox different masks; asymmetric rotations and symmetric ambiguity; hanging callbacks; unknown transitions; same-looking history-dependent states; failed-to-reach versus falsified objectives; static target outside the delta; changed extractor interpretation; and mismatch on the first action of a batch.

## Deferred or separate

Automatic multi-repair generation, A*, a new builder, latent bridges, production weight training and a new memory subsystem remain deferred. Rich visual input deserves an early separate representation comparison. HUD changes are reported separately without silently erasing them or equating unchanged pixels with unchanged hidden state. MTP, cache behavior and concurrent serving remain on the independent performance line; actual local HTTP overlap has been demonstrated, GPU capacity has not.

## Evidence and limitations

Sources inspected: archived tycho/wmlib_template.py:110-205 and 1543-1590; tycho_nooa/dynamics_template.py:41 onward; packaged chassis/solver_agent.py:981-1059; chassis/arc_llm.py:53 onward; nooa_worker.py:183-235; CURRENT-STATE-UPDATE.md; shared board threads 305 and 314.

Local evidence: EVIDENCE.json and reproduce.py. These checks do not qualify an implemented replacement; none exists yet. The earlier suspected bbox defect was rejected by execution and is not an outstanding finding. External paper and competitor claims in the supplied review were not independently reverified in this pass.

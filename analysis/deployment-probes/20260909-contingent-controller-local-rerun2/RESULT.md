# Bounded contingent controller: locally qualified

Implemented in deployment/kaggle/contingent-controller-20260909, contract `f118fc0b9dc1b86bfab41a8db3f20a9b7859d6844ab00647e8b53999383eade8`. Donor designer-player-discovery-20260908-v2 is unchanged at `cce5829f641067d4e2e7668b0c7c29443e0a5c2874d7e87b73e6964a8edd7f76`.

## Behavior

The Actor can compare or submit up to four two-decision candidates over up to four weighted hypotheses. The controller groups worlds by their predicted observable outcome and selects one common continuation per group. It ranks provisional expected completion first and estimated total action/recovery cost second. No entropy bonus, calibrated RHAE estimator, general search tree, training, or scheduler is added.

The first action and at most one observation-contingent continuation use existing submit_actions gates. The dispatcher handles the continuation without another neural call. Typed metric mismatches, stale/skipped evidence, new user guidance, protocol boundaries, unchanged grids and failed/unresolved committed-model checks return control to the Actor. Root grids, hypotheses, weights, assumptions, costs, branch choices and stop reasons remain in contingent_events.jsonl; the normal decision packet exposes the latest event. The observation metric establishes a match only within its scope, not truth of the hypothesis or all prose assumptions.

## Validation

Full successor suite: **111 passed in 56.23 seconds**, including real subprocess lifecycle and notebook-extracted package checks. See pytest.log and junit.xml. Freeze verification and notebook payload hash checks passed. No external model calls or Kaggle launch; launch_ready=false.

The new actual adapter → LocalSolver → host → headless → CodeAct cases used a deterministic local HTTP model:

| Case | Actual actions | Model calls |
| --- | --- | ---: |
| Ordinary scripted CodeAct | UP, LEFT | 2 |
| Controller: observed branch A | UP, LEFT | 1 |
| Controller: observed branch B | UP, RIGHT | 1 |
| Controller: no predicted branch | UP, DOWN after Actor handoff | 2 |

Per-case RESULT.json files, requests/responses, accepted actions, transitions and controller events are under pytest/raw/test_actual_contingent_worker_0 through _3. Every case has two accepted actions. This is evidence of eliminating a model crossing under a supplied plan, not measured Qwen throughput or a model-quality comparison.

The finite-world diagnostic in ../20260909-contingent-controller-local/worlds executed independent primitive transitions against supplied hypotheses:

| Case | Fixed commitment | Controller |
| --- | --- | --- |
| Relevant probe | Both worlds completed, mean 4 actions | Both completed, 2 actions |
| Same route optimal in both worlds | Both completed, 2 actions | Skipped the probe, both completed, 2 actions |
| Progress also teaches | Both completed, mean 4 actions | Both completed, 2 actions |
| Every prediction wrong | Fixed route happened to complete | Stopped after one action in both worlds |
| Predicted goal event absent | Fixed route happened to complete | Stopped after one action in both worlds |

Unknown/false-goal cases test conservative handoff, not improved completion. The finite-world harness ends at handoff; it does not test recovery by a neural agent. These are action counts, not RHAE; the fixed-commit policy is not the full Qwen baseline. No aggregate superiority claim follows.

## Failure encountered and repaired

The first aggregate run failed fixture setup because the pytest output parent directory did not exist; that run is retained in ../20260909-contingent-controller-local. The next run passed 109 tests and failed one real integration assertion. Its raw second request exposed the inherited WAIT validator reading the newly published successor turn and falsely rejecting the previous turn's real submission. The validation retry bypassed the dispatcher and thus the controller. The successor now binds WAIT validation to the entering dispatcher turn, and a deterministic regression reproduces the old failure while rejecting genuinely unbacked WAIT results. The final full suite passed. Earlier failed evidence is retained; no result was overwritten.

## Limits and next comparison

Weights, objective beliefs, completion estimates and costs remain model-authored and uncalibrated. A small partial observation metric can miss unmodeled changes. The interface is optional; ordinary single-action exploration is retained. The controller is bounded to one continuation and does not infer good alternatives itself. It stops at level boundaries; no cross-level option learning was tested.

Live Qwen adoption, proposal quality, matched-runtime full-game completion and RHAE remain untested. The next quality comparison should hold model, total runtime and evidence access fixed, compare the full current solver to this successor, and include failures in the denominator. This local run authorizes no score-improvement claim or production promotion.

## Reproduction

Use the existing NOOA venv and set PYTHONPATH to this checkout's solvers/tycho_nooa/src and /home/pranav/projects/arc-agi-3-reference/tycho. The exact test command, frozen identity and elapsed time are in EXECUTION.json and STARTED.json; run.py preserves the runner. Fresh output directories are required for a new screen. The implementation patch relative to the donor is in IMPLEMENTATION.patch.

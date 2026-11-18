# Deep structural self-test Kaggle run — result

## Correction after inspecting the complete IPC artifacts

The reachability interpretation below is withdrawn. `eligible_detector_attempts`
in `nooa_public_probe.py:236` counts an older gate requiring step >= 25 and six
earlier batches. It does not count the new typed self-tests. The actual IPC
contains four `decision_experiments.jsonl` entries and four corresponding
`experiment_results.jsonl` entries: two metric matches and two
`none_of_predictions` results. Structural self-testing was exercised.

The first experiment illustrates a remaining semantic weakness: the question
asks whether RIGHT moves the cross, but its predicate is any grid change.
That predicate matched although the host prediction feedback shows the cross
center stayed at [32,21]. A metric match does not establish its hypothesis.
The schema correctly calls the update `model_proposed_update`; the test itself
still measures an insufficient discriminator.

The 44 completed call records contain 354,208 completion tokens, including
322,305 reasoning tokens (91.0%). The 37 actor records account for 211,914
completion tokens and 6,189.6 seconds of recorded call time; seven summarizer
records account for 142,294 tokens and 4,216.9 seconds. These durations must
not be added into a wall-clock breakdown: their sum exceeds session duration
and overlap/accounting requires further tracing. Routine reasoning was enabled
globally in this run although the candidate documents routine reasoning off
with a separate bounded deep-investigation tool.

The ERROR and outer timeout remain valid observations. They do not erase
the completed gameplay evidence or prove infrastructure malfunction. The
whole-game result is censored at the budget; claims below that zero detector
eligibility proves structural-test unreachability are superseded by this note.

Revised next decision: measure routine reasoning off on fixed captured inputs,
test hypothesis-specific predicates with confounding changes, and qualify
bounded end-to-end completion before another long reliability run.

## Verdict

The Kaggle notebook ended in `KernelWorkerStatus.ERROR` after the official
launcher exhausted its precommitted 7,200-second timeout. The run is
**operationally invalid for judging solver capability**.

The preserved partial trace nevertheless falsifies a narrower claim: under
this configuration, the structural self-test was not practically reachable
within the run budget. The solver executed 10 actions, but recorded zero
eligible structural-detector attempts and completed zero levels.

No retry was launched.

## Frozen treatment

- Kaggle kernel: `pranavagrawai/structural-self-test-deep-20260909`, version 1
- Game: `ls20`
- Discovery arm: `D`
- Solver arm: `mindset`
- Reasoning: enabled
- Maximum actions: 100
- Official-launcher timeout: 7,200 seconds
- Notebook timeout: 10,800 seconds
- Contract SHA-256: `cce5829f641067d4e2e7668b0c7c29443e0a5c2874d7e87b73e6964a8edd7f76`

## Runtime outcome

- Model verification passed.
- SGLang warmed successfully in 149.203 seconds.
- The official game launcher ran from `2026-09-09T09:05:36Z` to
  `2026-09-09T11:05:06Z`.
- The launcher returned `124` (`timed_out: true`).
- Teardown and lifecycle containment passed.
- Request-path status is `invalid`; the probe's claim ceiling says that the
  run establishes no deployment-feasibility claim.

## Preserved solver evidence

- 1 game episode
- 10 executed actions across 4 submitted batches
- Actions: `RIGHT, LEFT, UP, UP, UP, UP, LEFT, LEFT, LEFT, DOWN`
- 9 grid-hash changes
- 0 levels completed; final state `NOT_FINISHED`
- 44 model-call records: 37 actor and 7 summarizer
- Calls by logical sequence: `9, 4, 6, 15, 10`
- 3 host batch stops, all caused by prediction mismatch
- 0 skipped actions and 0 unexecuted batch-tail actions
- 0 eligible structural-detector attempts

The first four decisions became actions after approximately 18.1, 14.3,
21.5, and 25.2 minutes respectively. The fifth logical sequence accumulated
10 actor-call records but produced no further action before the outer timeout.
The six-action tail of the fourth batch executed rapidly once that decision
arrived. This localizes the dominant bottleneck to model/decision crossings,
not game stepping.

## Interpretation boundary

The trace does **not** show that the structural testing policy is bad: it did
not exercise that policy. It shows that the configured solver could not reach
the policy often enough—or at all—before decision latency consumed the run.

The repeated prediction mismatches are valid mechanics evidence, but they are
not objective progress. Because the run was operationally invalid, the zero
level score must not be entered as a solver-capability failure in a broader
reliability estimate.

## Evidence

- `PRECOMMIT.json` and `EXPERIMENT-CARD.md` preserve the treatment and decision
  rules written before launch.
- `raw/nooa-join-public3/result.json` contains the top-level invalid verdict
  and lifecycle result.
- `raw/nooa-join-public3/official-launcher-result.json` contains the preserved
  game, action, prediction-check, and model-call summaries.
- `raw/nooa-join-public3/nooa-join-launcher.stdout.log` preserves the action
  timestamps.
- Kaggle recording SHA-256:
  `e5302a1d9fd884af4b0d792374dbbd4ae6c727d5fe5f2ce2531a19832d2d063e`.

## Decision

Do not spend another deep run merely by extending this timeout. That would buy
more of the same bottleneck and still confound the intended self-test with
decision latency. Before another consequential run, establish why each logical
decision generated 4–15 actor-call records and why the structural detector had
zero eligible attempts. A next card should vary that reachability bottleneck,
not the game-specific mechanics.

# q6 and Kaggle join review — 2026-09-05

Status: review complete for the scope below; candidate not ready to launch.
Baseline: `0141a36ed018b3fb9cd78ed8b834a1c298f107b8` (also current HEAD).
No committed delta; reviewed relevant working-tree changes and untracked sources.
No production files changed. This directory contains the review and reproductions.

## Scope and runtime

Reviewed adapter/runtime diffs, the untracked join worker, headless dispatcher,
factory, notebook builder, probe aggregation, join tests, q6 before/after patch,
offline assertions, and replay/sweep detector and endpoint code. Read the supplied
handoff and q6 experiment card as specifications, allowing for the handoff's
explicit no-extra-invocation amendment. This is not a review of every historical
run artifact, the stale 0158 source snapshot, or the entire Tycho implementation.

The official loop calls `MyAgent.is_done()` then `choose_action()`.
`MyAgent._reconcile_pending()` turns the successor frame into an action result.
`LocalSolver` verifies observation continuity and exchanges protocol-v3 messages
with a worker subprocess. `HostSession` publishes NOOA file IPC and drains a
submitted batch one official action at a time. `headless.py` runs the real
`MdArcSolverAgent`, whose `submit_actions()` validates gates and appends a batch.
Tycho supplies DTOs and protocol here; its gameplay machinery is not executing.

In local research, `launcher.py` starts the actor and `harness.py` owns environment
steps, observation history, prediction scoring, batch interruption, and terminal
evidence. Kaggle replaces that harness with `nooa_worker.py`. This replacement is
the critical seam: pinning the same actor does not preserve the host behavior.

The q6 addition evaluates stalls while rendering the state dashboard, raises an
integer-ID interlock, exposes `resolve_stall()`, and refuses ordinary submission
until the IDs agree. Resolution currently records a declaration and immediately
reopens ordinary batching; it does not implement the card's consequence loop.

## Standards

1. **P1 — The worker executes stale batch actions after level completion.**
   `nooa_worker.py:145-149,168-188` drains the queue without examining level
   advancement. Native `harness.py:807-812` discards the remaining actions.
   With `UP,DOWN,LEFT` and completion after `DOWN`, the real join emits `LEFT`
   on the new level. Immediately after observing completion, the only published
   state is still step 0. The new state arrives after the stale action.
   This violates the actor's documented batch contract
   (`chassis/solver_agent.py:1112`) and the user's evidence/actual-constraints rule.
   Repair the boundary transition and require publication before the next decide.

2. **P2 — Qualification asserts a weaker property than its stated requirement.**
   `test_join_qualification.py:287-305` says the old tail must not reach a new
   level, but asserts only output count and adapter level. It passes with the
   defect above. The user's requirement is a test that distinguishes live
   explanations; this test does not distinguish interrupted from uninterrupted
   batches. `repro_join.py` adds both discriminating checks and fails on both.

Judgment: duplicated host lifecycle logic is the consequential design smell.
The native harness and worker implement different semantics behind the same actor
interface. Establish shared host behavior or explicit parity fixtures before
adding another host-dependent gate. Private-helper imports in the candidate
factory also create coupling, but no additional failure was established there.

Count: **2 findings; worst P1, stale actions on a new level.**

## Spec

1. **P1 — Control receives treatment.** The card says control logs detection and
   “plays on; no review, no suspension.” q6 never reads `ARC_STALL_REVIEW`.
   `solver_agent.py:1119` always opens the interlock and `1377-1385` enforces it.
   Reproduction with the flag set to zero still opens the interlock. Implement
   shadow detection separately from treatment enforcement.

2. **P1 — The online detector is not the calibrated detector.** The card requires
   recomputation “using `replay.trigger_stall_repetition` at the frozen operating
   point.” Live `solver_agent.py:1092-1114` compares the last of six submitted
   batches with the preceding five and excludes tokens of length <=2. Replay
   (`replay.py:349,374-392`) compares a current rationale with six predecessors
   and retains short tokens. Live evaluation also follows execution of the last
   submitted batch. Seven identical `go up to x` rationales yield live 0 versus
   replay 1 detection. Establish one definition and clock, with parity checks;
   thresholds alone do not make these detectors equivalent.

3. **P1 — Resolution omits the action and consequence contract.** The card
   requires probe predictions with two distinct outcomes, a single legal action,
   and post-action falsifier/authority records. `resolve_stall()` at `1147-1188`
   accepts a probe without predictions and reopens ordinary multi-action
   submission. Its `STALL_RESOLVED` event precedes action execution. There is no
   required `AUTHORITY_REVIEWED` / `CONSEQUENCE_OBSERVED` / `AUTHORITY_RESOLVED`
   sequence, falsifier evaluation, or basis-after observation. P12 cannot be
   computed as specified. Folding review into an existing invocation does not
   require deleting these semantics.

4. **P2 — Evidence is truncated before measurement.** The card says bases are
   “compared verbatim as authored.” `solver_agent.py:1181-1183` cuts both bases
   and the falsifier at 500 characters. A valid switch differing after character
   500 records identical bases. Preserve full authored evidence; truncate only
   the display.

5. **P2 — Malformed resolutions bypass the specified stop counter.** The card
   requires abort after eight consecutive rejected resolutions. Validation
   returns at `solver_agent.py:1163-1176` never call the existing rejection logger
   (`846-884`). Eight malformed calls produce zero logger calls. Add explicit
   resolution rejection accounting and verify the actual abort consumer.

6. **P1 — Kaggle does not preserve the required q3c host gates/evidence.** The
   card holds fixed “the q3c scribe, and every q3a/q3b/q3c/q4 gate.”
   `nooa_worker.py:255-263` always writes empty `known_from_here` and null
   `last_retrodiction`, disabling the actor's re-entry and failed-prediction
   gates (`chassis/solver_agent.py:1159-1171,1216-1226`). Running the real
   submission method with populated evidence rejects the moves; substituting
   the worker's output accepts them. The worker also never writes
   `level_N_terminal.json`, so the scribe's `level_terminal()` returns an error
   and fallback notes contain missing facts. Batch no-impact interruption is
   absent. Outcome labels differ too: WIN plus a level increment is `level_up`
   in the worker (`72-81`) but `terminal` in the native harness (`177-186`);
   HUD-only changes are also treated differently. Port and qualify this host
   contract before using local baselines to interpret a Kaggle trial.

Count: **6 findings; worst P1, absent experimental contrast and host parity.**

## Verification and limits

From `deployment/kaggle`:

```bash
uv run --frozen pytest tests/test_my_agent.py tests/test_nooa_join.py tests/test_local_solver.py tests/test_join_qualification.py -q
uv run --frozen python ../../analysis/local-probes/20260905T-stall-authority-q6/review/repro_join.py
```

Suite: **45 passed, 1 skipped, 41.35 s**. Reproduction: **exit 1**, boundary state
absent and first new-level action `ACTION3` instead of fresh-batch `ACTION1`.

From repository root, use
`/home/pranav/projects/arc-agi-3-reference/nooa/.venv/bin/python` for each:

```text
analysis/local-probes/20260905T-stall-authority-q6/test_g5_offline.py
analysis/local-probes/20260905T-stall-authority-q6/review/repro_q6.py
analysis/local-probes/20260905T-stall-authority-q6/review/repro_host_contract.py
```

Original G5 script: **15 assertions pass**, including source-string checks for
submission enforcement. q6 reproduction confirms findings 1, 2, 4, 5 and probe
acceptance in 3. Host-contract reproduction confirms inert-action and failed-score
gate bypass, missing terminal evidence, and WIN label mismatch. These two scripts
report existing defects and exit zero on their successful reproduction; they are
not passing acceptance tests. Host-contract reproduction uses the real worker and
actor methods with minimal state fixtures, not a running model. The join
reproduction uses real adapter/runtime/worker subprocesses and a stub actor.

No paid model, Kaggle launch, full repository test run, or live gameplay occurred.
The handoff's claim that all full-suite failures predate its changes was not
independently verified. Scorer, manifests, runtime integrity assertion, remaining
qualification sequences, budget and card amendments are acknowledged unfinished
work, not discoveries of this review.

## Resume

Repair and qualify the Kaggle host contract first. Then unify online/offline
stall semantics, implement control versus treatment and the full resolution
trace, and freeze a scorer against scripted artifacts. Regenerate provenance and
size the pre-flight only after those checks pass. The current green tests cannot
support a q3c-equivalent pre-flight or an interpretable q6 contrast.

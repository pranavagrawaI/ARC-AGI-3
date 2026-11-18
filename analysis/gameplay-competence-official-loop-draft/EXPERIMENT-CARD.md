# gameplay-competence-official-loop — DRAFT (not frozen; no run identity; no launch authorized)

Status: draft written 2026-09-15 while R7 (`actor-critic-construction-kaggle-20260915T113548Z`) was running. Becomes a card only when copied into a fresh run directory under a `<identity>` with the three user decisions below filled in and the user has said "launch it" for that identity. R7's outcome does not change this card's question; it decides whether the construction overlay is machinery-valid enough to be the treatment.

Question: On public LS20 under the official `Agent.main` loop with the same Actor, same model, same budgets, does enabling host-owned construction (automatic freeze after the first executed action; frozen query-then-execute; assimilate on matching successor) change gameplay — levels_completed, WIN/GAME_OVER, executed actions, actions-to-first-level — relative to the same Actor with construction disabled?

Posture: first gameplay measurement of the construction join; paired within-kernel comparison; no prior gameplay evidence for construction exists. R2–R7 measured machinery validity only (coordinator probe driver, `test_join_qualification.build_agent`, test Agent base, 8 actions, 24 chats) and are explicitly NOT gameplay evidence. Historical LS20 references under the official loop (single runs, no control): v8 17 actions / 3.7 calls per action; v10 6 actions / 80 Actor calls / 13.3 calls per action; both zero levels.

Contrast: two arms in ONE kernel, same served Qwen3.8-27B-FP8 SGLang instance, run sequentially after a shared warmup:
- A (baseline): official `main.py --agent myagent --game ls20` via `run_official_agent.py`; `MyAgent` adapter from `nooa-join/adapter.py`; NOOA worker command WITHOUT constructor transport (`ARC_CONSTRUCTION_ARM=0`); incumbent critic on.
- B (treatment): identical, except the worker command is `headless.py --constructor-transport http --constructor-url <relay>/chat/completions --constructor-wall-seconds 45 --constructor-hard-seconds 50` so the Actor is `LiveConstructionAgent` (R6/R7 host-owned freeze use). Constructor calls are charged to arm B's own admission budget and request purse.
Order: A then B (fixed; the shared model is warm for both because warmup precedes A). The arms must differ in exactly one binding: the worker command's constructor arguments. Any other difference (env var, max actions, timeouts, adapter bytes, prompt bytes outside `CONSTRUCTION_API`) invalidates the comparison — see `codex` board thread 383 (arms differed in eligibility → invalid).

Prediction (each refutable by the run):
- P1 Activation: in arm B `host-events.jsonl` records `request_construction` after the first executed `transitions.jsonl` row, within the first 120 s of admission, with a `FROZEN.json` (digest-valid) or a bounded FAILED status after at most two constructor requests. Zero activations is a measured result (Actor executed nothing), not a machinery pass.
- P2 Use: if a freeze exists, at least one `query_frozen_predict` bound to an executed `(action, step)` and at least one `assimilate_frozen` on a matching successor before arm B's admission ends.
- P3 Cost: arm B executes fewer or equal actions than arm A within the same 600 s admission (construction spends ≥45 s and adds prompt text). Stated so that "more actions in B" would be a surprise worth investigating, not a claim.
- P4 Levels: NO prediction of a level in either arm (v8/v10 reference: zero). levels_completed is reported per arm; a level in exactly one arm on a single paired run is an anecdote, not a difference.
- P5 Tempo: Actor calls per executed action per arm reported against v10 LS20 (13.3); no prediction.

Measure (raw artifacts, per arm workspace under `/kaggle/working/<out>/<arm>/`):
- `qualification/game-progress.jsonl`: `levels_completed`, `stop_reason`, `actions` (executed), `game_elapsed_seconds` — this is the primary gameplay record (same producer/fields as v10's `audit.py`).
- Official launcher stdout/scorecard for WIN / GAME_OVER / NOT_FINISHED and the official action count; must agree with `game-progress.jsonl` executed actions after separating submitted batches from executed actions and host-skipped tails.
- `ipc/model_calls.jsonl` (`request_purpose` ∈ actor/critic/constructor), `ipc/actions.jsonl`, `ipc/transitions.jsonl`, `host-events.jsonl` (request_construction / query_frozen_predict / assimilate_frozen with bound action/step), `FROZEN.json`, constructor `RESULT.json`, raw HTTP bodies.
- Server-side chat-completion count reconciled against application receipts + 1 warmup, as in R6/R7.
- Arm-equality check: byte-equal adapter, worker, prompt files and env dumps for A and B except the constructor arguments; recorded in `ARM-DIFF.json` before play.
Interpretation: levels and stop reason first; then actions and calls per action; then P1/P2 lifecycle; treat any arm-equality violation as invalid before reading numbers.

Cap (numeric; inherits the v10 official-loop envelope, not R7's probe envelope):
- One private Kaggle notebook, NvidiaRtxPro6000, internet off, competition attachment `arc-prize-2026-arc-agi-3`, session timeout 3600 s, one GPU hour reserved, quota ≥ 1 h at push. One push, zero retries within the identity.
- Model readiness ≤ 1200 s; notebook work deadline 3100 s; guardian 3500 s + 8 s; bootstrap trusted-shutdown reap 30 s (R7).
- Per arm: 600 s model-admission window, 200 Solver actions, Actor max 16384 output tokens, critic max 192, HTTP timeout 90 s, retries 0, reasoning off. Arm B constructor: 2 requests, 45 s soft / 50 s hard, per-POST 30 s.
- Total server chat requests: ≤ 260 including 1 warmup (v10 used 224 for three games; two LS20 arms at ≤ 130 each). Missing usage on any receipt stops further admission in that arm.
- Wall: A + B play ≤ 1200 s + constructor overhead; whole kernel expected ≈ 40 min (v10: 38 min for three games).
- Spend: ≈ 0.6–0.7 GPU-hours of the 27.89 h remaining (refresh 2026-09-19).

Stop:
- Success/normal: both arms end by `model_admission_budget`, `wall_budget`, `step_budget`, or `win`; orderly SGLang shutdown; guard passes with zero leftover descendants; kernel COMPLETE.
- Futility: arm A ends with 0 executed actions → still run arm B (activation is the question) but mark P3/P5 unexercised.
- Abort: model readiness > 1200 s, seal failure, wrong device, source drift, arm-equality violation detected before play → abort before any admission; preserve evidence; no second push.
- Remote terminal COMPLETE/ERROR/CANCEL_ACKNOWLEDGED ends observation; provider timeout is the backstop; the local monitor never cancels or resubmits.

Invalid if: arms differ in anything but the constructor arguments; the loop is not the pinned official `main.py` (revision and file hashes as in v11 `build_notebook.py`); an Actor DONE ends a game (v9 invalidation); OOM, transport failure, missing usage, or a launcher exception; construction activates in arm A; `game-progress.jsonl` and the official scorecard disagree on executed actions; source/notebook drift versus the frozen manifest; unresolved cleanup; competition submission of any kind (unauthorized; `KAGGLE_IS_COMPETITION_RERUN` refused as in v11).

Result map:
- Valid, B ≥ A on levels with P1+P2 held → one paired observation that construction did not hurt and its lifecycle ran under the real loop; next: repeat on S5I5 and CD82 in a new identity before any "better" wording; no submission claim.
- Valid, B < A on actions with zero levels in both (the P3 expectation) → construction costs tempo and buys nothing measurable on LS20 at 600 s; next: decide whether the freeze earns its 45 s (either shorten the constructor budget or move activation later) — a policy change, new card, user decision.
- Valid, P1 fails (no activation because the Actor executed nothing in 600 s) → the binding constraint is Actor tempo (memory: 13.3 calls/action), not construction; next: tempo work on the Actor before more construction runs.
- Valid, P2 fails with a freeze present → host-owned use is broken under the official loop but not under the probe; mechanical repair, new identity.
- Invalid → preserve, diagnose first failing boundary, new identity; no gameplay claim.

Outputs: `analysis/<identity>/` with manifest.json, authorization.json, launch-review.json, CONSUMED.json, launch.json, every raw CLI/SDK receipt, version-bound direct downloads of `game-progress.jsonl`, both arm workspaces' `ipc/*.jsonl`, `host-events.jsonl`, `FROZEN.json`, official launcher stdout, `PROCESS-GUARD.json`, `SHUTDOWN-REAP.json`, server log, request-count facts, final status and quota; RESULT.md separating observation / interpretation / permitted next action.

## Decisions the user owns before this becomes a card

1. Envelope: this draft takes v10's per-game envelope (600 s admission, 200 actions) for each arm — NOT R7's 24-chat probe envelope, which cannot measure gameplay (13.3 calls per action ⇒ ≈ 2 actions). Widening beyond 600 s per arm (e.g. 900 s) fits in the hour only if arm count stays at two.
2. Games: LS20 only (this draft) versus LS20 + S5I5 + CD82 with one arm each (no paired control) — the draft argues for the paired LS20 design; the three-game design answers "does it run everywhere" instead.
3. Baseline choice: same Actor with construction disabled (this draft) versus the frozen v10 notebook as historical baseline (cheaper: one arm, but confounded by every R2–R7 change to the chassis).

## Build required (not started; delegate scope once decided)

- New package `deployment/kaggle/actor-critic-construction-kaggle-r8-gameplay/` from R7: replace `kaggle_probe.py`'s hand loop with the v11 official-launcher composition (pinned official source copy + hash check, `agents/templates/my_agent.py`, `.env` OFFLINE with `ENVIRONMENTS_DIR` = the exact `ls20/9607627b` directory R4 validates, `run_official_agent.py` fail-fast) run twice with the two worker commands; keep R4 discovery, R5 contract, R6 freeze use, R7 reap byte-identical.
- Local qualification before freeze: the actual `main.py` path through the real CLI parser against the offline public LS20 with a fixture model server; arm-equality test; game-progress + scorecard reconciliation test; construction-activates-only-in-B test; existing 94 + gate 119 tests green.
- Gate/review/launch machinery: retarget `verification-r7` as done for R7.

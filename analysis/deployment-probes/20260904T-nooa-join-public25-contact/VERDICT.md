# Verdict — 20260904T-nooa-join-public25-contact

Kernel `pranavagrawai/prototype-nooa-join-public25-contact` version 1. Status ERROR. Elapsed 12185 s (~3.38 h). No retry.

## Probe

**Held.** `public_progress_error` null, `solver_sessions_error` null. All 25 expected games present. Schema `nooa-join-public-progress-v2`. Kernel ERROR is `check=True` on rc=71 after a worker thread died.

## Crash

Not GAME_OVER continuity. `TimeoutError: worker timed out waiting for decision` in `LocalSolver.decide` (`Thread-9`). Contact set `TYCHO_NOOA_WORKER_TIMEOUT_SECONDS=5400`. Official Swarm starts every game in its own thread; C=1 serializes model generation. One third-turn decide waited out the 90 min worker cap and the process group was SIGTERM'd.

## Play (truncated: ~2 actions/game of 8)

Cap was 8. Recorded 51 frames, 20 grid-hash changes, 52 IPC env actions, **0 levels**. Every game except `wa30` has two one-action batches. `wa30` submitted one batch of four `UP`s (3 frames recorded).

Moved on those first steps (20):

`ar25 bp35 cn04 g50t lf52 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30`

No grid change (5): `cd82 dc22 ft09 ka59 lp85`

First IPC action:

| first | games |
|---|---|
| UP | ar25 cn04 dc22 ls20? no — ls20 RIGHT; dc22 UP, m0r0 UP, wa30 UP×4 |
| directional | ar25 UP, bp35 LEFT, cd82 LEFT, cn04 UP, dc22 UP, g50t DOWN, lf52 RIGHT, ls20 RIGHT, m0r0 UP, re86 LEFT, sc25 LEFT, sk48 RIGHT, sp80 DOWN, tr87 DOWN, tu93 DOWN, wa30 UP |
| CLICK | ft09, ka59, lp85, r11l, s5i5, su15, tn36, vc33 |
| SPACEBAR | sb26 |

(Exact first batches are in `official-launcher-result.json` `solver_sessions.games.*.first_batch`.)

## Card

Stop: success if ≥1 game has `grid_hash_changes >= 1`. Timeout with nonempty progress is a result.

- Moved: 20/25. Levels: 0.
- Did not complete 8 actions.

Result map: **positive, truncated.** Rank the 20 that moved. Do not retry 25×8. Do not deepen the five no-change games on this sample. Eight-action contact was not measured; two-action first-touch was.

## Claim ceiling

The q3c join on R8-L3B can list all 25 public games and take a first action in each. It does not establish an 8-action budget, RHAE, or that any game completes a level. 25 concurrent Solver workers under C=1 can hit the 5400 s per-worker receive timeout before the 8-action cap.

Outputs: `analysis/deployment-probes/20260904T-nooa-join-public25-contact/kaggle-output/`.

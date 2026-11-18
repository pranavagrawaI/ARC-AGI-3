# R9-cd82 result — actor-critic-construction-kaggle-r9-cd82-20260915T183130Z (comparable paired contrast)

Machine-readable: `R9-cd82-RESULT.json`. Raw: `runtime-evidence-exact-request/` (79 files, sha256 in the JSON:
27 core paths + 57 per-arm files fetched by the session directories the driver recorded — no tree listing),
push/quota/target-list receipts, 75 status polls, `R9-cd82-CURRENT-TERMINAL.json`.

## Validity (before any number)

| Check | Result |
|---|---|
| Arm equality (`ARM-DIFF.json`, computed remotely before play) | `equal: true`; differing keys exactly {ARC_CONSTRUCTION_ARM, NOOA_AGENT_COMMAND, RECORDINGS_DIR, TYCHO_KAGGLE_WORKSPACE_ROOT}; forbidden [] |
| Admission | A 605.4 s (`wall_budget`), B 605.7 s (`model_admission_budget`) — both the same 600 s window, closed by the two co-equal budget guards |
| Official recording ↔ driver progress | reconciled both arms (11 recorded frames = 11 executed actions; final state/levels agree); `game-progress.jsonl` == `games` |
| Construction in baseline | none (arm A: no freeze, no request, 0 constructor calls) |
| Actor DONE / GAME_OVER / transport failure / cap | none |
| Accounting | 126 admitted = 126 receipted, complete; server 127 = 126 + 1 warmup, verified; purse 258 not approached |
| Teardown | guard passed, `killed_remaining_descendants []`; reap retired, zero leftovers; orderly shutdown; no GPU processes after |
| Provider | COMPLETE; runtime files by `version_number=1` |

**Verdict: VALID.** Finding 7's guard did not fire; finding 6 slack 215 s.

## Observation

| | Arm A (baseline) | Arm B (construction overlay) |
|---|---|---|
| Executed actions (= submitted) | **11** | **11** |
| Levels / state | 0 / NOT_FINISHED | 0 / NOT_FINISHED |
| Executed sequence | UP LEFT RIGHT LEFT DOWN UP SPACEBAR SPACEBAR RIGHT SPACEBAR DOWN | UP DOWN LEFT RIGHT SPACEBAR SPACEBAR UP `CLICK 31 41` LEFT LEFT SPACEBAR |
| Grid changed per frame | 11/11 | 9/11 |
| Actor calls / per action (P5) | 52 / **4.73** | 56 / **5.09** |
| Critic calls (wall) | 10 (16.3 s) | 7 (11.3 s) |
| Constructor | 0 | **1** call, `submit_candidate`, 19.6 s, 19,539 prompt tokens → `FROZEN.json` (43 lines, digest 260b4e2b…) |
| Freeze lifecycle (host-events.jsonl) | — | request → **9 × (query → assimilate)**, each bound to the executed (action, step) |
| Actor wall in admission | 576 s / 605 | 560 s / 606 |
| Actor prompt tokens mean / max | 42,504 / 73,566 | 44,889 / 83,303 |

References: v8 CD82 12 actions / 4.5 calls per action; v10 CD82 17 / 3.1; R8 LS20 A 10 / 5.7, B 19 / 3.16.
Timeline: push 20:00:03Z → supervisor 20:00:41–20:26:03 (1523 s) → COMPLETE first observed 20:38:25Z. GPU 0.44 h.

## Predictions

- **P1 activation — held.** Fired after the first executed transition; froze on its first request.
- **P2 use — held, with a precise exception.** 9 of the 10 post-freeze actions were queried and assimilated. The
  one that was not is step 8, `CLICK 31 41`: `live_agent.submit_actions` binds a frozen query only to an action
  string present in `available_actions`, and `CLICK 31 41` ≠ `CLICK`. This is the same mechanism the s5i5 card
  predicts will make its whole use phase inert — here observed live on one action.
- **P3 cost — no prediction; observed B = A** (11 = 11) in equal windows; the constructor cost 19.6 s.
- **P4 levels — held (0/0).**
- **P5 tempo — reported:** 4.73 (A) vs 5.09 (B); both below R8's baseline 5.7, above v10's 3.1 on this game.

## Interpretation

1. Second valid paired run of the join under the real official loop; on a directional surface the freeze is
   used continuously — the lifecycle is not a one-off event as in R8 (one query/assimilate) but repeats per
   action. Still unscored ABI replay: `supported_model: false`, `evidence_claim: ABI_VISIBLE_REPLAY_ONLY`.
2. Equal action counts with the overlay slightly slower per action (5.09 vs 4.73) — with R8's 19 vs 10 in the
   other direction, two pairs now bracket zero; no direction can be claimed. Construction neither cost nor
   bought tempo here.
3. Neither arm approached a level. ~95% of admission is model wall (~10–11 s per call on ~43k-token prompts),
   as in R8. Arm B's actions changed the grid on 9/11 frames vs 11/11 — two of its moves were no-ops; too few
   to read.
4. All submissions executed (11/11 both arms), unlike R8's LS20 gaps (18→10, 24→19): on this game the host did
   not decline any submitted batch.

## Permitted next action (result map: valid, P1+P2 held)

- Freeze this identity; no further pushes. This is the second paired observation with R8; executed actions per
  arm now read 10/19 (LS20) and 11/11 (CD82). With two games not showing the same direction, no multi-seed
  "better" test is warranted on this evidence; the honest statement is "no measurable effect on actions at 600 s".
- The `CLICK x y` bypass is now observed on a directional game too; extending host-owned freeze use to CLICK is
  an R6 capability change with its own card (PM ruling: not now).
- No competition submission is authorized or implied.

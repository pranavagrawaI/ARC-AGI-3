# R9-s5i5 result — actor-critic-construction-kaggle-r9-s5i5-20260915T183130Z (weaker contrast; in this run: overlay WITHOUT a freeze)

**Label, first:** the card bounded this identity to an overlay-with-inert-freeze contrast. In the run, arm B's
construction phase ran once and did **not** freeze (`BUDGET_EXHAUSTED`). So the numbers below are an
overlay-without-freeze arm against the baseline. They are **not evidence about host-owned construction**, and
the CLICK-inert mechanism the card predicted was **not exercised** — there was no freeze for it to leave unused.

Machine-readable: `R9-s5i5-RESULT.json`. Raw: `runtime-evidence-exact-request/` (79 files: 27 core + 57 per-arm
files by the recorded session paths — arm-A `s5i5-18d95033-axlwfb8w`, arm-B `s5i5-18d95033-xmnhchcq`; the only
planned path not retrieved is arm-B `FROZEN.json`, which was never produced), receipts, 78 status polls.

## Validity (before any number)

| Check | Result |
|---|---|
| Arm equality (`ARM-DIFF.json`, remote) | `equal: true`; differing keys exactly the four permitted; forbidden [] |
| Admission | A 609.1 s (`wall_budget`), B 627.1 s (`model_admission_budget`; 600 s window + the last in-flight call's tail, inside the 90 + 15 s reserve) |
| Official recording ↔ driver progress | reconciled both arms (33 = 33, 11 = 11; final state/levels agree) |
| Construction in baseline | none |
| Actor DONE / GAME_OVER / transport failure / cap | none — **no GAME_OVER frame in either arm (44/44 NOT_FINISHED)** |
| Accounting | 98 admitted = 98 receipted, complete; server 99 = 98 + 1 warmup, verified |
| Teardown | guard passed, `[]`; reap retired, zero leftovers; orderly shutdown; no GPU processes after |
| Provider | COMPLETE; files by `version_number=1` |

**Verdict: VALID.** Finding 7's guard did not fire; finding 6 slack 211 s.

## Observation

| | Arm A (baseline) | Arm B (overlay; no freeze this run) |
|---|---|---|
| Executed actions | **33** (64 submitted in 22 batches; 5 distinct actions; `CLICK 0 0` × 29; last action RESET) | **11** (11 submitted, 11 distinct-ish clicks) |
| Levels / state | 0 / NOT_FINISHED | 0 / NOT_FINISHED |
| Actor calls / per executed action | 38 / **1.15** (batching artifact) | 43 / **3.91** |
| Critic calls (wall) | 11 (18.0 s) | 5 (8.5 s) |
| Constructor | 0 | **1** call, 29.0 s, 19,409 prompt tokens → `BUDGET_EXHAUSTED`, `frozen: false`, host event `REJECTED` |
| Freeze lifecycle | — | none (no FROZEN.json; zero query/assimilate — trivially) |
| Actor wall in admission | 574 s / 609 | 578 s / 627 |
| Actor prompt tokens mean / max | 44,886 / 93,015 | 44,432 / 70,868 |
| Actor mean call | 15.1 s | 13.4 s |

References: v10 S5I5 17 actions / 3.9 calls per action; 2026-09-04 depth run 55 IPC actions with GAME_OVER at
frame 49. Timeline: push 20:43:45Z → supervisor 20:44:21–21:10:40 (1579 s) → COMPLETE first seen 21:23:26Z. GPU 0.45 h.

## Predictions

- **P1 activation — held by its second clause.** Construction fired after the first executed transition and
  ended with a bounded failure inside the request cap: one 29.0 s constructor call, then `BUDGET_EXHAUSTED`
  (45 s soft budget could not fund a second request). Cf. cd82: one call, 19.6 s, froze. On this game the
  constructor's single attempt did not yield a candidate.
- **P2 use — untested.** The prediction was "freeze produced AND zero query/assimilate". The premise failed. The
  observed zeros carry no information about the `CLICK x y` mechanism; the live evidence for that mechanism is
  cd82's step-8 `CLICK 31 41` (queried on no other click, queried on every directional action).
- **P3 — no prediction.** A 33 > B 11, but A's count is 29 repeats of `CLICK 0 0` submitted in batches and a
  closing RESET; B's 11 are individually chosen clicks. Executed-action count is not comparable across these
  two behaviours.
- **P4 levels — held (0/0).**
- **P5 tempo — reported:** 1.15 (A; batching) vs 3.91 (B) — B matches v10's 3.9 on this game.
- **GAME_OVER→RESET path — not exercised.** No GAME_OVER occurred. Arm A's final RESET was an Actor-submitted
  action; the official loop's GAME_OVER reset never ran.

## Interpretation

1. This identity did not deliver even the weaker contrast the card allowed: the overlay's construction phase
   failed to freeze, so arm B is the overlay's prompt text plus one failed 29 s constructor call. Read it as:
   the overlay and the official loop survived a click-only surface (P1 bounded failure, orderly stop, clean
   teardown) — and nothing more.
2. The baseline arm's 33 actions are mostly a degenerate loop (`CLICK 0 0` ×29 in batches) — the batch gate
   rewarding the most predictable action, the pattern already on record for this chassis. That is an Actor
   behaviour finding on click surfaces, independent of construction.
3. Neither arm approached a level; ~95% of admission is model wall, calls are slower here (13–15 s mean) on
   ~45k-token prompts than on LS20/CD82.
4. The constructor's outcome differs by game (cd82 froze in 19.6 s; s5i5 exhausted its budget at 29 s in one
   attempt). One attempt each; no claim about what the difference is due to.

## Permitted next action (result map: valid; P1 bounded failure; P2 untested)

- Freeze this identity; no further pushes. Do not write any host-owned-construction conclusion from it.
- If the CLICK-inert mechanism is to be tested on S5I5 deliberately, a freeze must exist first: that is a
  constructor-budget or constructor-attempt question (its own card), then, separately, the R6 CLICK binding
  capability change (PM ruling: not now).
- No competition submission is authorized or implied.

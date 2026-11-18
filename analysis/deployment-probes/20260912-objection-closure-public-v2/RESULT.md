# Objection closure public run v2 result

## Verdict

**Valid.** Kernel `pranavagrawai/arc3-objection-closure-v2-20260912` v1 ran
immediately (no queue), COMPLETE in ~38 minutes, 0.61 GPU-hours. All three games
received their 600-second admission window and ended by budget. 224 model
requests admitted, 224 completed, 0 failed. No DONE refusal occurred. Result
`passed`. Zero levels.

Every operational prediction on the card held (`audit.py`, asserted). The one
FAIL the audit printed — an objection pending at game end on LS20 and S5I5 —
is the card's stated exemption: both opened in the final call (6.5s and 17s of
admission left) and `budget_exhausted.json` ended the game.

## Against the references (single runs each; no control)

| game | stop | levels | actions | v8 | Actor calls | **calls / action** | v8 | critic wall | v8 | critic mean | max prompt | ratio to Actor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LS20 | admission | 0 | **6** | 17 | 80 | **13.3** | 3.7 | 2.2% | 26.0% | 1.6s | 2,068 | 6.8% |
| S5I5 | admission | 0 | 17 | 18 | 67 | 3.9 | 2.5 | 2.7% | 20.6% | 1.8s | 2,536 | 6.8% |
| CD82 | wall | 0 | 17 | 12 | 53 | 3.1 | 4.5 | 2.0% | 18.4% | 1.7s | 2,645 | 9.3% |

## Contract events (24 reviews, 23 objections)

| | LS20 | S5I5 | CD82 |
|---|---:|---:|---:|
| reviews completed / invalid / restated | 8 / 0 / 3 | 9 / 0 / 0 | 7 / 0 / 0 |
| objections opened → closed / expired | 7 → 6 / 0 | 9 → 8 / 0 | 7 → 7 / 0 |
| dispositions | narrow, defend ×5 | probe ×4, defend ×3, narrow | narrow ×5, probe, withdraw |
| reminders before disposition | 1, 1, 2 | — | — |
| probe budgets audited | 0 | 4 | 1 |
| recurrence rejections | 0 | 0 | 0 |
| ledger-rejected `closes` | 0 | 0 | 0 |

Zero invalid critic outputs in 24 (v8: 1 in 22; the saved-trace probes ran 2 in
12 at window 10 and 0 in 24 at windows 3 and 0). Every question bound. No
question stated a remaining-action count as fact; the two budget questions
(LS20 critic:8, S5I5 critic:9) gave seconds and asked which evidence could
still be obtained.

## What this establishes

1. **The critic is no longer a wall-clock problem.** 2.0–2.7% of session wall
   on every game, against 18.5–26.0% in v8 — an order of magnitude, on three
   games, with the same model. Restatement, the packet, and the ledger all
   worked as specified under live play.
2. **The Actor cannot end a game.** No DONE was attempted; the defense was not
   exercised. `test_done_refusal.py` remains the evidence that it works.
3. **The cost saving bought no levels and, on LS20, no actions.** LS20 executed
   6 actions in 80 Actor calls — 13.3 calls per action, against 3.7 in v8 with
   the expensive critic. S5I5 and CD82 sat near v8. With the critic at ~2%,
   the Actor's own inspection loop is the whole budget.

## What this does not establish

- Anything about levels. Three runs, three architectures, zero levels. The
  critic line of work has reached the point where its cost is negligible and
  its questions are grounded, and the games are not being solved; the binding
  constraint is the Actor's calls per executed action.
- Any causal claim about v10 versus v8. Single runs; no resolution band.
- Whether withholding cell values suppresses mechanics questions. It did not:
  roughly 14 of 24 questions concern changed-cell counts and bounding boxes
  ("why did UP change only 2 cells while LEFT changed 52?"). The counts alone
  are enough material. The difference from v8 is that such a question now
  costs ~2 seconds and one `defend` sentence rather than 30 seconds; LS20's
  five consecutive `defend`s show the Actor answering with measurements and
  changing nothing.

## Result map

Valid, operational predictions hold → the contract and its cost are
live-qualified. The permitted next steps are (a) a repeated-run card for a
behavioral resolution band before any v10-versus-v8 claim, and (b) a separate
diagnosis of Actor calls per executed action, which on LS20 is 3.6× v8's.
Nothing here authorizes promotion or competition submission.

## Evidence

- `AUDIT.json` / `audit.py`: per-game metrics, every question and `closes`
  payload verbatim, card checks.
- `raw/qualification/solver-workspaces/*/ipc/`: `model_calls.jsonl`,
  `critic_reviews.jsonl`, `objections.jsonl`, `request_lifecycle.jsonl`,
  `actions.jsonl`, `budget_exhausted.json`.
- `raw/qualification/game-progress.jsonl`, `solver-workspaces/game-stops.jsonl`.
- `PRELAUNCH.json`, `push.log`, `monitor.jsonl`, `postrun-status.log`,
  `postrun-quota.log`.

# Solver diagnosis, 12 September 2026

Successor to `analysis/solver-diagnosis-20260906/REPORT.md`. Same six frontiers
(board thread 119), same 1–10 scale of demonstrated capability. Evidence is the
three public Kaggle runs of 12 September (v8 valid, v9 invalid, v10 valid), the
72-call saved-trace critic probes, and the analysis scripts named at the end.
Every number below is computed by `actor_failure_analysis.py` or the run audits.

## Judgment

The critic is no longer where this solver fails. Across three runs it went from
18.5–26% of each game's wall clock to 2.0–2.7%, its questions bind to Actor
claims, and every objection is disposed of. Zero levels in all three runs.

The Actor fails in three ways the traces now make visible and separable:

1. **It never forms an objective it will bet on.** Across seven game-runs and
   about ninety submissions, the Actor predicted `level_up` or `terminal` zero
   times. Rationales are probe-only in 60–75% of submissions on the games with
   the most actions. It measures mechanics precisely and defends those
   measurements when challenged; it does not convert them into a plan that
   could end a level.
2. **It confabulates host events and acts on them.** v9-LS20: "Budget
   exhausted (100/100)" from turn 4 (no such surface exists), then DONE with
   152s and 193 actions left — the run died. v10-LS20: "Context was evicted.
   Re-orient" sixteen times in one 27-call turn while its prompt grew
   monotonically 53.8k → 70.2k tokens — a third of the game's window spent
   re-reading state that was never lost. Both on LS20, both reasoning-off.
3. **It does not follow an in-turn host instruction on the first attempt.**
   The objection injection says the next `submit_actions` must carry
   `closes=`; twenty times in v10 the Actor submitted without it, was refused,
   and resubmitted. About twenty extra calls, ~160s, is the contract's cost
   in the Actor's hands.

The critic work bought a clean look at the Actor. What it shows is a model
that measures well, plans little, and periodically believes things the harness
never said.

## Ratings

| Frontier | 09-06 | **09-12** | Judgment |
| --- | ---: | ---: | --- |
| 1. Speed and Action Economy | 3 | **3** | The critic tax is gone (26% → 2%) and the Actor absorbed the time: 3.1–13.3 calls per executed action, 6–17 actions per 600s game, batching earned once in three games. LS20-v10 spent 27 calls on one turn. |
| 2. Tools, Math, and Representation | 5 | **5** | Exact diffs, footprints, and shift rules are measured reliably and cited under challenge. No representation of the objective ever appears; helper writing fell to 0–5% of calls in v10, so the batch gate is never satisfied. |
| 3. Harness and Interface | 7 | **7** | Three-game fair-share budgeting, exact lifecycle accounting, and clean containment held in v10. v9 exposed a latent deadlock (Actor DONE) now closed and covered by a test that reproduces it. Tool results are still not preserved, which is why "100/100" cannot be traced. |
| 4. Metacognition and Judgment | 2 | **2** | Zero terminal-class predictions in ~90 submissions. Confabulated "budget exhausted", "stuck", and "context evicted" drove a quit and a 200-second spiral. Grounded objective questions are closed with `defend` or `narrow` and the policy does not change (LS20-v10: five consecutive defends). |
| 5. Learning, Evaluation, and Retention | 4 | **5** | A reproducible gate chain now exists: preserved trace → saved-trace probe under a card → loopback test that fails on the prior runtime → Kaggle run under an immutable card with pre-registered predictions and an executable audit. Still no resolution band, no cross-run retention of anything the Actor learned. |
| 6. Portfolio and Resource Allocation | 4 | **4** | 72 OpenRouter calls (cents) replaced what would have been Kaggle iterations. But v9 spent 0.30 GPU-h on an avoidable crash, and three Kaggle runs went to the same three games with the same outcome before the binding constraint was identified. |

Frontiers 1 and 4 are the same problem seen from two sides: the Actor's serial
inspection loop is long *because* it has no objective to make inspection stop.

## Evidence

### Where the Actor's calls go (v10, valid run)

| game | calls | actions | calls/action | share of calls: inspect / submit / notes+helper+plan | rejected submits (of which `closes`) |
|---|---:|---:|---:|---|---:|
| LS20 | 80 | 6 | 13.3 | 66% / 18% / 17% | 8 (6) |
| S5I5 | 67 | 17 | 3.9 | 52% / 42% / 4% | 10 (8) |
| CD82 | 53 | 17 | 3.1 | 51% / 47% / 0% | 8 (6) |

v8 for reference: 3.7 / 2.5 / 4.5 calls per action with the 20–26% critic tax.

### Objective formation (all seven game-runs)

| | v8 cd82 | v8 ls20 | v8 s5i5 | v9 ls20 | v10 cd82 | v10 ls20 | v10 s5i5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `level_up`/`terminal` predictions | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rationales mentioning an objective | 2/12 | 0/17 | 0/14 | 1/7 | 1/17 | 0/6 | 2/17 |
| probe-only rationales (v8, v10) | — | 10/17 | 1/14 | — | 4/17 | 4/6 | 13/17 |
| distinct actions used | 8/12 | 4/17 | 5/30 | 4/7 | 10/17 | 3/6 | 9/18 |

### Confabulated host events

| run/game | phrase | count | host surface | consequence |
|---|---|---:|---|---|
| v9 ls20 | "Budget exhausted (100/100)" | 8 cells | none found (states, observations, chassis, skill, NOOA, notebook env) | DONE at 152s/193 actions remaining; run ERROR, two games lost |
| v9 ls20 | "game is stuck / cannot solve" | 4 cells | none | same |
| v10 ls20 | "Context was evicted. Re-orient" | 7 cells, 16 re-orientations in turn 4 | none: prompt tokens rose 53.8k → 70.2k through the turn | 27 calls, ~200s, one action |
| v8 s5i5 | "budget" | 1 | the true admission status line | none |

Both incidents are on LS20, whose bottom "c-panel" bar erodes one cell per
action and which the Actor called "fuel". That coincidence is noted, not
established.

### What the critic contract now costs and delivers (v10)

23 objections, 21 disposed (narrow 7, defend 8, probe 5, withdraw 1), 0 expired,
0 recurrence rejections, 5 probe budgets audited, 3 free restatements, 0 invalid
outputs in 24. Critic wall 2.0–2.7% per game at 1.6–1.8s per call. Withholding
cell values did not stop mechanics questions (~14 of 24 concern changed-cell
counts); it made them cheap. The contract's remaining cost is the ~20 refused
submits above.

## What this does not establish

- That reasoning-off is the cause. Every run so far is reasoning-off
  Qwen3.8-27B. Confabulated host events and the absence of any terminal
  prediction are consistent with a model that has no reasoning budget, and
  also with the prompt's shape. Untested.
- That the "100/100" and "evicted" beliefs came from nowhere. Tool results are
  not preserved in `model_calls.jsonl`, so an unpreserved rendering cannot be
  ruled out. The prompt-token trace rules out actual eviction.
- Any comparison between v8 and v10 gameplay. Single runs, no band.

## The next discriminating test, in order of cost

1. **Preserve tool results and the rendered prompt** in `model_calls.jsonl`
   (harness, frontier 3). Without it, every "the model believed X" finding
   stays one step from its cause. Cheap; no model calls.
2. **Replay LS20-v10 turn 4 offline** once prompts are preservable, asking a
   fresh model call what in the context reads as eviction. Cents.
3. **One reasoning-on contrast on LS20** under a card: same runtime, thinking
   enabled, one game, measuring terminal-class predictions and confabulated
   host events — not levels. This is the frontier-4 question the board has
   deferred since 08-29, and it is now the cheapest thing that could change
   the model choice. Latency will roughly halve actions per game; the card
   should predict that and measure calls per action anyway.
4. **Retire the `closes`-omission cost** by accepting a bare submit as
   `defend` when the rationale cites a measurement — a contract simplification
   that removes ~20 calls per run. Deletion-shaped; test on the v10 traces
   first.

Not recommended: further critic work as a lever on levels; another same-design
Kaggle run without one of the above.

## Scripts and raw evidence

- `analysis/critic-closure-contract-20260912/actor_failure_analysis.py`
  (`--seq v10:ls20` prints the call sequence); `actor_failure_summary.json`.
- `analysis/deployment-probes/20260912-objection-closure-public-v2/audit.py`,
  `AUDIT.json`, `RESULT.md`; `…-public-v1/RESULT.md`;
  `…/20260912-epistemic-questioner-public-v1/RESULT.md`.
- `analysis/critic-closure-contract-20260912/CONTRACT.md` and
  `packet-live-probe-v{1,2,3,4}/RESULT.md`.

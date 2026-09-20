# periodic-v1 result — control 21 levels, candidate 21. Δ = 0. Retain Duck alone.

Kernels `taaf-duck-periodic-v1-control` v1 and `-candidate` v1, tree `baf01a8`, both RUNNING
~20:57Z → COMPLETE ~23:05Z 2026-09-19 (benchmarks 21:07:52–23:03:39 and 21:08:13–23:04:48), on
separate RTX PRO 6000s. `readback.py control/kaggle-output candidate/kaggle-output` reproduces
every number below.

## Primary outcome and rule (precommitted in CARD.md)
| arm | total levels | episodes ≥1 | ≥2 | by game ft09 / lf52 / ls20 / sb26 / su15 | actions | s/turn (median) |
|---|---|---|---|---|---|---|
| control (Duck alone) | **21** | 16 | 4 | 7 / 3 / 2 / 5 / 4 | 2063 | 58 |
| candidate (periodic K=20) | **21** | 15 | 4 | 10 / 2 / 1 / 4 / 4 | 2131 | 69 |

Δ = 0; Δ_g = ft09 +3, lf52 −1, ls20 −1, sb26 −1, su15 0; Δ − best game = −3.
**Rule: Δ ≥ 8 and positive after removing the best game → not met. Retain Duck alone.**
No 7200 s confirmation.

Predictions in the card: control 20–30 (21 ✓), candidate within ±4 (0 ✓), "a wash" ✓.
Exposure forecast (~90 actions, ~4 attempts median): actual median 96 / 85 actions per
episode, delivered reviews per episode `{1:1, 2:3, 3:7, 4:2, 5:3, 6:2, 9:1, 13:1}`, median 3,
19/20 episodes with ≥2 — "multiple delivered reviews in a substantial fraction" held.

## Validity
- Escaped exceptions 0; fallbacks 0; no game ended by a firing; every episode with ≥21 actions
  had ≥1 attempt (ls20_p1, 28 actions, had 1).
- Turn latency 58 vs 69 s/turn median (19 % apart, inside the 25 % band; critic wall is 430 s =
  0.6 % of cap, so the gap is turn composition, not the critic's own time).
- **Second-wave shortfall, symmetric.** The kernel cap (8100 s from kernel start, including
  ~10 min vLLM startup) cut the second wave ~4 min early: 10/20 episodes per arm ran
  3341–3346 s (control) and 3363–3394 s (candidate) instead of 3600, state `cancelled`.
  Levels completed before the cut are kept, per the card; nothing rerun. One candidate episode
  overran to 3625 s (the critic's seconds-left + 20 s grace at the boundary). For the next run:
  experiment cap = waves × cap + 15 min *and* kernel cap = that + startup.

## Delivery funnel (candidate, 85 attempts)
attempted 85 → generated 85 (0 fallback) → included in an actor request 85 → actor turn
completed 85 → a subsequent action in that turn 79 (the 6 misses are final turns at the cap).
Uptake (probe executed next) 28/85. All causes `periodic`; 0 tree causes, as designed.

## Secondary (not decision variables)
TAAF mean score 3.88 vs 5.09; RHAE estimate against the real human baselines
(`submission-20260918/rhae/`) 4.22 % vs 5.64 %. Both come from ft09, where the candidate reached
L3 in 2/4 passes vs 1/4 and did so in fewer actions (63–87 vs 92–116). With Δ_g = +3 on one game
and −1 on three others, this is inside pass-to-pass variance and the card says a single-game gain
does not advance the candidate. It is recorded because the board metric weights depth, and ft09
is the game where Step 3's tree critic lost −4.

## Reading (narrow, as frozen)
Duck + this critic, exposed no more often than every ~20 actions, did not outperform Duck alone in
this screen; it also did not hurt (0.6 % wall, no failures). This rejects the bundle for the
submission path. It does not settle periodic-vs-tree (the tree arm was not run at 3600 s), does
not indict the critic's content, and does not say K=20 is wrong. The regressions (three games at
−1) look like ordinary trajectory variance, not a mechanism; the one positive (ft09 +3) is also
one game.

## What I would and would not do next
Would not: add mechanisms to this critic (memory, PASS, dedup) hoping to convert a tie — the
screen says the advisory channel at this dose is neutral. Would: if a critic is pursued again,
change the *question* being tested, not the plumbing — e.g. a one-shot review at a fixed point vs
nothing, or an actor-requested critic — each as its own two-arm screen with this card's rule.
Otherwise Duck alone stays the arm, and the lever that moves RHAE is depth, which none of the
critic variants tried has touched.

# Step 3 result — paired wave: incumbent 14 levels, critic 11. Submit the incumbent.

Kernels `taaf-duck-step3-incumbent` v1 and `taaf-duck-step3-critic` v1, both pushed 20:39–20:40Z,
both RUNNING 20:40Z → COMPLETE 21:30Z, games 40 min each (2 waves × 1200 s), run concurrently
on separate RTX PRO 6000s. Tree `c81f3d3`. `compare.py incumbent/kaggle-output critic/kaggle-output`
reproduces every number below.

## Primary metric (precommitted in CARD.md): total levels over 20 episodes
| arm | total levels | episodes ≥1 level | by game (ft09 / lf52 / ls20 / sb26 / su15) | s/turn (median) |
|---|---|---|---|---|
| incumbent | **14** | 12 | 5 / 0 / 2 / 4 / 3 | 69 |
| critic | 11 | 10 | 1 / 1 / 0 / 5 / 4 | 59 |

Decision rule: winner submits; tie or within 1 → incumbent. Gap is 3 against the critic.
**Submit the incumbent (Duck alone).** Prediction in the card was incumbent 6–9, critic within
±2, "a ≥3-level difference would surprise me": the incumbent beat its own prediction (ft09 p1
reached L3 in 25 actions) and the critic landed exactly at the surprise line, on the wrong side.

## Validity
- All 40 episodes `gave_up` at the 1200 s cap; none ended by a firing; 0 critic errors.
- Turn latency 69 vs 59 s (both slower than Step 1's 46 s at 600 s; longer contexts). 15 %
  apart, inside the 25 % band, and in the critic's favor — the gap is not substrate.
- Stall firings on the hard games: ft09 9, ls20 3, lf52 1. Trigger ran.

## Critic arm mechanics (50 firings)
50/50 parsed, 50/50 probes legal, critic wall 252 s = 1.0 % of cap. Causes: level_start 30,
stall_no_change 10, stall_no_progress 10. Uptake (probe executed as the next action) 21/49
offered; by game ft09 7/14, lf52 2/5, ls20 5/7, sb26 5/14, su15 2/9.

## Where the 3 levels went: ft09 (5 → 1)
`stall_no_change` at threshold 3 re-fires every 3 no-change actions on ft09, where clicks
legitimately produce no visible change (Step 1: ft09 p1 0 % board_changed). One episode took 6
firings in 15 actions, every one recommending a click on "the large charcoal square"; the actor
followed 7/14 of these. Duck alone explores ft09 more freely and reached L1/L3/L1 in three of
four passes. The critic did not ask a question that dislodged a hypothesis; it installed one
and repeated it. sb26/su15 (+2 for the critic) is the one place the question arm looks better,
and it is within pass-to-pass noise (su15 p3 incumbent scored 0 in 10 actions).

## What this settles and what it does not
- Settles, for the submission: Duck alone is the arm. The critic in its current form does not
  earn the slot.
- Does not settle whether a critic can help: n=4 per game; the sb26/su15 +2 and the ft09 −4 are
  both single-game swings. The clearest actionable defect is the re-firing cadence, not the
  question quality — `stall_no_change` should not re-fire on the same latched hypothesis, and
  probes should be excluded from the "already asked" filter's blind spot (same probe 4× on ls20
  in Step 2, same target 6× on ft09 here).

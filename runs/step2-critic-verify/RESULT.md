# Step 2 verification result — Duck + critic (question mode) on Kaggle: PASSED

Kernel `pranavagrawai/taaf-duck-critic-verify-step2` v1. RUNNING 18:42Z → COMPLETE 19:12Z.
Games 18:51:30 → 19:11:31 (~9.5 min vLLM startup, then both games to the 1200 s cap).
Output in `kaggle-output/`; `readback.py kaggle-output` reproduces the tables.

## Kill rule
No game ended by a firing; no critic error in either trace; `taaf_setup_env.json` carried
`DUCK_CRITIC_ENABLED=1`, `DUCK_CRITIC_MODE=question`. **Proceed to the paired wave.**

## Firings (9 total; first live stall firings ever)
| game | actions | levels | turns | firings | causes | critic wall | parsed | probe legal |
|---|---|---|---|---|---|---|---|---|
| sb26 p0 | 102 | **2**/8 | 28 | 5 | level_start ×3 (L1, L2, L3), stall_no_progress ×2 | 11.0 s | 5/5 | 5/5 |
| ls20 p0 | 106 | 0/7 | 28 | 4 | level_start ×1, stall_no_progress ×3 | 9.7 s | 4/4 | 4/4 |

Per firing 1.7–4.0 s, ~1.2–1.6k prompt + ~50–70 completion tokens, on the actor's own vLLM.
Critic wall = 0.9 % of the 2400 s of game time. Duck turn latency ~43 s (Step 1: ~46 s), so
the critic did not visibly slow the actor. `stall_no_change` never armed (longest no-change
run 0 and 2 < 3): on Kaggle every ls20 action changed the board, so that trigger is not the
one that matters; `stall_no_progress` re-fired every ~27 actions as designed.

## Uptake (probe executed as the very next action)
3/9: sb26 a0 `MOUSE(3,20)`, sb26 a50 `MOUSE(1,10)`, ls20 a28 `UP`. The critic block reached
the actor prompt once per firing (5 and 4 "Critic report" blocks in the transcripts).

## What this does and does not show
- Shows: the tree runs end to end on Kaggle with the flag on; stall firings happen live, parse,
  cost ~1/20 of a turn, and the actor sometimes follows the probe.
- Does not show benefit. sb26 reached L2 (Step 1 at 600 s: L1 only, 4/4) but this is n=1 with a
  doubled cap; the paired wave at 1200 s is the only fair comparison.
- Quality flaws to watch in the wave, not fix before it: on ls20 the probe was `UP` in 4/4
  firings and the three stall questions were one question reworded (the "already asked" list
  suppresses wording, not substance). On sb26 the L3 `level_start` probe `SPACE` was ignored.

## Next
Step 3 paired wave as frozen: both arms, 1200 s cap, `DUCK_CRITIC_MODE` sets the contrast.

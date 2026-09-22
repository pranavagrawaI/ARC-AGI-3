# qwen38-commit — how much smarter is 3.8, measured where speed does not bind

Frozen 2026-09-23 before push.

## Why the earlier arms could not answer this
Arms 1–3 ran at a 600 s per-game cap, where a Duck turn costs 46–56 s and every episode is
hard clock-limited (~10 turns). Under that cap, levels = capability x speed, and 3.8's ~20 %
slower turn dominates. Speed-free metrics from those same runs tell a different story:

| arm | lv/100 turns | lv/100 calls | actions->L1 |
|---|---|---|---|
| 3.6 | 3.4 | 2.0 | 16.0 |
| 3.8 xhigh | **3.5** | **2.4** | **11.3** |
| 3.8 medium | 3.1 | 2.1 | 13.0 |
| 3.8 low | 2.0 | 1.2 | 12.0 |

Per unit of thinking 3.8 @ default is >= 3.6, and it reaches L1 in ~30 % fewer actions.

## Question
At the submission cap (7200 s/game, 12x the probe cap — where all 17 of 3.6's levels landed by
minute 78 of 120, and 3.8's 20 % slower turn projects to ~94 min, still inside), does Qwen3.8
reach **more** levels than Qwen3.6 on the same 25 public games?

## Arm
Duck, 3.8 @ default `reasoning_effort` (xhigh — best of the three on every speed-free metric,
and the checkpoint as its authors intend it). 25 public games, 1 pass, 28-way, 120 min/game,
540 min envelope. No critic. Same tree, same guard, same cap as the 3.6 reference run.

## The comparison (already recorded, not re-run)
2026-09-18 commit run, Duck on 3.6, identical recipe:
**13/25 games with >=1 level, 17 levels, TAAF 1.90** -> public 0.67 (ref 56355127).

## Reading, frozen in advance
| 3.8 levels on 25 games | reading | action |
|---|---|---|
| **>= 20** | clearly smarter once unclocked | submit; expect > 0.67 |
| **17–19** | at least as smart; probe cap was the whole story | submit to measure the hidden set |
| **13–16** | comparable, noise-dominated at n=25 | do not submit; report the tie |
| **<= 12** | genuinely worse, not a speed artifact | do not submit; 3.8 closed for real |

Also recorded per game: levels, minute of each level, actions->L1, s/turn, tokens. The
per-game level-minute distribution answers directly whether 3.8's slower turns bind at 120 min
or not — the question the 600 s arms could not reach.

## Cost
~2.3 h GPU (0 submission slots). The commit run is a prerequisite for any submission anyway,
so this buys the capability answer at no slot cost, and the submit decision follows the table.

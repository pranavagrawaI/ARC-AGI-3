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

## Reading — AMENDED 2026-09-22 19:50Z, before the result, after measuring the noise floor
The original table below was too lenient. Duck's within-game level variance is ~0.35
(measured on two independent repeated-pass runs, `analysis/level-noise-floor.py`). Two runs
over the same 25 games share game difficulty but each carries its own stochastic noise, so
sd(delta) = sqrt(2 x 25 x 0.35) ~= **4.0–4.3 levels**. A 2-sigma claim needs delta >= 8–9.

| 3.8 levels | delta vs 17 | what it supports | action |
|---|---|---|---|
| **>= 26** | +9 | real advantage at 2 sigma | submit |
| **21–25** | +4…+8 | suggestive (1–2 sigma) | submit; a confirming run would be needed for a claim |
| **13–20** | −4…+3 | **indistinguishable** | submit only to buy the 110-game hidden-set sample; do not call it smarter |
| **<= 12** | <= −5 | suggestive of worse | do not submit |

ORIGINAL (superseded, kept for the record): >=20 clearly smarter / 17–19 at least as smart /
13–16 tie / <=12 worse. It called +3 "clearly smarter" when +3 is 0.7 sigma.

**What this run answers regardless of the noisy total**, because these are high-n within the run:
- the minute each level lands, and s/turn at 28-way -> does the clock still bind at 7200 s?
- actions-to-first-level at n~13 instead of n~3 -> per-action efficiency.

Also recorded per game: levels, minute of each level, actions->L1, s/turn, tokens. The
per-game level-minute distribution answers directly whether 3.8's slower turns bind at 120 min
or not — the question the 600 s arms could not reach.

## Cost
~2.3 h GPU (0 submission slots). The commit run is a prerequisite for any submission anyway,
so this buys the capability answer at no slot cost, and the submit decision follows the table.

## Submit step (only per the amended table; kernel version 1)
```bash
export KAGGLE_USERNAME=pranavagrawai
export KAGGLE_API_TOKEN="$(python3 -c "import json;print(json.load(open('/home/pranav/.kaggle/credentials.json'))['access_token'])")"
export KAGGLE_KEY="$KAGGLE_API_TOKEN"
kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -k pranavagrawai/taaf-duck-qwen38-commit -v 1 -f submission.parquet \
  -m "Duck on Qwen3.8-27B-FP8 (default reasoning_effort), 28-way, 120 min/game; commit run <N> levels on 25 public games vs Qwen3.6's 17"
```
Token valid to 2026-09-23 06:14Z. One submission per day; the 2026-09-22 slot is unspent.
The rerun replays the notebook against 110 hidden games with `KAGGLE_IS_COMPETITION_RERUN=1`,
which is the 4.4x larger sample this whole exercise is trying to buy.

Reference to beat on the board: **0.67** public RHAE (ref 56355127, Duck on 3.6, same recipe).

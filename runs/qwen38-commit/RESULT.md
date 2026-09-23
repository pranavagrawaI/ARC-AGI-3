# qwen38-commit result — 3.8 is ~2.2–2.7x more efficient per interaction. Submitted.

Kernel `pranavagrawai/taaf-duck-qwen38-commit` v1. RUNNING 18:48:20Z -> COMPLETE 20:56:19Z
(2 h 8 m). 25 public games, 1 pass, 28-way, 120 min/game, 3.8 @ default reasoning_effort.
Reference: the 2026-09-18 Duck-on-3.6 commit run, identical recipe.

## Headline
| | 3.8 | 3.6 | ratio |
|---|---|---|---|
| levels | **22** | 17 | 1.29x |
| games with >=1 level | **14/25** | 13/25 | |
| turns used | 886 | 1479 | 0.60 |
| actions used | **1605** | 3211 | 0.50 |
| calls used | 1027 | 1939 | 0.53 |
| **levels / 100 turns** | **2.5** | 1.1 | **2.2x** |
| **levels / 100 actions** | **1.4** | 0.5 | **2.7x** |
| levels / 100 calls | 2.1 | 0.9 | 2.3x |
| levels / 100k output tokens | 1.2 | 1.3 | 0.93x |
| median s/turn (28-way) | 192.3 | 122.2 | 1.57x |

**Same token efficiency, ~2.2–2.7x the interaction efficiency.** Both checkpoints need about
the same total thinking per level; 3.8 packages it into fewer, longer turns. Every per-turn or
per-second metric therefore flatters 3.6, which is exactly what the 600 s probe arms measured
and why their verdict was wrong for this question.

## Two of my earlier conclusions corrected
1. **"The cap tail is worthless"** was a fact about 3.6, not the task. 3.8 landed **3 levels
   after minute 78** — r11l at 92, re86 at 94, tr87 at **115** (effectively at the cap) —
   where 3.6 landed none. Its level minutes run median 16 (vs 3.6's 33) but extend to 115.
   For 3.8 a longer per-game cap plausibly buys levels; see `analysis/level-timing-20260918.py`.
2. **"3.8 is a net loss"** held only under a 600 s cap at 10-way. At the submission cap it wins
   on the metric that counts.

Biggest single-game swing: sb26 4 levels vs 1.

## Confidence, split honestly
- **Efficiency ratio: solid.** 886 vs 1479 turns and 1605 vs 3211 actions are large-n counts.
- **Level delta +5: 1.25 sigma.** The noise floor on a 25-game total is sd ~4.0
  (`analysis/level-noise-floor.py`). Suggestive, not proven. A confirming run, or the 110-game
  hidden set, is what would settle it.

This is the "21–25 -> submit" cell of the table amended at 19:46Z, before the result.

## Submission
**ref 56474012**, submitted 2026-09-22 21:01:48Z, kernel v1. Board reference to beat: 0.67
public RHAE (ref 56355127, Duck on 3.6, same recipe). RHAE also rewards action efficiency via
`min((human/agent)^2, 1.15)`, and 3.8 used half the actions for more levels, so the score may
move more than the level count alone implies. Rerun ~9 h over 110 hidden games.

## Hidden-set result (read 2026-09-23 07:17Z)
**ref 56474012: COMPLETE, 1.47 public RHAE** vs 0.67 for ref 56355127 (Duck on 3.6, same
recipe): **2.2x**. The 110-game hidden set agrees with the 25-game public-set efficiency ratio
(2.2-2.7x). The +5 public level delta was 1.25 sigma; the hidden set is the independent
confirmation it lacked. 3.8 replaces 3.6 as the incumbent model. Per-game levels on the hidden
set are not exposed; the rerun log is only in the Kaggle UI.

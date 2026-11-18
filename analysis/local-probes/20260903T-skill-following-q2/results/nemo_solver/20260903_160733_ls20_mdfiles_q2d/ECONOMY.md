# q2d time and action economy

Reasoning off. 0 reasoning tokens on 129 calls. Wall 1492 s. $0.86.
Waste = HUD-only ∪ blocked from `no-impact-offline/score.json` (35 env steps).
Room 0 = env actions 1–24 (includes the winning UP). Room 1 = 25–123.

## Where the 1492 s went

| | seconds | share |
|---|---:|---:|
| LLM `latency_s` | 1490 | ~100% |
| CodeAct cells | 19 | 1.3% (overlaps LLM wait) |
| Environment steps (123 actions) | 0.08 | 0% |
| `result.json` wall | 1492 | |

The game is free. The clock is OpenRouter.

Input 7,018,327 tokens, of which cache_read 5,584,432 (80%). Uncached input 1.43M. Output 96,829. All output is completion, not reasoning.

## Two rooms

| | room 0 (win) | room 1 (stuck) | all |
|---|---:|---:|---:|
| agent turns | 11 | 13 | 24 |
| env actions | 24 | 99 | 123 |
| of which waste | 3 | 32 | 35 |
| playfield | 21 | 67 | 88 |
| wall | 602 s | 876 s | 1478 s |
| wall / env action | 25 s | 8.8 s | 12 s |
| wall / playfield action | 29 s | 13 s | 17 s |
| output tok / env | 1511 | 612 | 787 |
| waste share of env | 12% | 32% | 28% |
| LLM calls | 72 | 57 | 129 |

Human baseline for room 0 is 22 actions. They spent 24, three of them timer-only UPs in the shaft, then finished. Then they spent **longer** (15 min vs 10) on a room they did not finish, at a third wasted moves.

## Probe then spray

Turns 0–5: batch size 1, 0 waste, 41 s/action, 2478 output tok/action. They were looking.

Turns 6–23: batch 2–10, 30% waste, 10.5 s/env, 701 tok/env. Depth rounds fall from 8–14 down to 1. Turn 21 (the death) is the cheapest turn: 8.4 s, 1 LLM call, LEFT×8.

Batching saved clock per env action and spent the action budget on walls.

## Against thinking-on (not a matched contrast)

q2c, thinking on, killed on EmptyContentError: 267k output (91% reasoning), 10 env actions, ~80 min, ~26,700 output tok/env. q2d is ~34× cheaper in output per env action *because thinking is off* and because it actually submitted. That is not evidence that thinking would fail at compilation. It is evidence that thinking-on at this route is a different, much slower actor, and we have not yet seen that actor finish a turn contract.

## What this does not buy

Truncating HUD-only steps would have saved 35 env actions (28%), mostly in room 1. It would not have compiled H29 into the LEFT×8, and it would not have refreshed `predict` after the level-up. Those are still missing machinery.

Turning reasoning back on would raise tokens and wall per turn. Whether it raises *completions* is unmeasured. This file does not license a thinking-on rerun by itself.

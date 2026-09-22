# qwen38-low result — gate FAILED on both conditions. 3.8 line closed; no submission.

Kernel `pranavagrawai/taaf-duck-qwen38-low` v1. Pushed 18:10:56Z, RUNNING 18:11:04Z (8 s queue —
the morning's 3 h 35 m queue had cleared), COMPLETE 18:28:56Z.

## Gate (frozen in CARD.md before the push)
| requirement | threshold | measured | |
|---|---|---|---|
| median s/turn | < 46.3 | **55.5** | FAIL |
| levels over 10 episodes | >= 4 | **2** | FAIL |

Neither condition met. Pre-run prediction was 20 % for both; it cleared neither.

## The reasoning_effort sweep, four arms, one recipe
| arm | tok/call | calls/turn | **tok/turn** | s/turn | decode | actions | levels |
|---|---|---|---|---|---|---|---|
| 3.6 | 514 | 1.67 | **858** | **46.3** | 18.7 | 201 | **4** |
| 3.8 xhigh | 941 | 1.44 | 1355 | 55.2 | 26.3 | 120 | 3 |
| 3.8 medium | 862 | 1.50 | 1293 | 56.0 | 25.2 | 233 | 3 |
| 3.8 low | 762 | 1.62 | 1234 | 55.5 | 24.8 | 164 | 2 |

**The knob is nearly inert on the quantity that matters.** Over its full range it cuts tokens per
*call* 19 % (941 -> 762) and the model takes 13 % more *calls* per turn (1.44 -> 1.62). Token-work
per turn falls only 9 % and stays ~44 % above 3.6. Turn latency is flat across all three settings
(55.2 / 56.0 / 55.5). Tool parsing stayed perfect throughout (124/124, 144/144, 165/165).

## Conclusion
Qwen3.8-27B-FP8 decodes ~33 % faster per stream and insists on ~44 % more token-work per turn
regardless of instruction. On a wall-clock-bound task it is a net loss against Qwen3.6, and the
loss is a disposition of the checkpoint, not a configuration. Per CARD.md the 3.8 line is closed:
no further checkpoint arms.

## Submission decision
User rule, set before the result: "3.8 @ low only if it passes the rule." It did not pass, so
**no submission was made** and the 2026-09-22 slot is unspent. The incumbent's arm is already on
the board at 0.67 (ref 56355127) and resubmitting it cannot beat that score.

Next candidate with a positive expected value is the second-attempt design
(`analysis/pass-variance.py`: 4.00 -> 4.62 of 5 games reaching a level), which needs its own card.

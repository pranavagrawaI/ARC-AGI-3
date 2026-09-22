# qwen38-medium result — the verbosity is the checkpoint, not the instruction. 3.6 stays.

Kernel `pranavagrawai/taaf-duck-qwen38-medium` v1. QUEUED 12:52Z -> RUNNING 15:28Z ->
COMPLETE 15:47Z. Same recipe as arm 1 and 3.6 Step 1; the only change from arm 1 is
`reasoning_effort=medium`, which renders a system prompt byte-identical to 3.6's.

## Three-way, identical recipe (10 episodes, 600 s, 10-way, same 5 games)
| | 3.6 | 3.8 @ xhigh | 3.8 @ medium |
|---|---|---|---|
| per-stream decode | 18.7 tok/s | 26.3 | **25.2** (+35 % over 3.6) |
| output tok/call (est.) | 514 | 941 | **862** |
| reasoning share | 59 % | 80 % | **76 %** |
| calls / turn | 1.67 | 1.44 | 1.50 |
| **median s/turn** | **46.3** | 55.2 | **56.0** |
| actions (10 eps) | 201 | 120 | 233 |
| actions changing the board | 87 % | 90 % | **67 %** |
| levels (10 eps) | **4** | 3 | 3 |
| L1 on sb26/su15 | 4/4 | 3/4 | 3/4 |
| tool calls parsed | 197/197 | 124/124 | 144/144 |

## Verdict by the rule frozen in CARD.md
The card's three branches were: (a) arm 1 slow and this arm ~= 3.6 -> the instruction is the
cost; (b) **both arms slow -> the checkpoint is slower on this workload; 3.6 stays**; (c) this
arm faster or deeper than 3.6 -> 3.8 is a candidate.

**Branch (b).** Removing the xhigh instruction moved output per call 941 -> 862 against 3.6's
514: the instruction explains **19 %** of the excess verbosity, the checkpoint **81 %**. Turn
latency did not improve at all (55.2 -> 56.0 s). Qwen3.8 deliberates far more than Qwen3.6 by
disposition, and on a wall-clock-bound task it spends its real 35 % decode advantage and more.

My pre-run projection (~514 tok/call at medium -> ~19.5 s/call) was wrong, and the run is why
it was worth measuring rather than reasoning about.

Secondary, worth noting: the medium arm executed the *most* actions (233) but only 67 % of them
changed the board, against 87 % for 3.6 — more motion, less traction. `ft09_p0` took 69 actions
with a 0 % board-changed rate.

## Consequence
Do not put Qwen3.8 in the commit run. The submission arm stays Duck on Qwen3.6-27B-FP8
(the incumbent, [[duck-is-the-incumbent]]). 3.8 remains *qualified as runnable* — it serves,
parses 144/144, and loads faster — so this is a disposition on the arm, not on the port.

Untested remaining lever: `reasoning_effort=low` (the template's brevity instruction) is the
only knob that directly targets the measured defect. It is a new question, not a rerun of this
one, and it needs its own card. It does not gate the submission.

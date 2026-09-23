# Step 1: learnings from the 3.8 commit run (in progress, 2026-09-23)

Data: `runs/qwen38-commit/live/kaggle-output/artifacts/*events.jsonl` (25 games, 1 pass, 7200 s).
Reproduce the yield table: `python3 runs/qwen38-learnings/yieldcount.py <artifacts dir>`.

## L1. Half of 3.8's turns end without an action (the 60 s turn yield)
Duck ends an analyzer turn after `LOCAL_ANALYZER_YIELD_SECONDS` (Kaggle default 60,
`framework/kaggle.py:120`; checked between model requests, `tool_agent.py:1823`).

| run | executed | yielded (turn_time_budget) | error/timeout | games with 0 actions |
|---|---|---|---|---|
| 3.6 commit (2026-09-18) | 1123 (76 %) | 329 (22 %) | 27 (2 %) | none |
| 3.8 commit (2026-09-22) | 449 (51 %) | 400 (45 %) | 37 (4 %) | ls20 (42/42 turns) |

History is preserved across a yield (`preserve_history`), but each yielded turn adds a fresh
user message (full frame render) into a 31,744-token context budget, which pushes older work out.
ls20 re-ran the same inspection crop for 42 turns and never called `action()`.
The yield was tuned for 3.6's shorter calls; 3.8 thinks longer per call, so it trips twice as often.
A public 3.6 fork (`yw8837/lb-1-17-...`, board 1.17) changes exactly this knob to 90 s.
Candidate fix: raise or disable the yield for 3.8; check whether the context budget binds too.

## L2. The board changes on 76-100 % of actions in every game
3.8 is not stuck in no-change loops. v1's `stall_no_change` critic trigger would almost never fire;
a v2 trigger must key on lack of progress (turns since last level, hypothesis churn), not
on board stasis.

## L3. Levels are carried by a few games
sb26 (4 levels) + ft09/lp85/re86/tu93/vc33 (2 each) = 14 of 22 levels; 11 games have 0.
Actions per game range 0-173; tr87 used 173 actions for 1 level (UP x125).

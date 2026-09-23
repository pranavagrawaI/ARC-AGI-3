# Step 1: learnings from the 3.8 commit run (2026-09-23)

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

## L4. The carried world model is nearly unused (both models)
`wmstats.py`: 3.8 carries no world model on 50 % of turns and changes it on 14 %; 3.6: 61 % / 17 %.
The actor's beliefs live in its thinking and its short pre-tool text, not in the channel Duck
carries forward. v1's critic read `actor_beliefs` from that channel, so it was mostly blind.
A v2 critic must read the actor's own words (`said` text and thinking tail; see `reasontrace.py`).

## L5. Why games stall: a taxonomy from reading every stuck game
Method: `reasontrace.py` digests each turn's stated beliefs + actions (`reason/<game>.txt`);
I read all 11 zero-level games and the post-L1 turns of ar25, lf52, s5i5, su15, ka59, tr87.
Classification is my judgment from the transcripts, n=1 pass per game, without the true
solutions — treat counts as rough.

| mode | zero-level games (primary cause) | signature in the transcript | lever |
|---|---|---|---|
| A. harness starvation | ls20 | 42 turns, 0 actions (turn yield) | yield / context budget |
| B. worldview lock under a draining budget | sp80, tn36, sc25, dc22 | one goal premise held while permuting its parameters; failed tests never reach the premise; an affordance left untouched; an anomaly left unexplained | **the questioner** |
| C. perception / own-code errors | cn04, m0r0 | readings reversed repeatedly; turns spent debugging its own analysis code | tooling (reliable diff/tracker helpers) |
| D. on track, out of wall clock | g50t, wa30, sk48 | mechanics decoded, plan forming when the 2 h cap hit | speed (yield fix, faster decode) |
| unreadable | bp35 | no stated beliefs on most turns; scattered clicks | — |

Concrete B cases, with the question that was never asked:
- **sp80**: premise "the blue bar is a block to place on a target"; tried 5 alignments with the
  in-game timer draining; never asked *why the bar always stops 4 rows short of whatever it
  approaches*. Game over at action 30.
- **tn36**: premise "the goal is to drop the piece into the cup" from turn ~15; ~8 T-configurations
  tried; for 38 steps never clicked the T *bars*, only the stems ("I've only ever clicked stems").
  Game over at action 61. Question never asked: *why must the piece move at all?*
- **sc25**: committed to "the legend describes the grid"; submitted it 3 times, each failure blamed
  on the tail state; discovered the cab's colors only at the end ("a colored block I hadn't tracked").

Post-level-1 stalls add two more modes:
- **E. lossy cross-level transfer**: at level-up the actor writes a hasty rule for what cleared the
  level and carries it. lf52 believed L1 was "reduce to 1 piece", declared L2 impossible, and only
  much later re-derived "reduce to 2" from history. ar25 and s5i5 also re-derived L1 late.
  Lever: at every level-up, have the actor establish exactly what the final action changed.
- **F. retries work, but the prompt discourages them**: tr87 hit the in-game step budget, the level
  reset, and it cleared L1 in 9 actions; tu93 took 2 levels after game overs. tn36 instead wrote
  "per the rules I stop acting immediately on game over" and idled. The prompt's "stop immediately
  if a result reports game_over" is read as "stop playing". 3.8 hits game over in 6 games (3.6: 16).

## What step 1 says about step 2/3
1. Speed and the turn yield touch every game (A, D, and the 45 % yield rate). Cheapest, broadest.
2. The questioner has a clear target (B: 4 of 11 zero-level games) and a clear trigger: the same
   goal premise surviving several failed variant tests, especially while an in-game budget drains.
   Its best questions in these transcripts are about the unexplained anomaly and the untouched
   affordance, not a replacement hypothesis.
3. Level-up review (E) and the game-over wording (F) are one-line prompt/host changes.
4. Perception tooling (C) is real but the least certain lever.

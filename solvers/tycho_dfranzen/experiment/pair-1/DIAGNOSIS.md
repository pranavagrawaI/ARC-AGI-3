# Pair-1 diagnosis, 2026-10-06 (read from collected outputs; no GPU)

Pair-1 ran to COMPLETE (RUNNING 19:10Z, COMPLETE ~20:23Z on 2026-10-05; quota 4,845 s -> 13,428 s used).
It ran the source-v7 candidate WITHOUT the wave-1 port (task statement, event guidance, GAME_OVER legality, border-aware batch guard).

## Outcome
| game | control levels / score | hybrid levels / score |
|---|---|---|
| sc25 | 6 / 76.1 | 5 / 71.0 |
| re86 | 7 / 77.8 | 6 / 58.3 |
| ls20 | 4 / 23.9 | 1 / 1.2 |
Frozen map: ls20 and re86 are losses beyond margin, sc25 inconclusive -> LOSS (stop + offline diagnosis).

## What it is NOT
- Not crashes. "TimeoutError: worker deadline waiting for decision" is the game_deadline stop label on every game (stop_reason game_deadline, worker rc -15).
- Not the GAME_OVER legality bug. sc25 hit GAME_OVER once (seq 177) and the Actor chose RESET; play resumed and two more levels followed.
- Not worse play per level where the hybrid got there. re86 actions per level: hybrid 21/36/47/72/74/65 vs control 21/37/47/63/63/65.
- Not output truncation (1 max_output hit in 3 games) or tool failure (tool error rate 2-10%).

## What it IS: tokens per level, hence wall time per level
Generated tokens per level, k (control | hybrid incl. Builder/Scribe):
- ls20 L0: 18 | 223. L1: 34 | 135 (unfinished).
- sc25 L0: 23 | 100.
- re86 L3-L6: 23/27/51/61 | 57/58/91/89. (L0-L2 the hybrid was faster: done at min 7.4 vs 18.4.)
Level completion minute: re86 hybrid leads to L5 (27.2 vs 26.9) then falls behind as per-level cost doubles. ls20 L0 at 37.5 vs 16.2 min.

Drivers, in order of size:
1. Actor deliberation per action. ls20: 145 Actor calls, only 27 (19%) call take_actions; ~2K completion tokens/call, 83-87% reasoning; per-action cost 6.6K tokens (control 615). In re86/sc25 ~60% of calls act and the cost per action is 1.3-1.6x the control's.
2. Large prompts. Mean Actor prompt 63-103K tokens; ls20 context saws 15K-108K with 9 evictions. Mostly cache hits (87-91%), so it costs decode speed and re-reading, not prefill.
3. Builder on ls20: 4 sessions, ~634 s (~10.5 of 60 min), 63K tokens. Every session handed off and reported (the lifecycle fix works). Each report then cost 9-13 Actor verification calls. Recommendation followed 1 of 2 times. No sign it sped up play.
4. Objective discovery. At ls20 L0 turn 16 the Actor asks the Builder to "keep the objective OPEN". The control's prompt states the task, flags and level-transfer rule. That fits drivers 1 and 3 but is not isolated by this run.

## Implication for wave-1
The port fixes target the GAME_OVER crash (did not bite here), objective discovery (plausible part of L0 cost) and guidance. Neither port nor lean bounds the dominant cost, Actor deliberation per action. Lean isolates driver 3 only.

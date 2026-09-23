# qwen38-fixes arm FM — result (kernel taaf-duck-qwen38-fixes-fm v1, 08:01Z -> 10:11Z)
Readback: `python3 readback.py fm/live/kaggle-output ../qwen38-commit/live/kaggle-output`

| | baseline (commit run) | FM (yield 300 + prompt fixes + MTP3) | card check |
|---|---|---|---|
| levels (top game dropped) | 22 (18) | **19 (17)** | 18–25 = indistinguishable |
| games with a level | 14/25 | 14/25 | |
| turns / actions | 886 / 1605 | 632 / 1652 | |
| turns acted / yielded | 51 % / 45 % | **72 % / 22 %** | yield < 20 %: **narrow miss** |
| ls20 | 0 actions, 0 levels | **121 actions, 1 level** | >= 1 action: **pass** |
| MTP acceptance / mean accepted length | — | **71 % / 3.13** | drafter works: **pass** |
| gen throughput at 20–25 running | 341 tok/s | 303–333 tok/s | >= 25 % faster: **FAIL** |
| KV cache use at >= 20 running | 79 % | **97 %**, requests waiting in 674 log samples (baseline 6) | |
Per game: ar25 +1, cn04 +1, ls20 +1, cd82 -1, ka59 -1, lp85 -1, **sb26 -3** (4 -> 1).
Prompt-fix uptake: 26 turns carried the game-over retry note; 13 responses wrote a verified
cross-level rule.

## Reading
- The yield fix does what it was for: acted turns 51 % -> 72 %, ls20 unstarved. Levels did not move
  (indistinguishable, driven by sb26 -3).
- MTP works as a drafter but buys nothing at 28-way: the KV cache saturates and requests queue, so
  fewer games decode at once. Part of the KV pressure may be the 300 s yield itself (longer
  multi-call turns carry more context). Arm F (same fixes, no MTP) separates the two.
- Decision: the next arms run **without MTP**. MTP stays available behind its flag for a lower
  concurrency or a smaller draft (num_speculative_tokens 1) if it is revisited.

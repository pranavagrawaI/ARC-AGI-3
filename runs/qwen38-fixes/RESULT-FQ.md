# qwen38-fixes arms F and Q — result (both 10:15Z -> 12:26Z, same day, same tree)
Readback: `python3 readback.py q/live/kaggle-output f/live/kaggle-output`

| | baseline | FM (+MTP) | **F** (fixes) | **Q** (F + questioner) |
|---|---|---|---|---|
| levels (top game dropped) | 22 (18) | 19 (17) | **25 (22)** | 23 (20) |
| games with a level | 14 | 14 | **16** | **16** |
| turns acted / yielded | 51 / 45 % | 72 / 22 % | 73 / 22 % | 70 / 23 % |
| turns, median s/turn | 886, 162 | 632, 264 | 635, 251 | 593, 282 |
| ls20 actions | 0 | 121 | 188 | 74 |
| KV use at >= 20 running | 79 % | 97 % | **79 %** | — |

Readings against the card (sd of a 25-game delta ~4):
- **F vs baseline: +3 (+4 with the top game dropped): indistinguishable to suggestive.** The fixes
  work mechanically (acted turns 51 -> 73 %, ls20 unstarved); sc25 +2 and ar25 +2 came with them.
- **F vs FM:** KV 79 % vs 97 % -> the saturation was MTP's, not the yield's. MTP stays off at 28-way.
- **Q vs F: -2 (-2 with the top game dropped): indistinguishable.** Per game: ar25 -2, sp80 -1,
  ft09 -1, ls20 -1, wa30 -1, cd82 +1, cn04 +1, ka59 +1, lp85 +1. Locked games: tn36 0/0, sc25 2/2,
  dc22 0/0, sp80 0 vs 1.

Questioner mechanics (all pass): 263 firings, 260 delivered (99 %), 0 errors, 24 near-repeats
caught and regenerated, 0 duplicate deliveries, 0.3 % of game wall (median 1.8 s/question).
Uptake: 145 of 235 answerable questions got an `Answer to observer:` line (62 %). Cost is not
the question but the answer: Q ran 7 % fewer turns and +12 % s/turn vs F.

What the transcripts show (tn36, sp80, sc25, dc22 ledgers): the actor engages ("good catch",
"fair point", plans to verify), but the questions mostly probe side premises (is the bar a timer
or a progress bar? why is the dot 2 cells?) rather than the load-bearing one; in tn36 no question
reached the T bars; one form dominates ("Why must X be Y?"). The questioner is working as built
and not yet asking the questions that change outcomes.

## Submission (user decision 2026-09-23: "Submit F, keep improving Q")
**ref 56493155**, submitted 2026-09-23 12:31:34Z, kernel `pranavagrawai/taaf-duck-qwen38-fixes-f` v1.
Board reference: 1.47 (ref 56474012, same model without the fixes). A single board score is ±50 %
(duck-score-noise-vs-shift), so this reads the fixes' direction on the hidden set, not their size.

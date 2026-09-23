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

## Lock-break preview on the first pair (questioner v2), `lockbreak.py`
| points | n | break rate (new target within 10 actions) | level within 10 actions |
|---|---|---|---|
| Q, at delivered questions | 232 | 54 % | 0.9 % |
| Q, at matched control points | 222 | 52 % | 0.9 % |
| F, at matched control points | 237 | 59 % | 0.8 % |
v2 did not make the actor reach for anything new more often than no question at all. Base rate ~55 %,
so the card's bar for v3 is Q3 question points >= F2 matched points + 10.

## Pair 2: F2 vs Q3 (questioner v3), both 13:05Z -> ~15:20Z
**Launch error (mine):** the cards asked for 2 passes, but the share notebook hardcodes `bm.n_passes = 1`
(kernel notebook line ~407), overriding the pickled `benchmark.passes: 2`. My dry-run check read the
pickle, not the notebook. Each arm ran 25 episodes, i.e. a second one-pass pair.

| | F (pair 1) | F2 (pair 2, identical config) | Q (v2) | Q3 (v3) |
|---|---|---|---|---|
| levels (top dropped) | 25 (22) | 20 (17) | 23 (20) | 20 (17) |
| games with a level | 16 | 16 | 16 | 13 |
| answered questions | — | — | 62 % | **78 %** (173/223) |
- **F vs F2 = 25 vs 20 with identical config**: the noise floor in the flesh (predicted sd of a delta ~4).
- **Q3 vs F2: 0.** Pooled over both pairs (50 episodes/arm): questioner 43 vs fixes-only 45.
- **Lock-break**: Q3 question points 57 % (n=221) vs F2 matched 63 % (n=230). Pooled with pair 1:
  questions 55.5 % (n=453) vs no-question control 61 % (n=467), about -5.5 points (~1.7 sigma).
  The card's bar was +10; the sign is the other way in both pairs.
Questioner mechanics clean again: 248/250 delivered, 0 errors, 0 duplicates, ~0.4 % of wall.

## Reading
The questioner as built (v2 and v3) does not raise levels and, in both pairs, the actor explored
*less* after a question than at matched points without one. The actor engages more with v3 (78 %
answer lines), but engagement goes into reflection and verification rather than into trying anything
new. The offline grade measured question quality; the live runs show that question quality in this
form does not convert into behaviour. Stop spending GPU on this formulation until something changes
what the actor does with a question.

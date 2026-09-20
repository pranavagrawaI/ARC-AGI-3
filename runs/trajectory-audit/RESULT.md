# Trajectory audit — where does available evidence stop becoming useful action?

Control arm (Duck alone, 3600 s), episodes chosen before reading reasoning: **ft09_p1** (L3, the arm's
best) vs **ft09_p0** (L0, same game, same wave, same 3600 s); plus **sb26_p0** (L1 at action 9, then
177 actions without L2). `digest.py` renders a transcript per attempt (time, thinking, code,
executed actions, outcomes); line numbers below refer to `periodic-v1/control/kaggle-output/transcripts/`.

## ft09_p1 — three levels, stopped by the clock
- **Last confirmed completion:** L3 at action 116, 22:06:45 (58 min 53 s elapsed).
- **What the transitions established:** L1 — only the charcoal-framed grid is editable; clicks toggle
  b/R; matching any source grid, XOR/AND/OR/majority/all-b/all-R all fail. L2 — one special cell, 7
  actions. L3 — special cells with an OO centre encode white→orange, those with an RR centre encode
  white→red (the encoding inverts by centre colour).
- **What Duck encoded:** step 26 (31 min in): "centre-cell white positions ↔ outer red positions",
  *verified against all three source grids in view* ("✓ ✓ ✓") before applying to the target. Step 32:
  the OO/RR inversion, derived after an OR-of-all-cells attempt and an all-orange attempt failed.
- **First consequential gap:** none that cost a level. L1 took 31 min of hypothesis cycling before the
  encoding idea; L3 took 21 min because the rule changed and the first two applications were guesses.
- **Stopping reason:** entered L4 with 67 s left; the episode was cut at the cap while inspecting the
  new layout. **Budget-limited progress**, not a reasoning failure.
- **Competing explanation:** none needed for the stop. For the 31-min L1: the search was unstructured
  (no enumeration, no use of the three free worked examples until step 26).

## ft09_p0 — zero levels, one misread premise
- **Last confirmed completion:** none. 105 actions, two GAME_OVERs (attempt counter exhausted), 60 min.
- **What the transitions established:** identical to p1 through step 13 (same toggles, same failed
  patterns). Duck's own tool output at line 1114–1119 (full rows 44–49) shows the bottom-right centre
  cell rows 48–49 = `WWgggg`.
- **What Duck encoded:** line 1408, inside a `[THINKING]` block: "Bottom-right center detail … Row 48:
  ggggWW" — a transcription that duplicates the bottom-left cell. Never printed `[44:50]` by code.
  Line 2483 → world model (line 2750): "BL and BR centers are identical."
- **First consequential gap:** step 14 (21:26, 20 min in). With the wrong centre, the centre-encoding
  hypothesis decodes to the BL pattern (`R b b / R ? R / b b R`), which had already failed at step 5.
  Step 15: "So the center pattern mapping theory is also wrong." **The correct hypothesis class was
  formed, tested once against a mis-transcribed target, and discarded.** The correct decode is
  `R b b / R ? R / R b b` (p1, step 26). 40 more minutes of XOR/XNOR/checkerboard/pairwise-XOR followed.
- **Stopping reason:** cap, after two counter resets.
- **Competing explanation:** Duck did not misread but believed the two centres were genuinely equal.
  Rejected: its own line-1118 dump is unambiguous and it never re-measured.
- **Smallest distinguishing check:** replay the step-14 prompt with one added instruction — "print
  the exact cells you are decoding with code before decoding" — and see whether the decode becomes
  `R b b / R ? R / R b b`. (Not run.)

## sb26_p0 — L1 in 9 actions, then 55 min of guessing on L2
- **Last confirmed completion:** L1 at action 9 (2 min). L2 layout: 7 colours on top, a red frame with
  3 slots + a fixed green cross, a green frame with 4 slots, SPACE to submit.
- **What the transitions established:** SPACE never changes the frames; it changes exactly one cell on
  row 53 (an attempt counter) and eventually GAME_OVER. So the game gives **no positional feedback**;
  the rule must be inferred from the layout.
- **What Duck encoded:** step 8: "After pressing SPACE, the colors changed! … Wait … SPACE didn't
  change anything visible." It never identified the counter (row 53 is outside its printed slice) and
  never recorded refuted arrangements. Hypotheses tried: top-row order, warm/cool split, empty frames,
  bottom-row order, "click the connector", ACTION7.
- **First consequential gap:** 15 submissions for 9 distinct arrangements; the first arrangement
  (top-row order) was resubmitted at actions 53, 93, 178 after "reconsidering from scratch" each time.
  Refuted states were not retained — this is a tool-only episode (0 notes).
- **Stopping reason:** cap, one GAME_OVER, final state = first state.
- **Competing explanation:** the rule is not inferable from the layout for this model at all, and the
  repeats are a symptom, not the cost. Plausible: even the 9 distinct arrangements were guesses.

## Prevalence check (both arms, HUD-masked exact (state, action) repeats)
Cells changed only by non-completing SPACE presses are masked. Repeats of an already-refuted pair:
control 37/2063 actions (1.8 %), candidate 37/2131; repeated *submissions* 4 and 6. Refuted-state
repetition is real (sb26, su15) but small at the action level; it is not the dominant cost.

## The one candidate failure
**Decoding from transcribed values instead of measured ones, then discarding the hypothesis class
on a single contradicted test.** Cited: ft09_p0 lines 1114–1119 (measurement) vs 1408 (transcription)
vs 2483/2750 (encoded premise) vs step 15 (rejection). Evidence X: the exact cell contents were in
tool output. Assumption Y: the recalled contents. Cost Z: the level, ~40 min, and the only hypothesis
that solves the game. The contrast episode p1 avoided it by (a) printing the cell with code (steps 7,
14) and (b) validating the mapping on the three in-view examples before acting (step 26).

A small change at this boundary — "decode only from values printed by code in this turn; check a
mapping against every reference instance on the board before applying it" — predicts p0's step 14
decodes to `R b b / R ? R / R b b`. The competing explanation (Duck would still reject after one
failed application) predicts no change. Distinguishable by replaying the saved step-14 prompt.

Not established by this audit: how often the transcription error occurs across episodes (needs
per-episode reading; the mechanical repeat detector does not see it), and whether the ft09_p1 L1
search would speed up. sb26's failure (no feedback channel, unstructured hypothesis search) is a
different failure and is left as recorded.

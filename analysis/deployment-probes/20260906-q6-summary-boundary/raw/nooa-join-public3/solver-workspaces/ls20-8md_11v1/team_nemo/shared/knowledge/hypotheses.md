## H1: Player is the {0,1} cluster
- Shape (5 cells), anchor (min row, min col): initial (31,20).
- Pattern:
    r0: .0.
    r1: 100
    r2: .1.
- Status: untested — test by moving and seeing the blob translate 1 cell.

## H2: Actions UP/DOWN/LEFT/RIGHT move player 1 cell, clamped at bounds
- Status: untested. Deciding test: submit LEFT once; expect blob at (31,19) with
  old cell (31,22)/(32,22)/(33,21) reverting to 3 (surround color), and new cells
  (31,19)/(32,19... wait, need full 5-cell footprint at new position) showing
  the blob pattern. If walls, blob stays (no_impact).

## H3: '3' is the walkable/territory color; '4' is open/void; '5' is wall/structure
- The blob sits inside a big 3-colored territory (rows 8-49, cols 14-53).
- Status: provisional. If a 5-cell slide into 4-space breaks the pattern, 4 is walkable too.

## H4: Level goal unknown — some object (the 'c' 12-colored target at rows 45-46 cols 34-38?
  or the 9-cluster at rows 47-49?) must be reached/pushed. The 9s at rows 55-60 cols 3-8
  in a 5-frame look like a second entity or a hazard.
- Status: untested.

## Known geometry (initial state, level 0)
- Big 3-territory: rows 8-49, cols 14-53, with internal 4-holes:
  - hole A: rows 9-15, cols 33-39 (a 7x7 hole with a 555-annulus and 9-blobs inside)
  - hole B: rows 30-34, cols 25-28 (a 5x4 4-hole in the middle)
  - region around rows 45-49: c-target at (45-46, 34-38) + 9-block at (47-49, 34-38)
- Player at (31,20) inside territory, near hole B.
- Bottom-left 5-box (rows 53-62, cols 1-9) contains a 9 blob (rows 55-60, cols 3-8).
- Bottom band rows 61-62: 'b' (11) strip cols 13-54; '8' 2x2s at cols 56-63.
- '5' appears as frame/wall everywhere on the border-ish.

## H1: {0,1} cluster at (31,20) is the player
- Status: CONTRADICTED as to LEFT motion (cluster unchanged, 52px moved elsewhere).
- Re-test: try DOWN. If the cluster slides to (32,20) it's the player.

## H5: one of the 9-clusters is the player (moved by LEFT)
- The 9-blob at (47-49,34-38) is in an open 3-territory directly below the c-target.
- The 9-blob at (55-60,3-8) is inside a 5-frame at bottom-left.
- Status: untested. Deciding test: LEFT again or DOWN; compare which 9-blob moved.


## Hypothesis: b-strip wraps around when fully consumed
- Status: observed once (step 42->43)
- Evidence: b-strip at col 54 (1 cell) at step 41, GONE at step 42, reappears at cols 13-54 at step 43
- Deciding test: consume the b-strip again and see if it wraps to 13-54

## Hypothesis: c-block jumps when b-strip wraps
- Status: observed once (step 42->43)
- Evidence: c-block jumped from (row 25, col 49) to (row 45, col 34) at the same step b-strip wrapped
- Deciding test: consume b-strip again and see if c-block jumps

## Hypothesis: 8-blocks shrink when b-strip wraps
- Status: observed once (step 42->43)
- Evidence: 8-blocks went from cols 56-63 (12 cells) to cols 56-60 (8 cells) at step 43
- Deciding test: consume b-strip again and see if 8-blocks change


## Hypothesis: TOP_WALL=15 for c-block
- Status: confirmed (step 50: c_top=15, UP blocked, no_impact)
- Evidence: c_top went 35->30->25->20->15 over 4 UP steps, then blocked
- Note: earlier c_top was stuck at 25 (steps 23-42) when c_left=49 — possibly a different constraint or the b-strip wrap reset

## Hypothesis: c-block blocked = action is no-op (b_left does NOT advance)
- Status: confirmed (step 50: UP at c_top=15, b_left stayed 19)
- Evidence: no_impact outcome, b_left unchanged

## Hypothesis: c_top=25 was a soft clamp when c_left=49
- Status: untested
- Evidence: c_top stayed at 25 for 20 steps (23-42) while c_left=49, but c_top went past 25 (to 15) when c_left=34
- Deciding test: move c-block to c_left=49 and try to go above c_top=25


## Step 74: DOWN blocked at c_top=45->50 (rows 50-54, cols 34-38 are color 4)
The c-block is now surrounded by color-4 cells below. Fixed walls don't capture this.
The model v13 uses fixed walls as approximation but needs grid-level collision checks.

## Open question: what happens at b_left=54?
b_left=42 now, 12 steps to reach 54. The b-strip wrap-around at step 42-43 is a key
unresolved mechanic. Need to observe what happens when b_left exceeds 54 again.


## Step 87: b-strip consumed and gone
- b_left reached 54 (single cell at col 54), then RIGHT consumed it: b_left=None
- 8-blocks unchanged: still 8 cells (cols 56-57, 59-60, rows 61-62)
- Model v17 matched: predicted b_left=None correctly
- NEXT: model predicts b-strip reappears at b_left=13, b_right=54 (full 42-cell strip)
- This is the SECOND observation of b-strip wrap-around (first was at step 42-43)
- Key question: what is the WIN CONDITION? The c-block has been moving around for 87 steps with no level completion.


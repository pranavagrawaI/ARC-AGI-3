## Turn 0 (step 0 -> step 1)
Action: LEFT
- outcome: move, diff_pixels=52, level_completed=false
- Settled grid is byte-identical to step 0 (num changed cells: 0).
- No event_frames available.
- The {0,1} player cluster did NOT shift left. The 52 changed px are in some
  sub-action animation that we cannot see because event_frames is empty.
- H1 (player = {0,1} blob, LEFT slides it) is CONTRADICTED: the cluster's anchor
  stayed at (31,20).
- New tentative reading: maybe LEFT does something else (e.g., pushes a
  neighboring object, or the "player" is one of the 9-clusters, or the {0,1}
  blob IS the player but LEFT is blocked and the 52px is an "attempt" animation).

## Carry-Forward
- Player identity unclear: candidates = {0,1} cluster at (31,20), 9-blob at
  (47-49,34-38), 9-blob at (55-60,3-8), or one of the 'b' or '8' tiles.
- Next experiment: DOWN (a direction with a clear open corridor below the {0,1}
  blob inside the 3-territory). If the blob translates to (32,20) -> it is the
  player. If the blob doesn't move but something else changes, re-examine.

## Turn 1 (step 1 -> step 2)
Action: DOWN
- outcome: move, diff_pixels=2, level_completed=false
- Changed cells: (61,14): 11->3, (62,14): 11->3
- The b-strip (color 11, rows 61-62) lost its col-14 cells.
  Step1: b-strip cols 13-54 (42 cells)
  Step2: b-strip cols 15-54 (40 cells) -- wait, (61,13) and (62,13) are now 3 too!
- So the b-strip went from cols 13-54 to cols 15-54? No:
  Step1: row61 col13=11, col14=11 -> Step2: col13=3, col14=3
  So 2 cells lost on the left edge. b-strip: 13-54 -> 15-54 (40 cells).
- The {0,1} blob at (31,20) did NOT move.
- H1 (player={0,1} blob) is FURTHER CONTRADICTED.

## Updated Understanding
- LEFT (step0->1): 52px animation-only, settled grid unchanged.
- DOWN (step1->2): 2px permanent, b-strip left edge shifted right by 2.
- The b-strip is at rows 61-62, initially cols 13-54.
- The 8-blocks at cols 56-63 are unchanged.
- The {0,1} blob is static decoration (or a very different kind of entity).

## Turn 3 (step 3 -> step 4)
Action: LEFT
- outcome: move, diff_pixels=52
- Prediction: MATCHED (c_left 24->19, b_left 16->17 — both correct)
- c-block now at cols 19-23, b-strip at cols 17-54
- Model v3 (LEFT: c_left -= 5, b_left += 1) is CONFIRMED for LEFT.

## Current state (step 4)
- c-block: rows 45-46, cols 19-23
- 9-block: rows 47-49, cols 19-23 (tracks c-block)
- b-strip: rows 61-62, cols 17-54
- 8-blocks: rows 61-62, cols 56-57, 59-60, 62-63 (unchanged)
- {0,1} blob: rows 31-33, cols 20-22 (static, unconfirmed purpose)
- b-strip left edge has 3 '3' cells (cols 14,15,16) before it starts at 17

## Confirmed dynamics
- LEFT: c_left -= 5, b_left += 1 (c/9 block shifts 5 cols left, b-strip left edge +1)
- DOWN: b_left += 2 (b-strip left edge +2, c/9 block unchanged)
- UP/RIGHT: untested

## Turn 4 (step 4 -> step 5)
Action: RIGHT
- outcome: move, diff_pixels=52
- Prediction: MISMATCH (predicted no change; actual c_left 19->24, b_left 17->18)
- RIGHT = inverse of LEFT for c/9 block (c_left += 5), but b_left STILL +1 (not -1)
- Model v4 repaired: RIGHT: c_left += 5, c_right += 5, b_left += 1 (verified vs actual: True)

## Confirmed dynamics (level 0)
- LEFT:  c_left -= 5, c_right -= 5, b_left += 1
- RIGHT: c_left += 5, c_right += 5, b_left += 1
- DOWN:  b_left += 2 (c/9 unchanged)
- UP:    untested (predict: no change)
- b_left is a MONOTONIC COUNTER: +1 per LEFT/RIGHT, +2 per DOWN.
  It has gone 13 -> 14 -> 15 -> 16 -> 17 -> 18 across steps 0-5.
  This looks like a progress/timer bar — or the b-strip's left edge advances
  toward the 8-blocks at col 56. When b_left reaches 56, the b-strip vanishes
  (reaches the 8-blocks). That may be the WIN condition (or a loss).

## Open question
- What does UP do? If UP is b_left -= 1 (inverse of DOWN's b_left += 2?), that
  breaks the monotonic-counter theory. Test UP next.
- Game goal: unknown. b-strip advancing toward 8-blocks is the main narrative so far.


## Turn 19 (step 44): b-strip wrap-around discovered, c-block row tracking added

**Key discovery:**
- b-strip at step 41: 1 cell (col 54). Step 42 (UP): GONE. Step 43 (UP): reappears at cols 13-54 (42 cells).
- The b-strip DISAPPEARS when fully consumed (b_left reaches 55 = b_right+1), then WRAPS AROUND on the next step.
- At step 43, the c-block also jumped: from (row 25, col 49) to (row 45, col 34). This is a +20 row, -15 col jump.
- At step 44 (UP): c-block moved to (row 40, col 34) — only -5 rows, consistent with normal UP.

**Updated dynamics (v9):**
- All directions: b_left += 1
- LEFT/RIGHT: c_left ± 5 (clamped 19-49)
- UP: c_top -= 5 (was not tracked before)
- DOWN: c_top += 5
- b-strip wraps around when fully consumed

**Unresolved:**
- Why did c-block jump +20 rows and -15 cols at step 43? Is this related to b-strip wrap?
- Does the b-strip always wrap to 13-54, or does the left edge depend on something?
- What is the winning condition?

**Current state (step 44):**
- c-block: rows 40-44, cols 34-38 (c_top=40, c_left=34)
- b-strip: cols 14-54 (b_left=14, b_right=54)
- 8-blocks: cols 56-60, rows 61-62


## Turn 21 (step 50): c-block TOP_WALL=15 confirmed, no-op blocking confirmed
- c_top: 35->30->25->20->15 over 4 UP steps, then blocked at step 50
- b_left: 15->16->17->18->19, stayed 19 at step 50 (no-op)
- Updated model v11: TOP_WALL=15, BOTTOM_WALL=59, no-op when blocked
- Current state: c_top=15, c_left=34, b_left=19


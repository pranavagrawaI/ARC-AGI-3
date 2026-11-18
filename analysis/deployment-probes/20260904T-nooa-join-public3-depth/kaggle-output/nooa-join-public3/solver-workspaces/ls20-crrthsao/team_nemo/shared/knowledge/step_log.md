# Step log
## turn 0 (step 0)
Initial grid: big 3-region rows 8-49; 4 bg; 5 border; small 0/1 cluster (31-33,20-22); bottom status band rows 61-62 (11 gray bar, 8 orange marks). Submitted UP.
## turn 1 (step 1)
After UP: 0/1 cluster UNCHANGED. 9-block (was rows 45-47 x 34-38) -> rows 42-44; 12-block (45-46 x34-38) -> rows 40-41 x34-38 (moved up ~5); green 3 +2 cells. So UP moves the 9/12 object up; 0/1 cluster is static (a marker/target?).
## Turn 4 (step 4): RIGHT confirmed 5-col horizontal move. Object back at (45,34)/(47,34).
- Model: 5 cells/step movement. Target at (31,21). Need ~3 UP + ~3 LEFT.
- Plan: move UP and LEFT toward target.

## Turn 5 (steps 5-6): UP, UP batched successfully. obj_r: 45→40→35. Predictions CORRECT.
- Batching now open. Moving toward target (31,21).

## Turn 6 (steps 7-9): 3x LEFT blocked by wall! Object stayed at (35,34).
- Wall (blue=4) at cols 29-33 blocks LEFT from object at col 34.
- Wall has gaps at rows 25-29 and rows 45-49.
- Strategy: go UP to rows 25-29 (gap), then LEFT through the gap.
- Need: UP, UP (35→30→25), then LEFT through gap.

## Turn 7 (steps 10-11): UP, UP successful. obj_r: 35→30→25.
- Now at row 25, in the wall gap (rows 25-29). Can move LEFT through the gap.

## Turn 8 (steps 12-14): LEFT x3 through wall gap successful! obj_c: 34→29→24→19.
- Now at (25, 19). Target at (31, 21). Need DOWN + slightly RIGHT.
- Predictions all correct. Batching working well.

## Turn 9 (step 15): DOWN → obj at (30,19). TARGET (0/1) COLLECTED/DISAPPEARED!
- Object overlapped target at (31-33, 20-22). Target vanished.
- 11-bar timer: 28 cols wide (was 42 at step 0). ~1 col/step. ~27 steps remaining.
- Level NOT_FINISHED. Need to find the actual win condition.
- 8-marks unchanged. Walls unchanged.

## Turn 10 (step 16): RIGHT → obj at (30,24). Target REAPPEARED (static marker confirmed).
- 0/1 cluster is NOT a collectible. It's a STATIC goal marker that hides when object overlaps.
- Timer: 11-bar width 27 (step 16). ~1 col/step. ~27 steps remaining.
- Object at (30,24). Target at (31-33, 20-22).
- Overlapped target at (30,19) but level didn't complete.
- New hypothesis: need to reach a different location (exit?) after touching the target.
- 8-marks: 3 pairs at cols 56-57, 59-60, 62-63 (rows 61-62). Static.

## Turn 11 (step 17): DOWN → obj at (35,24). Prediction CORRECT.
- Timer: 11-bar width 26 (step 17). ~1 col/step.
- No special cells in playfield other than object and target.
- Found 9-cell pattern in bottom-left (rows 55-60, cols 3-8): looks like "H" or "8" shape.
- This pattern is OUTSIDE the playfield (border/status area).
- Playfield: rows 8-49, cols 4-52. Green (3) with blue (4) walls.
- I'm stuck on the win condition. Overlapped target at (30,19) but no level completion.
- Decision: RESET and try a different strategy.

## Turn 12 (step 18): RESET → obj back at (45,34). Timer reset to 42 cols.
- CRITICAL DISCOVERY: 9-cell pattern at rows 11-13, cols 35-37 (INSIDE the yellow wall at top).
- Pattern (3x3): 999/559/959. Looks like a letter/number.
- Object starts on RIGHT side of wall (cols 34-38). Target on LEFT side (cols 20-22).
- Plan: go UP to the 9-cell pattern and see if it triggers anything.

## Turn 17 (step 43): LEFT x6 + UP x3 → obj at (30,19). Target HIDDEN (overlapping).
- Level STILL NOT_FINISHED. Second time reaching target from a different path.
- Timer: width 19. ~19 steps remaining.
- Win condition is NOT "overlap target." Must be something else.
- New idea: maybe need to approach from a specific direction, or stay at target until timer expires.


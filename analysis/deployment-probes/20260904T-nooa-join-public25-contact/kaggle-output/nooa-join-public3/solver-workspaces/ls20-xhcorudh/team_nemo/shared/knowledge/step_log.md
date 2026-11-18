
## Turn 0 — initial analysis (step 0, level 0)
Grid: playfield rows 0..51 (64 cols), HUD rows 52..63.

Structure decoded:
- 4 = background. 5 = left border band (cols 0-3 rows 0-51, full-height left wall).
- 3 = maze walls (enclosed rooms). Rooms: top room rows 8-15 cols 32-40; middle room
  rows 16-49 cols 14-53 (with sub-walls at rows 16-24 cols 34-38? actually a 5-wide
  column room rows 17-24 cols 34-38; and bottom room rows 40-49 cols 19-53).
- 9 = a small 5x5 sprite (entity) at rows 47-49 cols 34-38 (mean 37,36 in 1st obs?
  NO: first obs showed 9 in top room rows 11-13 cols 35-37 AND bottom room 47-49 cols 34-38
  AND HUD 55-60. After load state (same grid? latest_state at turn 0 step 0):
  9 cells: 21 in playfield rows 11-49 cols 34-38 -> TOP sprite rows 11-13 cols 35-37 (7 cells)
  + bottom sprite rows 47-49 cols 34-38 (14 cells). Wait count 7+14=21. OK.
- 12 = 2x5 bar rows 45-46 cols 34-38 sitting directly ABOVE the bottom 9 sprite.
- 0/1 tiny 5-cell glyph rows 31-33 cols 20-22 (mid-maze).
- HUD rows 52-63: 5x... left box (cols 1-9) contains a 9-color pattern (rows 55-60 cols 3-8)
  = a HUD copy of the sprite? middle bar rows 61-62 cols 13-54 color 11; right box rows 61-62
  cols 56-63 color 8.

Hypotheses:
- H1: two 9-sprites, one must reach goal; 12 bar = goal pad / portal.
- H2: 0/1 glyph = level number or objective marker.
- H3: HUD left box mirrors current entity state; 11/8 bars are HUD meters.
- H4: 5 = wall band on left; 3 = walls; 4 = floor.

Test plan: try moving. First move RIGHT (entity nearest to col 38-53 wall? bottom sprite at
cols 34-38, right wall of its room at col 53; top sprite cols 35-37 in room cols 32-40).
Need to figure out which sprite (if any) we control. Move UP/DOWN/LEFT/RIGHT one step and
diff to see which 9-sprite (or the 12 bar) moves.


## Turn 1 — RIGHT executed (step 0 -> 1)
Prediction: move. OUTCOME: move. Correct.

Key finding: the 5x5 block rows 45-49, cols 34-38 (12-bar rows 45-46 + 9-sprite rows 47-49)
moved RIG**T as ONE unit to cols 39-43. The 9-sprite and 12-bar are a single fused entity.
The top 9-sprite (rows 11-13, cols 35-37) did NOT move — it is a separate static/other entity.
0/1 glyph at (31-33, 20-22) unchanged.

So the controllable entity is the fused 12+9 block at rows 45-49 (5 rows tall, 5 cols wide).
The 12-bar sits ON TOP of the 9-sprite — 12 = "roof" or "carry" indicator.

New position: 5x5 at rows 45-49, cols 39-43. Center (47, 41).

Updated hypothesis:
- We control the fused entity (9-sprite with 12-roof).
- The 0/1 glyph might be the GOAL (a tiny target). It's at rows 31-33, cols 20-22 (3 cells).
- The top 9-sprite (7 cells, rows 11-13) is a distractor or another entity.
- The 12-roof might disappear when we reach the goal (or it IS the goal indicator).

Goal hypothesis: reach the 0/1 glyph at (31-33, 20-22).
Current pos: rows 45-49, cols 39-43. Need to go UP and LEFT.

Maze structure: the entity is in the bottom room (rows 40-49, cols 19-53).
Walls: 3-color. Let me check what's to the LEFT and UP of the entity.


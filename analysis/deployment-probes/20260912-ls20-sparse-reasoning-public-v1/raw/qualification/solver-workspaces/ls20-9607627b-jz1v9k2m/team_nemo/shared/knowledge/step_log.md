
## Turn 0 (step 0, level 0)
Initial observation, no actions yet.
- 64x64 grid. bg=4. Structures: 3=floor region (rows 8-49, cols 14-53, 892 cells),
  5=outer walls, 9=pixel glyphs (three 9-pixel clusters in small 5-boxes at rows
  9-16 cols 29-35, a 3x5 9-block at rows 47-49 cols 34-38, and a bottom-left
  5-box with 9s at rows 55-60 cols 3-8), 12=2x5 block rows 45-46 cols 34-38
  (sits on top of the 9-block), 11=bottom bar rows 61-62 cols 13-54,
  8+5=bottom-right corner rows 61-62 cols 56-63.
- Player-like sprite at rows 31-33 cols 20-23: 0 at (31,22),(32,22),(32,23);
  1 at (32,21),(33,22) — a small 2-color mark on the 3-floor.
- Bottom-left 5-box (rows 53-62, cols 1-10) contains 9-glyph pixels — looks like a
  separate small screen / minimap or glyph panel, NOT walkable (it's off the 3-floor).
- Actions: RESET UP DOWN LEFT RIGHT. No CLICK.
- Objective unknown. Hypotheses: (a) move sprite to the 12/9 block (rows 45-49,
  cols 34-38); (b) the 9-clusters are a target pattern to collect/match;
  (c) it's a maze to navigate.
- No helper model yet. Batching not earned.


## Turn 1 (step 1, level 0) — after RIGHT
Measured (last_transition):
- 52 cells changed, bbox rows 45-62 cols 13-43.
- Sprite (0 at r31-32 c21-22; 1 at r32-33 c20-21): UNCHANGED (static landmark).
- The 5-wide block (rows 45-46 = 12, rows 47-49 = 9) shifted RIGHT by +5: cols 34-38 -> 39-43.
- Bottom bar (rows 61-62, color 11): lost col 13 (11->3), i.e. left end advanced +1.
- diff_pixels 52, outcome 'move'.
Hypotheses (unconfirmed):
- The 12/9 block is a controllable object; RIGHT slides it +1 tile (5px).
- The 0/1 sprite is a fixed decoration (maybe a "goal marker" or "start marker").
- The bottom bar is a second object (or a meter) whose left end moved +1.
Open questions:
- Does LEFT reverse this exactly? (reversible?)
- Is the block the player, or does the sprite push it?
- What is the win condition? (align block to something? reach a location?)
Next: probe LEFT to test reversibility.


## Turn 2 (step 2, level 0) — after LEFT
Measured (last_transition): 52 cells changed.
- Block (rows 45-46=12, rows 47-49=9, 5-wide): before cols 39-43 -> after cols 34-38.
  So LEFT moved it -5. FULLY REVERSIBLE with RIGHT. ✓ Clean single-tile slide.
- Bar (rows 61-62, color 11): before left edge col 14 -> after col 15. Lost col 14.
  Bar left edge: step0=13, step1(RIGHT)=14, step2(LEFT)=15. Monotonically increasing
  by 1 on EACH move regardless of direction. => a move counter / odometer (shrinks from
  left as moves accumulate). NOT tracking the block.
- Sprite (0/1) unchanged (static landmark).

CONFIRMED rules so far:
- RIGHT: block += (0,+5) [dx=+5, dy=0]; LEFT: block += (0,-5).
- Each move increments bar-shrink by 1 from left.
- Sprite is static.
Open questions:
- UP/DOWN: does the block move vertically? The block is at rows 45-49 (near bottom of
  the 3-floor which spans rows 17-49). Is there room above? Floor rows ~17-49.
- Win condition: align block somewhere? The static 0/1 sprite, the top 5-box with 9
  glyphs, the bottom-left 5-box with 9 glyphs — are these targets/patterns?
- The block has 12 (top 2 rows) + 9 (bottom 3 rows). The top 5-box (rows 9-16) contains
  a 9-glyph pattern. Maybe the block is a "tile" to be placed into a matching slot.

Next probe: UP — test vertical movement of the block (does it slide up a tile?).


## Turn 3 (step 3, level 0) — after UP
Measured: block rows 45-49 -> 40-44 (dy=-5). 12 at 40-41, 9 at 42-44, cols 34-38.
Bar: left edge 15 -> 16 (lost col 15). Monotonic counter confirmed.
- 52 cells changed, all within block+bar. Reversible slide confirmed in 4th axis.
- Sprite static.

CONFIRMED mechanics:
- Block (5x5: 2 rows 12 + 3 rows 9) is the controllable object.
- RIGHT dx=+5, LEFT dx=-5, UP dy=-5, DOWN likely dy=+5. One tile per action.
- Bar (color 11) left edge +1 per move = monotonic move counter.
- Sprite (0/1) static landmark.
- Floor is 3-colored tiles; walls are 5 (the small 5-boxes at top rows 9-16 and the
  bottom-left 5-box rows 53-62) sit inside/around the floor.

Next: probe DOWN to confirm dy=+5. The block at rows 40-44 has room below to 49.


## Turn 4 (step 4) — after DOWN
DOWN confirmed: block rows 40-44 -> 45-49 (dy=+5). Bar 16->17. All 4 directions
move the block exactly 1 tile (5px), fully reversible. Sprite static.

## World model (code-ready)
- Block anchor = top-left of 5x5, colors 12(2 rows)+9(3 rows). Current anchor (45,34).
- Floor = color 3 tiles. Walls = color 5.
- Static target marker (0/1) at rows 31-33, cols 20-23.
- Move counter bar (color 11) left edge: step n -> col (13+n).
- Objective: move block to align with the static 0/1 marker (rows 31-33 cols 20-23).
  Target anchor approx row 30, col 19 (marker spans rows 31-33; block center row 47
  currently; need block up ~16 rows and left ~15 cols). Marker is 2x4, not 5x5, so
  "align" likely = block center/near the marker region.

Plan: build helper world_model.py (encode block pos + bar + barrier info; predict
slides). Then use it to drive block toward the marker. Marker at (31-33, 20-23).
Block currently (45-49, 34-38). Delta to reach marker rows: block top row 45 -> ~30
= 15 up. Cols 34 -> ~19 = 15 left. But need to check floor/wall path.


## Turn 5 (step 5) — after UP
Block rows 40-44 (top 40), cols 34-38. Bar left 18. Prediction MATCHED (frozen model sha
fbd0a818...). batch_allowed = True now.
Block at top-left (40,34). Target marker (0/1) at rows 31-33, cols 20-23.
Plan: batch LEFT + UP to approach. Block needs to go LEFT (col 34 -> ~20) and UP (row 40 -> ~30).
Reachable block top-left positions (multiples of 5 offset from 34 col, 40 row):
col: 34,29,24,19 ; row: 40,35,30.
Target block top-left to cover marker rows 31-33 cols 20-23: top (30,19).
Path: UP,UP (40->35->30) and LEFT,LEFT,LEFT (34->29->24->19) = 5 moves.
But do it carefully: approach marker and watch for objective response.
I'll batch a few moves now that batching is earned.


## Turn 6 (step 8) — batch UP,UP,LEFT,LEFT,LEFT
- UP (step 6): block (40,34)->(35,34), diff 52. matched.
- UP (step 7): block (35,34)->(30,34), diff 52. matched.
- LEFT (step 8): BLOCKED / no_impact! Only 2 cells changed (bar 20,61 & 20,62: 11->3).
  Block stayed at (30,34). My model predicted (30,29). MISMATCH.
- 2 queued LEFTs skipped due to mismatch.
- Batching LOST (mismatch). Must re-qualify with one matched action.

## WALL DISCOVERED
Rows 30-34, cols 29-33 are background-4 (a wall corridor), NOT floor-3. So the left
approach to the marker is blocked by a wall. The floor at marker rows 30-34 spans
cols 14-28 and 39-53 but NOT 29-33. The marker (cols 20-23) is in the LEFT floor
section; the block (cols 34-38) is in the RIGHT floor section. They are separated by
the wall corridor at cols 29-33.

## Revised objective hypothesis
The 0/1 marker is on the left side of a wall the block cannot cross here. The block
must be moved DOWN around the wall, or the objective is the 9-glyph frames. 
Let me check if there's a path around the wall. Floor rows: 17-49. The wall corridor
cols 29-33 — at what rows does it exist?


# Actor Beliefs - Level 0

## Observations so far
- Grid 64x64, background 9.
- Colors mapping: a=10 vertical barrier cols 30-32 rows 0-62, b=11 right column 63 rows 0-63, bottom row 63 cols 0-62 is 5 (value 5).
- Components:
  - 5-shape at (~15-23,18-26) with interior zeros at (16,19),(16,22),(16,25),(19,25),(22,25). Looks like a 5-border with 5 holes.
  - 4-shape L at (~15-23,36-43) left arm 3 cols wide, top 3 rows full width 8 cols.
  - b-shape (11) L at (~45-53,51-59) similar L shape to 4-shape but on right side.
- Actions available: RESET, UP, DOWN, LEFT, RIGHT, SPACEBAR, CLICK, UNDO.
- No obvious player avatar visible. No HUD text observed.

## Hypotheses
- Game likely about spatial matching / symmetry using symmetry.py utilities.
- Barrier of a (10) separates left area (shapes 5 and 4) from right area (b shape). Possibly need to bring left shapes to right or match them.
- Zeros inside 5 shape may be holes or movable tokens.
- Arrow keys may move a cursor or move a selected piece. CLICK may pick/place.
- SPACEBAR may activate or rotate.
- Goal possibly to make 4 shape congruent to b shape (both L shapes) under some symmetry quotients.
- 5 shape with holes may need to match something else? Could be a piece to move.
- The bottom row of 5 and right column of 11 may be boundaries/walls.

## Tests performed
- Checked grid stats, components, shapes.
- Executed UP action: 4 and 5 shapes moved up 3 rows, zeros moved with them, column 63 top cell changed 11→5.
- Executed DOWN action: shapes moved back down 3 rows to original position, column 63 second cell changed 11→5 (move counter).
- World model builder now reports movable 4/5 shapes move 3 cells per UP/DOWN with whole bounding-box copy, monotonic move counter painted top-down on column 63.
- Executed LEFT action: confirmed horizontal move 3 cells. 5 shape moved left 3 cols, 4 shape (right side) moved right 3 cols due to mirroring across barrier.
- Verified 4 shape is congruent to b shape (11 L) under translation, size 45 both, directly congruent True.
- Executed DOWN action: shapes moved down 3 rows.
- Level 0 completed with LEFT action aligning 4 shape over 11 shape.
- Level 1 start: barrier at cols36-38 with holes every 3 rows, 4 shape top left, 5 shape top right, 11 shape bottom left. DOWN action produced no cell change, suggesting mechanics changed: may require selection via CLICK before moving.

## Open questions
- What moves? Level 1 may require CLICK to select shape before arrow moves.
- Level 1 may involve moving shapes through barrier holes.

## Next steps
- Try CLICK on a shape cell to select it.

[checkpoint: turn ended with no model-authored update]

[checkpoint: turn ended with no model-authored update]

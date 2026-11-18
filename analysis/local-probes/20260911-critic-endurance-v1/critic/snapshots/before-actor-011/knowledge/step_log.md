## Turn 0 — initial inspection
- Level 0, step 0, NOT_FINISHED. Actions: RESET, UP, DOWN, LEFT, RIGHT.
- Grid layout (color = hex digit):
  - 4 = dark blue background, fills most of the right side.
  - 5 = purple, left strip (cols 0-3) + bottom area + bordering frames.
  - 3 = orange, large maze-like structure rows 8-49, cols 14-53.
  - 9 = teal, scattered: inside an 8x8 room (rows 9-16, cols 14-21), a vertical column, and inside a bottom-left box (rows 55-62, cols 1-10).
  - 0/1 = black/dark-red, tiny 2x2 cluster at rows 31-33, cols 15-22 (a player? cursor?).
  - 11/8 = pink/magenta, bottom-right (rows 61-62, cols 13-54 pink; 56-63 mag).
  - 12 = yellow, rows 45-46 cols 34-38 (5 cells) — goal/target marker?
- Hypothesis A: 0/1 cluster is the player; directional keys move it.
- Hypothesis B: 9 (teal) is movable.
- Will probe with DOWN (single action) to see which entity moves.


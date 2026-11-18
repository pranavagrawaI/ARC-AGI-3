# Game notes

## Established mechanics (level 0 — SOLVED in 25 actions)
- Board 64x64, bg=5. Objects are "crosses" made of a long arm pair; the SELECTED object has a WHITE (0) cell at its centre.
- Arrow key moves selected object exactly **3 cells** per action.
- SPACEBAR toggles/selects the other object (white marker jumps).
- 3x3 dark(4) rings with a coloured dot centre = targets. **Win = every dot of colour C is covered by the arms of the object of colour C** (all objects aligned simultaneously). Level 0 won when yellow cross centred (9,15) covering its 4 yellow dots and blue cross centred (24,48) covering its 4 blue dots.
- Bottom row 63 (colour 15 → 1 from right end) = step/time budget: consumed 15 cells after 24 actions (probabilistic-ish, ~0.6/action). 64 cells total. Not yet seen as a loss cause.
- Cross arms pass through rings harmlessly; no death observed.

## Level 1 (start)
Objects (3):
- ORANGE X, colour 12, DIAGONAL cross (arms along the 4 diagonals, length 11 each), centre (18,27) — SELECTED (white).
- MAGENTA, colour 13, diamond outline |r-30|+|c-39| = 9 (one cell (30,48) appears missing), centre (30,39).
- BLUE, colour 9, axis cross arms ±13, centre (42,48).
Dots:
- 12: (39,9),(42,24),(57,9)
- 13: (3,21),(9,27),(12,12)
- 9: (36,27),(48,15),(48,33),(60,27)

### Derived targets (same-cover hypothesis)
- Orange X centre (48,18): r+c=66 arm covers (42,24)&(57,9); r-c=30 arm covers (39,9). All dist 9 ≤ 11. Δ from (18,27) = +30 rows, -9 cols → DOWN x10, LEFT x3 (13 actions).
- Blue cross centre (48,27): row 48 covers (48,15),(48,33); col 27 covers (36,27),(60,27). Δ = +6, -21 → DOWN x2, LEFT x7 (9 actions).
- Magenta diamond centre (12,21): |r-12|+|c-21|=9 covers (3,21) 9, (9,27) 3+6=9, (12,12) 9. Δ = -18, -18 → UP x6, LEFT x6 (12 actions).
Total ≈ 34 moves + 2 selections. Budget 64 bar cells, ~0.6/action → OK.

## TODO / unknowns
- Does SPACEBAR cycle through 3 objects? order?
- Does the magenta diamond behave as a movable object (and is its "centre" (30,39))?
- Bar semantics / loss condition.

[checkpoint: model committed DOWN with no model-authored beliefs update]

[checkpoint: model committed DOWN with no model-authored beliefs update]

[checkpoint: model committed LEFT with no model-authored beliefs update]

[checkpoint: model committed UP with no model-authored beliefs update]

[checkpoint: model committed UP with no model-authored beliefs update]

[checkpoint: model committed LEFT with no model-authored beliefs update]

[checkpoint: model committed SPACEBAR with no model-authored beliefs update]

[checkpoint: model committed LEFT with no model-authored beliefs update]

[checkpoint: model committed LEFT with no model-authored beliefs update]

[checkpoint: model committed SPACEBAR with no model-authored beliefs update]

[checkpoint: model committed UP with no model-authored beliefs update]

[checkpoint: model committed UP with no model-authored beliefs update]

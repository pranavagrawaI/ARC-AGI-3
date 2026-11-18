### turn 0 (step 0)
Observed initial grid. Playfield: color-3 floor with color-5 walls, a plus-shaped
2-color object at (31-32, 21-22), arrow patterns (9 up/right, 12 left, c down) in a
5x5 icon at rows 8-16 cols 31-37, a 3x3 block of 9s at (47-49, 34-38), and a bottom
strip rows 61-62 cols 13-54 of 11s. Side icons at (53-60, 1-9) and a door-like 5x5
at (8-16, 31-37).
Next: probe RIGHT to learn which entity moves and its dynamics.

### turn 1 (step 1)
Executed RIGHT (probe). Measured: the 5x5 arrow pattern at (45-49,34-38) shifted +5 cols to (45-49,39-43).
The 2-color blob at (31-33,20-22) did NOT move. (61,13),(62,13) changed 11->3 — minor HUD strip change,
unexplained (possibly cosmetic).
=> The movable entity responds to RIGHT is the arrow-pattern object, not the blob.
Carry-forward: maze is color-5 walls / color-3 floor. Blob plus-shape at (31-33,20-22).
Question now: does the arrow object move one step per action or 5? Probe with one LEFT and measure.


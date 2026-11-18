## Turn 0 (step 0) — initial layout
Grid 64x64, bg=5. Structural elements (all measured, no object labels yet):
- Rows 0-8, cols 16-63: a wall/band region of 3s with a 4-edge at cols 16-17.
- Box A: rows 3-12, cols 3-12 (10x10). Top half rows 3-7 = 0s; bottom half rows 8-12 = 15s (full, 50 cells).
- Box B: rows 24-32, cols 25-38 (9x14), border 2. Inner 7x14 mostly 15s but 6 inner cells (rows 28-30, cols 28-30) are 0. 15-count 84/98.
- Box C: rows 2-6, cols 34-39, border 4. Inner 3x3 mostly 15s but 3 cells (rows 3-5, col 35) = 0.
- Row 63 all 4s; also a 4-line at rows 16-17 (cols 0-15).
Open questions: is there a player entity? Do arrow keys move something? Does SPACEBAR/CLICK fill/erase 0s or 15s?
- H1: arrows move a player token | untested | test: one RIGHT
- H2: SPACEBAR/CLICK toggles paint (fill 0 holes) | untested


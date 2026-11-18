# Hypotheses

| id | hypothesis | status | evidence |
|----|-----------|--------|----------|
| H1 | 4x4 grid-aligned movement: UP/DOWN/LEFT/RIGHT move the player token (4x4, color 14) by 4 cells (one square), if the path strip is free of wall blocks (4x4, color 4) | untested | layout is 4-aligned: player (48-51, 32-35), target rows 28-31 (28-31? no, 28-31 is wall) |
| H2 | walls are 4x4 blocks of color 4; collision blocks movement into/through them | untested | wall blocks at (28-31,16-19), (24-27,32-35), (36-39,32-35) |
| H3 | level completes when the player's 4x4 body lands on the 10x10 target zone (color 2, rows 29-30? actually 28-31) centered (29,33) | untested | target blob rows 29-30? check: color 2 at rows 29,30 cols 31-40 (from render: row29 '922222222229' cols 28-37, row 30 same) |
| H4 | SPACEBAR does nothing / interacts | untested | |
| H5 | step = 4 cells (one square); can the player move freely inside the 10x10 target (2 steps of 4)? | untested | |

Notes on exact geometry (level 0):
- player: rows 48-51, cols 32-35 (center (50,33))
- target: 10x10 rows 29-38? no. color2 rows: 29,30 only? From render: row 29: cols 28..37 pattern '9 22222222 229'... need exact.


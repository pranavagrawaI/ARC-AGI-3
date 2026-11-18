# Hypotheses
Level 0 (game start). Grid facts (verified in code, turn 0):
- Colors: 0=unused, 4=blue, 5=orange, 9=dark bg, 10=white, 11=red.
- Static walls: full white vertical bar cols 30-32 (rows 0-62); red right border col 63;
  orange bottom row 63 (cols 0-62); corner cell (63,63)=11.
- Orange vertical bar col 24, rows 15-23.
- Orange shape (rows 45-53, cols 51-62): rows 45-47 filled cols 51-59; rows 48-53 only cols 51-53 (an L / 7 shape).
- Blue shape (rows 15-23, cols 36-44): rows 15-17 filled cols 36-44; rows 18-23 only cols 36-38 (a mirrored L / 7 shape).

H1 | untested | Player entity exists and moves with arrows; moving into 9 is a wall (no_impact).
    Test: submit UP, check diff of non-wall cells (a player would be a non-9 region that shifts).
H2 | untested | Win = collect 4 (blue) pieces: blue L is 45 cells = 9x5 pattern; orange has 49 cells
    (49 = 7x7?). Orange count is 103 total (includes row 63 = 63 cells + col24 9 cells + shape).
    Compute: 63 + 9 + 31 = 103; shape cells = 103-63-9 = 31. Blue = 45. 10 bar = 189. 11 = 109.
H3 | untested | Hitting the orange "door" shapes (col 24 vertical bar / L-shapes) opens passage through wall col 30-32.
    Test: RIGHT x6 from some row hits col 24; RIGHT x7 hits col 30 (white wall).

## Evidence log
turn 0: grid parsed, no actions yet.

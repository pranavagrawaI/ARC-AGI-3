# Hypotheses (level 0)

| # | Hypothesis | Status | Evidence / deciding test |
|---|------------|--------|--------------------------|
| H1 | Movement is 1 cell per action, blocked by non-3 cells | untested | Submit LEFT: center 21->20 means 1-step |
| H2 | Movement is SLIDE: sprite travels until blocked by wall | untested | Submit LEFT: center jumps to col 14/15 (torso left edge) |
| H3 | Level goal: touch the blue (9) patch at rows 47-49, cols 34-38 (just below orange 12 strip rows 45-46) | untested | reach patch and watch for level_completed |
| H4 | 3=walkable floor, 4=wall, 5=wall/border, 9/12=walkable (goal items) | untested | try crossing 12/9 if in path |
| H5 | Sprite = plus shape: center (0) + arms (1: down, left; 0: up, right), center at bbox center | confirmed | grid turn 0: cells (31,21),(32,20),(32,21),(32,22),(33,21) |
| H6 | HUD (static): yellow bar rows 61-62 c13-54 = timer/progress; red 8-dots rows 61-62 c56-63 = lives(3x2 blocks); head box rows 8-16 c32-40 = icon; bottom-left 5-icon = decoration | untested | watch HUD change over time |
| H7 | Torso has internal wall strips: 4 at rows30-39 c29-33, rows40-44 c24-33 and c14-18 | confirmed (turn 0 grid) | run-length analysis |

## Grid anatomy (turn 0)
- Body (3) bbox rows 8-49, cols 14-53. Head box rows 8-16 (3-border, 5 fill, blue icon 9).
- Neck rows 17-24 cols 34-38. Torso rows 25-49 (see H7 for holes).
- Sprite (player) center (32,21) inside torso.
- Goal candidates: blue 9 patch rows 47-49 c34-38 + orange 12 strip rows 45-46 c34-38 (together a 2-row orange / 3-row blue rectangle at bottom-right of torso).


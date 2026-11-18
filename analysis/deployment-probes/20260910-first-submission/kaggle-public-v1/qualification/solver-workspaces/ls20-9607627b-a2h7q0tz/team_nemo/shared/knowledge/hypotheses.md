
# Hypotheses

| Hypothesis | Status | Evidence |
|---|---|---|
| Player is the 0/1 glyph at (31-33, 20-22) center (32,21) | untested | only small distinct cluster in main region |
| Main region (rows 8-49, cols 14-53, color 3) is walkable space | untested | player inside it, walls are 4 |
| 4 (blue) is walls | untested | frames the whole region, surrounds everything |
| 3 (red) is floor | untested | fills the walkable region |
| 5 (green) is walls/obstacles | untested | appears in door sub-structure rows 9-15 |
| 9 (orange) = decorative/obstacle dots | untested | scattered small clusters |
| 11 (bottom strip rows 61-62 cols 13-54) = progress bar/HUD | untested | long horizontal strip at bottom |
| 12 (cyan, 5x2 at rows 45-46 cols 34-38) = goal marker (upper half) | untested | in a sub-room at bottom of main region |
| 9 at rows 47-49 cols 34-38 = goal marker (lower half) | untested | directly below cyan, same 5-wide |
| Top-left sub-structure (rows 8-15, cols 34-40, with 5 and 9) = a door/portal | untested | unusual pattern within the 3-region |
| Bottom-left room (rows 53-60, cols 1-11) separate region with 9 clusters | untested | enclosed in 5, not connected to main region? |
| Win condition: reach/step on goal marker at (45-49, 34-38) | untested | goal-like position |

## Map layout (row:col bounds)
- Main walkable region: rows 8-49, cols 14-53, floor=3, walls=4 (outer border)
  - Sub-rooms separated by 3-walls:
    - Room A: rows 8-15, cols 34-40 (door structure with 5/9)
    - Room B: rows 17-24, cols 29-53 (top)
    - Room C: rows 25-29, cols 14-53 (middle band, connected to B via col 38-39 rows 17-24? need to check)
    - Room D: rows 30-39, cols 14-18 (small left room, connected via rows 30-39 gap? NO - check col 19 at rows 30-39)
    - Room E: rows 30-39, cols 19-38 (main room, player here)
    - Room F: rows 30-39, cols 39-53 (right room, connected via col 25-26 rows 40-44?)
    - Room G: rows 40-44, cols 19-38 (below room E)
    - Room H: rows 45-49, cols 14-53 (bottom band, contains goal marker)
  - Wait, let me re-examine the connectivity more carefully.
- Bottom-left room: rows 53-60, cols 1-11, floor=5, walls=4, has 9 clusters
- Bottom HUD: rows 60-62, cols 52-63 (5, b, 8 pattern) - progress indicator?
- 11 strip: rows 61-62, cols 13-54 - lives? progress?


| Hypothesis | Status | Evidence |
|---|---|---|
| The 12/9 5x5 cube IS the player (moves on direction keys) | confirmed (step 1) | RIGHT moved cube right 5, player glyph static |
| 0/1 diamond at (32,21) is a static marker/gem | untested | didn't move on RIGHT |
| Cube moves 5 cells per key press | untested (need second test) | observed 5-cell shift on first RIGHT |
| 11-strip (bottom) is a lives/moves counter | untested | lost 2 cells at left end on RIGHT |
| 4 (blue) = walls | untested | |
| 3 (red) = floor | confirmed-ish | cube moved through 3-floor |
| Goal: reach the 0/1 diamond? reach the door (5/9 structure)? | untested | |


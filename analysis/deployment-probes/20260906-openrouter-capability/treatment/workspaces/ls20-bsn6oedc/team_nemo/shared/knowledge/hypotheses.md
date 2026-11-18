# Hypotheses

| # | Hypothesis | Status | Evidence |
|---|-----------|--------|----------|
| H1 | Player is the 0/1 blob at (31-33, 20-22); movement = move in that color | untested | rare 5-cell cluster, interior |
| H2 | 3 = ground/wall (892 cells, big region rows 8-49 cols 14-53) | untested | surrounds player |
| H3 | 4/5 = background/water (fills full 64x64 grid) | untested | 2609+439 cells |
| H4 | 9 = collectible/target items (3 clusters: (11,35-37), (47-49,34-38), (55-60,3-8)) | untested | sparse clusters |
| H5 | 12 = special item (2x5 block at (45-46, 34-38)) | untested | right above cluster 2 of 9s |
| H6 | 11 = status bar / floor (bottom rows 61-62) | untested | horizontal band |
| H7 | 8 = status indicator (bottom-right rows 61-62 cols 56-63) | untested | 12 cells in .88.88.88 pattern |
| H8 | Level 0 goal: walk to collect 9-items or reach a specific location | untested | TBD |

## Deciding tests
- Submit one move (e.g., RIGHT) and see which cells change. If the 0/1 blob moves by one in that direction, H1 confirmed.
- If the blob can't cross color 3 (or 4/5), those are walls.

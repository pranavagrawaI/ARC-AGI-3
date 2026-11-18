# Step log

## Turn 0 (step 0, level 0)
Initial observation analyzed in code:
- 64x64 grid. Colors present: 4 (bg, 2609), 5 (wall, 439), 3 (floor, 892, one big
  connected component rows 8-49), 9 (45 cells: 3 clusters — (11-13,35-37) ring,
  (47-49,34-38) solid 3x5, (55-60,3-8) 6x6 ring), 12 (2x5 block at (45-46,34-38)
  directly above the (47-49,34-38) 9-block), 8 (12 cells, bottom rows 61-62
  cols 56-63 — HUD digits?), 11 (84 cells, bottom rows 61-62 — HUD background?),
  0/1 (5-cell sprite at rows 31-33, cols 20-22 — presumed PLAYER).
- Sprite 0/1 cells: (31,21) 0; (32,20) 1, (32,21) 0, (32,22) 0; (33,21) 1.
  All surrounded by floor 3 in all 4 directions.
- Player reference defined as topmost 0 = (31,21).
- World model v2 written: encode extracts player/sprite/walk(0,1,3,9,12)/items9/
  blocks12/hud; predict = 1-cell translate of sprite if target walkable, else no-op.

Experiment: UP from (31,21) -> expect sprite at (30,21), grid changes only by
sprite cells (old 0/1 cells -> floor 3), nothing else moves.

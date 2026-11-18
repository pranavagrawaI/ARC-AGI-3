# Hypotheses

| # | hypothesis | status | evidence / deciding test |
|---|-----------|--------|--------------------------|
| H1 | 0/1 5-cell cluster at (31-33,20-22) is the player sprite, moves 1 cell per dir action | untested | test: UP -> sprite at (30,21) |
| H2 | 9 = collectible items; touching them removes them (maybe counted in HUD) | untested | test: walk onto a 9 |
| H3 | 12 block at (45-46,34-38) = goal/lift over the 9-block at (47-49,34-38) | untested | test: approach/walk into it |
| H4 | colors 8/11 rows 61-62 = HUD display (static, no effect) | untested | monitor after moves/pickups |
| H5 | walkable = {3,9,12} (+sprite cells); 4/5 block movement | untested | test: bump into wall, step onto 9/12 |
| H6 | goal = collect all 9? or reach 12? | untested | discover via level_up event |

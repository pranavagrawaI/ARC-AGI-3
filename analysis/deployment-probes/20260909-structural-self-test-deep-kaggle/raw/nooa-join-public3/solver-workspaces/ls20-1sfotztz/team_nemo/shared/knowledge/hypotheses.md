# Hypotheses
| # | hypothesis | status | deciding test |
|---|---|---|---|
| H1 | cross (0/1) @center(32,21) is the player | untested | RIGHT probe: cross shifts right |
| H2 | 3 (olive) is a breakable wall; entering cell becomes 4 | untested | RIGHT probe: (31,22),(32,23),(33,22) 3->4 and player advances |
| H3 | 9 (maroon) = collectable item; goal = collect all 9s (45) | untested | step onto 9 later; watch disappearance + level progress |
| H4 | 5 (purple) = walkable floor (rooms) and outer border | untested | step onto 5 later |
| H5 | old glyph cells revert to '4' when player leaves | untested | RIGHT probe: (31,21) etc. become 4 (or 3?) |
| H6 | 8/b (silver/azure) bottom-right zone = exit/goal | untested | reach it |
| H7 | 5x5 block (12+9) is the moving object; jumps 5 (own width) in action direction, overwriting 3, leaving 3 behind | untested | LEFT probe: returns to cols 34-38 |
| H8 | b strip (rows 61-62) is a timer/health: erodes 2 per action from left end | untested | LEFT probe: erodes left end again (cols 15-16) |
| H9 | 3 (olive) is traversable floor/wall to movers (block crossed it) | supported | RIGHT probe: block crossed 3s |
| H10 | cross (0/1) is static (decoration/spawn), not the player | supported | RIGHT probe: unchanged while block moved |


| block = player, jumps exactly 5 cells per dir action on '3' arena, leaves '3' trail | CONFIRMED 2/2 exact (model v1) |
| b-strip (rows61-62) erodes 1 col from left per action (timer/fuel, ~40 left) | CONFIRMED 2/2; direction-independent |
| 0/1 glyph (31-33,20-22) is static | CONFIRMED (untouched in 2 transitions) |
| GOAL: block must reach glyph area (then what? touch/cover/trigger) | UNTESTED - approaching via (30,34)->(30,24) |
| top 9-cluster, bottom-left 9-cluster, 8-pattern, 4/5 borders = scenery | UNTESTED (no changes so far) |
| 5x5 block may pass THROUGH glyph cells (collision rule exception at goal?) | UNTESTED |


| GOAL = 5x5 block must land exactly on glyph tile (30,19), overwriting 0/1 glyph (then level_up) | TESTING NOW (final action of 7-batch) |
| movement blocked whenever any dest cell != 3 (strict) | implied by model v1; glyph tile is the first test of an exception |
| 9-clusters / top box / bottom-left cluster = static scenery outside arena | SUPPORTED (position relative to reachable tiles; untested by contact) |
| b-strip = per-action timer/fuel, ~32 cols left after batch | CONFIRMED mechanic (3/3); purpose (loss condition?) untested |


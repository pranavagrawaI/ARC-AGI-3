
## Turn 0 — initial (all provisional, no moves taken yet)
| claim | evidence | tested scope | open conditions |
|---|---|---|---|
| 9-A (rows 11-13, cols 35-37) inside 3-box = start marker | static grid only | not moved | could be player |
| 9-B (rows 56-62, cols 1-9) = decoration/second player | static grid only | not moved | could be player |
| 11-bar = timer | static grid only | not moved | could be wall |
| 12-trail rows 35-36 cols 39-43 = goal spot / spawn | static grid only | not moved | could be trail |
| 0/1 cross = target | static grid only | not moved | could be decoration |
| 8-block = teleporter | static grid only | not moved | unknown |


## Turn 1 updates
| claim | evidence | tested scope | open conditions |
|---|---|---|---|
| Player = 5x3 solid 9-block, moves 5 rows per UP | step 1 | UP only | LEFT/RIGHT/DOWN untested |
| 12 = 2x5 "shadow" above player, moves with player | step 1 | UP only | persists? disappears at boundary? |
| 11 = timer bar, loses 2 cells per move from left | step 1 (70->68) | one move | what happens at zero? |
| 9 at rows 11-13, cols 35-37 = static decoration in 3-box | step 1 (unchanged) | not moved | could be goal |
| 9 at rows 55-60, cols 3-8 = static decoration | step 1 (unchanged) | not moved | could be goal |

## Objective (still unresolved)
- H1: reduce 11 bar to zero (timer = steps remaining?)
- H2: reach the 0/1 cross (rows 31-33, cols 20-22)
- H3: reach the 8-block (rows 61-62, cols 56-63)
- H4: collect or interact with the 3-box (rows 8-16, cols 35-43)
- H5: reach the lower-left 9-cluster


# Hypotheses

| id | hypothesis | status | evidence / deciding test |
|----|------------|--------|--------------------------|
| H1 | The only usable action is `CLICK x y` (column x, row y, 0-indexed); there is no free player. | untested | only CLICK/RESET in available_actions |
| H2 | `bb` (color 11) is the player token, moves to / toward the clicked cell. | untested | it's the only compact block in an open field |
| H3 | `9` (12px block at cols 60-63, rows 24-35, split by the `5` band rows 28-31) is the GOAL to reach. | untested | positioned as an endpoint |
| H4 | `5` band (rows 28-31, cols 20-63) is a wall/border crossing the board; `4` may be a wall too. | untested | `4` hugs the `bb` player's left side |
| H5 | Clicking a wall cell / outside playfield is inert or rejected. | untested | click a `3` cell |

## Layout (level 0, initial)
- 7 = status bar row 0 (HUD).
- 3 = open field, left region cols 0-31 (rows 1-63) and right region below row 35.
- 0 = open field, right region cols 32-63 (rows 1-63).  Two open colors!
- 5 = horizontal band rows 28-31, cols 20-63.
- 9 = block rows 24-35, cols 60-63 (but rows 28-31 there are `5`).
- 4 = 4-px blob at (46-49, 46-49).
- 11 (bb) = 2-wide block, rows 28-31 cols 38-39 and rows 44-49 cols 50-51, plus rows 46-47 cols 48-49.

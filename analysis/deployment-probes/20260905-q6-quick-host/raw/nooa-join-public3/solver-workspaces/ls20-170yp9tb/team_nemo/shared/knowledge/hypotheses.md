# Hypotheses

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|----------|
| H1 | Agent moves 5 cells per action in the pressed direction | confirmed | UP: rows 45->40, RIGHT: cols 34->39 |
| H2 | 11-band (rows 61-62, cols 13-54) is a timer/lives counter, shrinks 2 cells/step | untested | lost 2 cells on UP and RIGHT |
| H3 | 5-color = wall (impassable) | untested | agent has not yet tried to enter 5 |
| H4 | 4-color = wall or boundary | untested | seen at edges and in the central column gap |
| H5 | 9-pixels inside top 3/5 block (rows 11-13) are a goal/target | untested | static pattern, never moves |
| H6 | 0,1 pixels at (31-33, 20-22) are another entity or goal | untested | static |
| H7 | 8-block bottom-right (rows 61-62, cols 56-63) is exit/portal | untested | static |
| H8 | 3-color = floor (passable) | confirmed | agent moves through 3 freely |
| H9 | Game over when 11-band reaches 0 | untested | band is shrinking |
| H10 | Goal is to reach a specific position or collect items | untested | unknown |

## Carry-Forward
- Agent: 5x5 sprite (12 top 2 rows + 9 bottom 3 rows), moves 5 cells/step
- 11-band: 2 rows tall, shrinks 2 cols/step from left (col 13 first)
- Static scenery: 5-walls, 4-walls, 9-pattern top, 0/1 cluster, 8-block
- Unknown: what wins the level, what 0/1/8/9-patterns do


# Hypotheses

| id | hypothesis | status | evidence / deciding test |
|----|-----------|--------|--------------------------|
| H1 | red(3) is walkable ground; player starts on it | untested | LEFT probe: sprite should shift left on red |
| H2 | 0/1 sprite = the player; arrows move it | untested | LEFT probe: only the sprite region changes |
| H3 | dark(4) corridors are holes/obstacles (or walkable) | untested | move toward a 4 corridor, see if player enters or is blocked |
| H4 | goal = reach/collect the 12+9 object (rows45-49, cols34-38) | untested | navigate player to it; watch for level_up |
| H5 | 9-clusters in closed brown boxes are static decoration / other items | untested | do not move; ignore for now |

## Movement sub-hypotheses (to discriminate after first probe)
- M1: 1 cell per action (maze).
- M2: slide until a wall (ice).
- M3: player blocked by red => red is wall, dark is floor (contradicts H1).

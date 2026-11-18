# Hypotheses

| id | hypothesis | status | evidence |
|----|-----------|--------|----------|
| H1 | Player = rigid 5-cell plus sprite (colors 0 hub + 0/1 arms); one direction action moves it exactly 1 cell; vacated cells revert to floor 3 | untested | sprite seen at rows 31-33/cols 20-22, hub (32,21); no motion observed yet |
| H2 | Color 3 = walkable floor; color 4 = wall (sprite blocked); color 12 pad = win on contact | untested | 3 fills the big room; 4 encloses rooms; 12 is a 5x2 pad in the room's SE corner |
| H3 | Color 9 cells = items (collectible, non-fatal); 5/8/11 are HUD chrome, not playfield | untested | 9 appears in 3 clusters (top room, big room above goal, bottom-left box inside a 5-walled box); chrome patterns on rows 53-63 |
| H4 | Level 0 goal = touch the 12-pad | untested | pad is the only goal-like marker on the playfield |

## Deciding tests
- H1: submit DOWN; expect sprite at rows 32-34 / cols 20-22, 7 changed cells (2 vacated, 2 entered, 3 recolor), reward > 0, no level_up.
- H2: first step is into floor 3 (already implied by H1 test); wall test later at a known 4 boundary.
- H4: 26-step BFS plan to pad stored in self.v.plan (DOWN x14 then RIGHT x12); last step predicted level_up.

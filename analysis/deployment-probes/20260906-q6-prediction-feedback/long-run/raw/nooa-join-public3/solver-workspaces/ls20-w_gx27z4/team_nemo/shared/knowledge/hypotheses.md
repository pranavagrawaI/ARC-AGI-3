# Hypotheses (level 0)

H1 | status: untested | "UP/DOWN/LEFT/RIGHT move a controllable player by one cell, no sliding."
- Player candidates: color 0 (white, rows 31-33), or color 1 (red, rows 32-33).
- Deciding test: press RIGHT; observe which cell changes (diff cells + new position).

H2 | status: untested | "The goal is to reach the color-9 (teal) pad at rows 55-56, cols 3-6."
- Evidence: pads also appear as "eyes" (9s inside 5-boxes) at rows 11-13 — maybe collectibles or just decor.
- Test: after reaching the pad, watch for level_completed.

H3 | status: untested | "Color 3 (orange) is the maze wall; the white 'path' (color 4) is walkable floor."
- Corridors visible: vertical col ~20 (rows 17-44), then row ~30 corridor cols 20-53, then col ~34, then col ~53.
- Walls (5=yellow?) frame the top-left region; colors 8/11/12 at rows 61-62, cols 56-63 look like a score/HUD band.

## Open questions
- What does color 5 (yellow) vs 4 (white) vs 3 (orange) mean?
- Are 0 and 1 both part of the player (a 2x2 sprite: 0s at (31,21)(32,20), 1s at (32,21)(33,20))?


## Turn 4-5 results
- LEFT: assembly 39->34 (confirmed). Step counter off by 1 (removed from model).
- UP: assembly 40->35 (confirmed). Model MATCHED. Batch allowed.

### Updated model (v6)
- Assembly (12-cyan 2x5 + 9-teal 3x5 = 5x5 block) slides 5 cells per action in pressed direction.
- Moves through walls. Clamped at boundaries (floor at row 50+, grid edges).
- 0/1 cross at (31,20) is STATIC decoration.
- Step counter (color 11 in HUD) decrements each action (rate unclear).
- Assembly reachable positions: row in {0,5,10,...,45}, col in {0,5,10,...,59} (clamped).

### H4 | status: untested | "Goal is to reach a specific position"
- Evidence: no level_completed yet after 5 moves. No obvious trigger.
- Test: explore positions and watch for level_completed.

### H5 | status: untested | "Goal involves the 9-teal 'eyes' at rows 11-13, cols 35-37"
- Assembly at (10, 35) would cover rows 10-14, cols 35-39 (eyes inside).
- Test: move assembly to (10, 35) or nearby.

### H6 | status: untested | "Goal involves the 0/1 cross at (31,20)"
- Assembly can't align with cross (not on 5-grid). Cross is at col 20, assembly cols are 0,5,10,...
- Closest: assembly at (30, 20) covers rows 30-34, cols 20-24. Cross is at rows 31-33, cols 20-22.
- Test: move assembly to (30, 20).

### H7 | status: untested | "Goal is the 9-teal pad at bottom-left (rows 55-60, cols 3-8)"
- Assembly can't reach there (row 55 not on 5-grid from 45; col 3-8 not on 5-grid from 34).
- Probably not the direct goal.


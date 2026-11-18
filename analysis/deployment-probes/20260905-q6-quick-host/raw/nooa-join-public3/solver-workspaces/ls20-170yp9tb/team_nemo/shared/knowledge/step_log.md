# Step log

## Turn 0 (initial observation)
State: level 0, step 0, NOT_FINISHED. Available: RESET UP DOWN LEFT RIGHT (no USE, no CLICK).
Colors seen: 3 (892 px, rows 8-49 cols 14-53 — big central mass), 4 (2609 px background), 5 (439 px scattered), 9 (45 px, rows 11-60 cols 3-38), 11 (84 px, rows 61-62 cols 13-54 — bottom band), 8 (12 px, rows 61-62 cols 56-63), 12 (10 px, rows 45-46 cols 34-38), 0 (3 px at (31,21),(31,22),(32,21)), 1 (2 px at (32,20),(33,21)).
Plan: probe mechanics with single actions before batching. First probe: UP.


## Turn 1 (after UP probe)
Result: UP moved the agent (12-top + 9-body, 5x5 at rows 40-44, cols 34-38) from rows 45-49 — i.e. UP = -5 rows. 11-band (bottom, rows 61-62, cols 13-54) lost 2 cells ((61,13),(62,13) 11->3). Retrodiction correct (move).
Hypotheses:
- H1: agent moves 5 rows/cols per action (confirmed for UP).
- H2: 11-band = lives/timer, shrinks 2 cells per step (untested count).
- H3: goal unknown. Static 9-pixels inside 3/5 block (rows 11-13, cols 35-37) may be a target or decoration.
Known static scenery: 5-block top-left (rows 0-7 cols 0-3), 5-block left (rows 53-62 cols 1-11 with 9s inside rows 55-60), 3/5 structure top-center (rows 8-16, cols 34-40), 8-block bottom-right (rows 61-62, cols 56-63), 0/1 pixels (rows 31-33, cols 20-22), bottom 11/3 band (rows 61-62), 'b' rows? no — color b (11) band.
Plan: probe RIGHT to confirm horizontal step=5 and watch the 11-band.


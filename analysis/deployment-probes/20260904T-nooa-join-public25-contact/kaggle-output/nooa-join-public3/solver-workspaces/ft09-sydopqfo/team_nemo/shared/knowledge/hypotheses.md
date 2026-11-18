# Hypotheses (level 0, 'the game')

| id | hypothesis | status | evidence / deciding test |
|----|-----------|--------|--------------------------|
| H1 | 64x64 grid = 32x32 2x2 tiles; board 0..30, tile-row 31 = status bar (12) | confirmed | all tiles uniform except 32 status-bar tiles; 6 color kinds |
| H2 | CLICK x y selects the tile at (y//2, x//2); flood-fills its 4-connected same-color run | untested | first experiment this turn |
| H3 | effect: run flips 0<->2, 8<->9; 4-runs inert; 5 = wall | untested | watch which colors change |
| H4 | gravity: flipped tiles fall/fill within the column or run region | untested | pick a 9-tile with 5 below (3,2); expect fall direction signal |
| H5 | status bar (12) shows a score/timer, changes on clicks | untested | diff_summary should show status band rows 62/63 |
| H6 | goal: clear all non-5 board tiles (or make a color extinct) | untested | watch level completion |

## Board facts (turn 0, step 0)
- 14 uniform 3x3 blocks of 8 or 9 at tiles rows {1-3,5-7,9-11,18-20,22-24,26-28}, cols {2-4,6-8,10-12,19-21,23-25,27-29} (some missing in the band rows 5-7).
- 4x4 blue (4) block at tiles rows 16-29 x cols 17-29 (124 tiles) with 2x2 9-blocks at (18-20,18-20),(22-24,22-24),(18-20,22-24),(26-28,22-24)... and 3x3 green (2) corner blocks at (16-18,16-18),(28-30,16-18),(16-18,28-30),(28-30,28-30).
- 2x2 black/gray (0/8) checker clusters inside several 3x3 blocks: at (5-7,6-8) and (5-7,23-25) pattern:
    8 2 0 / 2 0 8 / 0 2 8  (left one) — 16 black, 16 gray total on board.
- color census: 0:16, 2:36, 4:124, 5:556, 8:112, 9:180 (out of 1024 tiles).
- CLICK coords: "CLICK x y" = col x, row y in GRID pixels (0..63); tile = (y//2, x//2).
- Test click: CLICK 4 6 -> tile (3,2), the 9 at the bottom-left of the top 3x3 block at rows 1-3 cols 2-4, with 5 directly below (boundary tile).


## Turn 1 retrodiction — CLICK 4 6 was no_impact
- Predicted `move` (9-run at tiles rows1-3 cols2-4 flips 9->8). **Wrong: outcome=no_impact, grid identical.**
- CLICK 4 6 = grid col 4, row 6 = tile (r=3, c=2) = a 9 on the bottom-left of that 3x3 9-block, with a 5 directly below (boundary 9).
- Refutation targets: "click center works", "9-run flips 9->8", "run includes this tile".
- NEW hypotheses:
  - H7: only the CENTER tile of a 3x3 block is clickable (edge tiles inert) — test: CLICK interior 9 (tile (2,3) = CLICK 6 4).
  - H8: run is 8-connected (diagonal included) — would change run sizes.
  - H9: a click only affects the run if it is exactly size N (magic number). 3x3 runs n=9; 4-block n=124.
  - H10: clicks toggle a per-tile "selected" state that shows only in animation/event frames; settled grid unchanged until some threshold. (no event_summary this turn though.)
- Next experiment (this turn): CLICK 6 4 (interior 9, 3x3 of 9s around it). If still no_impact → click must target a different tile class (e.g. the 0/2/4 clusters, or status-bar-adjacent). Then try CLICK inside a 0/2/8 cluster and inside the big 4-block.


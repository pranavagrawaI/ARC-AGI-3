# Hypotheses
| H | status | evidence / note |
|---|--------|------|
| H1: player dot (0/1) at (31,21) is the moving avatar | contradicted | After UP (step1), p stayed at (31,21) — dot did NOT move. It may be a static marker or the dot moved and my argmin caught a leftover 1-cell. |
| H2: UP = player steps up 1, blocked by wall 3 | contradicted | UP produced 52 changed cells: c-block 2x5 shifted UP 5 (rows 45-46 -> 40-41), 9-block 3x5 shifted UP 5 (47-49 -> 42-44), b-bar lost 2 cells at (61-62,13). Player dot unchanged. |
| H5: UP shifts dynamic objects (c, 9, and part of b-bar) up by 5 rows; vanishing/clip at edges | confirmed (partial) | Measured exactly: 3->12 at rows 40-41, 3->9 at 42-44, 12->3 at 45-46, 9->3 at 47-49, 11->3 at (61,13),(62,13). |
| H6: goal = bring c-block to some target (maybe bottom-right 8-8-8 area, or a dock) | untested | Need to find what "winning" looks like; the 8-8-8 pattern at bottom-right (rows 61-62, cols 56-63) could be a dock slot. |

## turn 2 (step 2) — after DOWN
- Confirmed: DOWN reversed the UP shift exactly (c back to rows 45-46). Reversible.
- MEASURED: 'b' bar (rows 61-62) depletes from LEFT edge by 1 col per action. 84→82→80 cells. Likely an action-cost meter.
- Rewrote wm.py v3 (encode c + n9[stack cols only], predict UP/DOWN shift ±5). Preflight ready.
- NEXT: probe LEFT (expect: b-bar loses 1 more col; c-stack shifts left by 5 if directional shift applies horizontally).


## turn 3 (step 3) — LEFT result
- MEASURED: LEFT shifts the whole vertical stack (c-block 2x5 + 9-block 3x5) LEFT by 5 cols: c cols 34-38 -> 29-33, 9 cols 34-38 -> 29-33. b-bar lost 2 more cells (left edge 15->16).
- So UP/DOWN/LEFT/RIGHT all shift the (c + 9) stack by 5 cells in that direction. It's a single combined 5x5 stack object (c on top 2 rows, 9 on next 3 rows).
- The 'n9' encoder earlier was grabbing stray 9s (rows 11-13) — ignore those. The REAL 9-block is the 3x5 attached below c.
- Model is now: one 5x5 object (c 2 rows + 9 3 rows) that translates by 5 per directional action. Walls (3) are static. Need to find what "winning" is (probably align stack to a target/dock).
- b-bar depletes 1 col per action = move counter / resource.


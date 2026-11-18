# Hypotheses — Level 0 (start, turn 0)

## Grid layout observations (step 0)
- Big 3-colored structure (color 3 = 892 cells) rows 8-49, cols 14-53: looks like a maze/arena.
- Color 12 block: 2x5 (rows 45-46, cols 34-38), sits just above a 5x3 color-9 block (rows 47-49, cols 34-38).
- Colors 0/1: small 5-cell glyph (rows 31-33, cols 20-22) — candidate PLAYER marker (head at (31,21)?).
- Color 9 (45 cells) in many clusters: rows 11-13, rows 25-30 (a vertical band at col 27?), rows 47-49, and bottom-left rows 55-60 cols 3-8/7-8 (outside the 3-structure).
- Bottom band: color 11 (84 cells) rows 61-62 cols 13-54 — candidate HUD/status strip. Color 8 (12 cells) rows 61-62 cols 56-63: pattern of 2-wide blocks (56-57, 59-60, 62-63) — could be an icon.
- Colors 4 and 5 = background/sand (outside arena).

## H1: Player glyph is the 5-cell 0/1 cluster near (31-33, 20-22); moves with UP/DOWN/LEFT/RIGHT.
- status: untested
- test: submit RIGHT; if glyph shifts right by 1 (same 0/1 pattern), confirmed; if nothing changes, glyph is static decoration.

## H2: Color 12 = goal/target (or pick-up); color 9 clusters = collectibles or hazards.
- status: untested. Decide by what the player touches first and what changes.

## H3: Bottom rows 61-62 (color 11 strip + color 8 blocks) = HUD (step counter/icons).
- status: untested. Check whether it changes on any move.

## H4: Arena walls = color 3; player moves on the 3 background? But player glyph is inside the 3-area...
- Actually the 0/1 glyph is INSIDE the 3 structure. So 3 = floor, something else = wall? There is no obvious wall color yet. Maybe 3 = floor and there are no walls yet, or 9 = obstacles.

## H1 REJECTED: 0/1 glyph is NOT the player. It's static.
- evidence: RIGHT moved 12/9 block, not the glyph.

## H5: Player = 12/9 combined block, moves 5 cells per action.
- status: testing (LEFT should move -5)
- test: submit LEFT; expect 12 block back at cols 34-38.


## H5 CONFIRMED: Player = 12/9 block, step size 5.
- evidence: RIGHT moved +5, LEFT moved -5. Both 52 diff pixels.

## H6: Goal = 0/1 glyph at (31,21). Reach it by aligning player center to (32,21).
- status: testing (submitting 6-step BFS path)



## Turn 0 — initial state (level 0)
Playfield: rows 10–53, cols 0–63. Top/bottom (rows 0-9, 54-63) = border (3); row 63 = 0 (status band?).
- col 32 is a wall of 0s, splitting the field into LEFT (cols 0-31) and RIGHT (cols 33-63).
- LEFT half is color 4 with 4x4 darker(0) patches at cols 8-11, every 6 rows (r10,16,26,46).
- RIGHT half is color 5, same 4x4 0-patch pattern at cols 8-11, 33-36, 43-46, 53-56.
- LEFT objects (color 2 = maroon 2x10 blocks; accents inside):
  * (r18-19, c10-15): 2-block, 9-inside at r17? no — 9 at rows 18-23 c8-11... mapping: 8/9 are 6x5 squares
  * 8-square at (r17-21, c42-47) [right], 9-square at (r20-23, c8-11) [left]
  * 9-square (r34-38, c42-47) right; 8-square (r30-33, c12-17) left w/ d-square (r30-33,c18-21) beside it
  * 2-blocks: (r18-23,c22-27) w/ 11-core(r20-21,c24-25); (r38-43,c8-13) w/ 14-core(r40-41,c10-11)
- RIGHT: 8-square (r17-22, c42-53ish?) — need cleaner extraction.
- Small 0 "gaps" inside 5-field at (r16-19,c33-36),(r33-36,c43-46),(r36-37,c33-36)...
Hypotheses (untested):
- H1: 4x4 0-patches are holes in walls that can be toggled open.
- H2: 8/9 are "player" or "key" entities; 2-blocks are targets; 11/14 cores are locks.
- H3: arrow keys move an entity (which one? none has a clear player sprite).
- H4: CLICK toggles patches.
Next: write initial helper (encode grid into objects), then single probe action LEFT/UP to see what moves.


## Turn 1 — after UP
- core14 (player) moved from (40,41,10,11) to (38,39,10,11) — UP by 2 rows
- It's inside 2-block (38,43,8,13); the block didn't move
- Row 63 status band: "3000..." (was all zeros) — col 0 now shows '3'
- Hypothesis: player moves in 2-row steps within its block
- Next: press UP again to see if it stops at block boundary or exits


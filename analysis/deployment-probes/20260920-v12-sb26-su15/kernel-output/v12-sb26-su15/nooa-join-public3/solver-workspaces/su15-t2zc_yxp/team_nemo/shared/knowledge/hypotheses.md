# Hypotheses

## Measured rules (level 0)
- Every CLICK consumes 2 black HUD cells at the right end of row 63 (shrinking right-to-left). Start: 62 black cells (col 0..61). After 5 clicks: 52 black (col 0..51). Cost is 2/click regardless of target.
- CLICK x y where x=row index? No — x appears to be column, y=row? Let me check: I submitted "CLICK 48 15" and nothing moved in the field. I submitted "CLICK 10 53" and the black square at (52-54,9-11) swapped with white at (58-60,3-5). The click coords (10,53) don't match either block location. So CLICK args are NOT the destination.
- Wait: maybe CLICK x y means click at column=x row=y? (10,53) -> col10 row53. The black square was at rows52-54, cols9-11. YES! col 10 row 53 is INSIDE the black 3x3!
- So CLICK x y = CLICK column x, row y (x first = column, y second = row).

Re-check earlier clicks with this interpretation:
- CLICK 48 15: col48 row15 -> the grey 3x3 at rows11-13 cols43-45? No, row15 col48 is... the white blob area (rows11-13, cols42-46)? row15 is below that. Actually the white 9-blob is at rows 11-13. Row 15 col 48 is empty grey field. So that click hit empty field -> no effect (just HUD cost).
- CLICK 20 43: col20 row43 -> on the diagonal trail? The trail goes from ~(17,46) to ~(57,6). At row43, the trail would be around col... let me compute: it's roughly col = 46 - (row-17). At row43: col ~ 26. Col20 row43 might be off the trail. No effect.
- CLICK 44 13: col44 row13 -> the grey 3x3 square is at rows 11-13, cols 43-45. YES! That's inside the grey square. But no playfield effect was seen.
- CLICK 10 53: col10 row53 -> the black 3x3 at rows52-54, cols9-11. YES. This caused the swap.

REVISED:
- Clicking the grey 3x3: no effect (just HUD).
- Clicking the black 3x3: SWAPS it with the white 3x3.
- Both become white after swap? Or the rendering changed because of selection state?

## Object inventory (after step 5)
- Grey background (4) top half, (5) bottom half, split at row 10.
- Diagonal grey trail from top-right to bottom-left.
- White blob (9) at rows 11-13, cols 42-46 (5x3 area).
- Grey 3x3 square at rows 11-13, cols 43-45 (inside the white blob).
- Two 3x3 squares at bottom-left: one at (52-54, 9-11), one at (58-60, 3-5). Both now white (15) after the swap.
- HUD: row 63, black (0) cells = budget, 2 consumed per click.

## Objective hypothesis
Move the black square along the trail to reach the white blob / grey square destination.
The swap suggests clicking the black square moves it to the other 3x3 position.
Maybe: click black square -> it moves to next waypoint on the trail?

| hypothesis | status | evidence / deciding test |
|---|---|---|
| sprite at (31-33,20-22) is player (0/1 2x2) | untested | submit UP; check if the 0/1 pattern translates |
| 9 cells = trail of player path | untested | after move, see if 9 appears at old position |
| 12 block (rows45-46,cols34-38) = goal | untested | reach it, see level_up |
| b bottom band = HUD/water, 8 = counter | untested | do actions change 8 pattern? |


## Turn 1 findings (after LEFT):
- Player sprite did NOT move. diff_pixels=52, changed_bbox=[45,13,62,38]: the 12-block (rows 45-46, cols 34-38) slid LEFT by 5 cells; the 9-block (rows 47-49, cols 34-38) slid LEFT by 5 cells; (61,13),(62,13): 11->3.
- H1 (sprite translation) WEAKENED: player dot unchanged.
- New: the 12/9 blocks in lower-mid grid shifted left 5 on LEFT action => maybe "LEFT" is a camera/world shift (scrolling) rather than player motion, or moves certain objects.
- (61,13): 11->3 at bottom band: suggests bottom band is shifting left too? only col 13 changed from 11 to 3 (edge of b-band).
Decision: test with RIGHT (expect blocks shift right by 5) to confirm scroll hypothesis.


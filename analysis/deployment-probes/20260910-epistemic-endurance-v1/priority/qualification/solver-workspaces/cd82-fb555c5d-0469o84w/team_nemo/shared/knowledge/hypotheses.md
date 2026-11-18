# Hypotheses

## Initial layout (step 0, level 0)
Colors: 5=gray bg (3236), 4=blue (164), 3=green (359), f=white (143), 0=black (164), 2=red (30).
- Border: row 0 all green(3); col 63 all blue(4); bottom-left region rows 6-8 all green; rightmost columns 48-62 gray(5) with some blue(4).
- White squares: rows 8-12 cols 3-13 (5x11?); actually rows 8-12, cols 3-12.
- Diamonds: rows 3-6 cols 10-13 (diamond shape) and rows 3-6 cols 35-38.
- Black cells: rows 3-6 cols 3-9 (7x4=28), plus 4 cells at rows 7 cols 38-41, plus row 7 col 11-14? Check.

## H1 (step 1): "UP" moves the player up
Evidence: step 1, 1 cell changed: (63,63) 4->5. This contradicts "UP moves up" — only the bottom-right corner changed color.
Possible: (63,63) is a step counter or player indicator. Need to check where player is.
Test: try LEFT, RIGHT, DOWN to see which corner/cell changes.

## H2 (step 2): LEFT moves the red/white rectangle left
Evidence: step 2 (LEFT): 200 cells changed in bbox rows 21-37, cols 14-38.
- Red(2) count: 30 -> 43 (grew)
- White(15) in game area: grew
- The object appeared to shift left AND down, with shape changes
- Before: rectangle at rows 24-32, cols 25-38 (red border, white interior, row 32 partially gray)
- After: shape at rows 21-33, cols 14-30 (red+white, different shape)

This is NOT a simple translation. The object deforms when moving.
Possible: the object is a "ball" or "blob" that rolls/slides with gravity.
Or: the object is a fixed shape but the VIEWPORT scrolls.

Need to test RIGHT to see if it reverses, and UP/DOWN for vertical movement.


## Updated understanding (step 14)
- Ball (red+white rectangle) at rows 24-32, cols 25-38. Can move LEFT/RIGHT only (UP/DOWN blocked).
- SPACEBAR collects black cells adjacent to/below the ball. Collected 50+15=65 cells so far.
- 99 black cells remain: top-left (rows 3-7, cols 3-12), top-right (rows 3-7, cols 36-38), bottom (rows 39-43, cols 27-36).
- CLICK on black block had no effect (only border cell changed).
- Border cell (63,*) changes on every action (seems like a step counter).
- Objective hypothesis: collect all black cells to complete the level.


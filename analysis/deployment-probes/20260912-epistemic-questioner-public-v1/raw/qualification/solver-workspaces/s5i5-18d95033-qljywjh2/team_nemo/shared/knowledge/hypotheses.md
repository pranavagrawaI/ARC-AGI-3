# Hypotheses (level 0)
Measured facts (steps 0-17):
- Only RESET, CLICK available. Any CLICK (object or background) adds 1-2 yellow(4) cells
  to the bottom bar (row 63), growing from the RIGHT (col 63) leftward. Objects NEVER change.
- Bar now 22/64. No level transition yet. state NOT_FINISHED.

Objects (static so far):
- Top-left 3x3 (rows 9-11, cols 29-33): border 14(cyan), center 13(orange) = "button B"
- Top-right plus (rows 9-11, cols 50-53): 13(orange) plus = "target P1"
- Big square (rows 18-24, cols 35-47): nested rings G/Y/C with a B(3) vertical line down middle
- Small rect (rows 27-34, cols 9-11): 3-tall column, B top row + C fills, 13 at (34,10)
- Big square 2 (rows 35-47, cols 22-28): nested G/Y/C with B at row 41 (middle)
- Bottom-left plus (rows 51-53, cols 7-11): 13(orange) plus = "target P2"
- Bottom bar row 63: all 3(blue) except right side filling with 4(yellow)

H1: The bar is the win condition — fill all 64. UNTESTED (only 22/64).
H2: The objects form a pattern-matching / counting puzzle; the bar is a counter of
    something (clicks? correct clicks?). The orange plus + orange-center button are
    the "special" markers.
H3: Objects encode a specific target count; need to reach that count on the bar.

Open: what makes the level complete? Need to test object-specific clicks vs the bar.


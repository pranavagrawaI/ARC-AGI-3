
# Level 0 hypotheses

## Observed (step 0, measured from grid_array)
- Frame 64x64, bg=4 (light gray).
- Rows 0-7, cols 17-45: a black(5)-bordered 28x8 bar containing four 6x6 hollow
  squares at cols 18-23 (yellow=9), 26-31 (light green=e), 34-39 (pink=b),
  42-47->41-45 (white=f). 4x4 gray(4) holes in the middle of each square.
- Rows 24-35, cols 17-45: a large hollow rounded-square ring, corners 0 (magenta),
  edges 8 (orange).
- Inside the ring at rows 29-30: four 2x2 red(2) squares at cols 22-23, 28-29,
  34-35, 40-41 (centered under each upper square).
- Row 53: full-width dark red(2) line (64 cells).
- Rows 57-60: four solid 4x4 colored squares (col offset x): e@18-21, f@26-29,
  9@34-37, b@42-45 — same x-centers as the upper squares.

## Hypotheses (objective + mechanics)
- H1: a sliding/puzzle game — press SPACEBAR to cycle something;
  red squares / bottom blocks must be moved/matched.
- H2: CLICK selects or moves a colored square; UNDO reverts; RESET restarts.
- H3: goal is to align bottom 4x4 blocks (e,f,9,b at 18,26,34,42) with the
  target positions; red markers may be targets.

## Untested
- What SPACEBAR does. What CLICK does. Whether any action moves the red 2x2
  markers or the bottom 4x4 blocks.


## Step 1 — SPACEBAR (measured)
- Outcome: exactly 1 pixel changed: cell [53,63] red(2) -> 3 (orange? need to verify)
- Row 53 is the full-width dark-red(2) "progress line"; SPACEBAR advances it
  by 1 cell from the right end, changing color to 3.
- Hypothesis: SPACEBAR fills/erodes the progress line by 1 cell per press.
  The line might need to be completely converted, or it might be a
  "timer/progress bar" toward level completion.
- Open: what color 3 is (check palette); how many presses total; does the line
  fill left-to-right or right-to-left; interaction with red 2x2 markers inside
  the ring.


## Structure (measured, step 0-4)
- 4 large colored squares: blue-ish (color 9: rows 1-6, cols 18-23), green (14), yellow (11), white (15), each 6x6 with B-colored pattern inside.
- Ring (color 0 white border): rows 24-35, cols 17-46.
- Inside ring: red 2x2 line at rows 29-30, cols 22-23, 28-29, 34-35, 40-41.
- 4 more red 2x2 blocks at bottom: rows 57-60, cols 18-21, 26-29, 34-37, 42-45 (NNNN/pppp/bbbb/YYYY colored 4x4).
- Row 53: full-width dark red line (color 2). SPACEBAR converts 1 cell/press right-to-left (2->3).
- CLICK 22 29 (on 2x2 red line at rows 29-30 cols 22-23): blocked/no_impact.

## Hypothesis
- The red 2x2 line inside the ring is the "puzzle" — it's a sequence of 4 markers.
- SPACEBAR is a timer/progress, not the puzzle mechanic itself.
- CLICK blocked on line cells. Maybe CLICK on the colored squares or 4x4 blocks does something.
- OR: the puzzle is to drag/move the 2x2 line to align with something.
- OR: SPACEBAR sweeps and when it reaches the line's position (cols 22-41), something happens.
  Line is at cols 22-41; sweep is at col 61 (3 cells from right). 61-22=39 more presses to reach col 22.


## Step 39 — CLICK 22 29 (BREAKTHROUGH)
- Clicking on a dot while a block is selected TRANSFERS the block's color to the dot.
- Dot 1: was 2x2 color-2 (dark red) → now 4x4 color-9 (orange)
- 9-block: frame removed (deselected), center changed from 9 (orange) to 2 (dark red)
- Sweep advanced: col 62 (2->3), now 2 cells wide
- The dot expanded from 2x2 to 4x4 when colored.
- Top panel order: 9, e, b, f → dots should probably be filled in same order.
- Next: select e-block, click dot2; select b-block, click dot3; select f-block, click dot4.


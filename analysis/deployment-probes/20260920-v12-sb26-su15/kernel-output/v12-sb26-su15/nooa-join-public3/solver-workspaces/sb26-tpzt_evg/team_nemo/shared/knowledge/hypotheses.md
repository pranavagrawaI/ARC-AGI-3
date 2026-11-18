# Hypotheses

## Game Structure
- 4 colored boxes at top (rows 0-7): yellow(9) cols 17-24, cyan(E) cols 24-31, magenta(B) cols 31-38, white(F) cols 38-45
- Each box has a nested pattern: outer border of color, inner 4x4 of color 5 (light gray)
- Gray box in middle (rows 24-35, cols 17-46) with 4 green dots at (29,24)(29,30)(30,24)(30,30)
- Full-width green band at row 53
- 4 colored 4x4 squares at bottom (rows 57-60): cyan(E) cols 18-21, white(F) cols 25-28, yellow(9) cols 32-35, magenta(B) cols 39-42
- Available actions: RESET, SPACEBAR, CLICK, UNDO

## Hypothesis 1: Color matching / puzzle
- Top boxes have colors 9, E, B, F
- Bottom squares have colors E, F, 9, B (different order)
- Maybe need to rearrange top boxes to match bottom order?

## Hypothesis 2: The green dots in the gray box are important
- 4 green dots at specific positions in the middle gray box
- The full-width green band at row 53 might be a goal line

## Open Questions
- What does SPACEBAR do?
- What does CLICK do (click on what)?
- What is the objective?
- Is there a player character?

## SPACEBAR mechanism (revised, step 2)
Claim: SPACEBAR paints the rightmost remaining green cell of row 53 orange.
Evidence: step 1 turned (53,63) 2->3; step 2 turned (53,62) 2->3, leaving both orange.
Tested scope: 2 presses from the initial state.
Open: does it stop at col 0? what completes a level? (fill whole band? then what?)
RETRACTED: "SPACEBAR moves a single cursor" — the old cell did not revert.


## SPACEBAR mechanism (confirmed over 3 presses)
SPACEBAR converts the rightmost remaining green cell of row 53 to orange, moving left.
Evidence: steps 1-3: (53,63)->(53,62)->(53,61) each 2->3, prior orange cells persist.
Open: purpose? Does it complete a level when fully filled? What is row 53 for?


## CLICK behavior (step 4)
Claim: CLICK x y on an object draws a black selection border around it.
Evidence: step 4: clicking bottom cyan square (19,58) produced a 6x6 black border around the 4x4 square.
Tested scope: 1 click on 1 object.
Open: what does a second click do? Can selected objects be moved?


## CLICK mechanic (confirmed, step 5)
Claim: CLICK is a two-step move: click source object to select, click destination to place.
Evidence: step 4 selected bottom cyan square (border appeared); step 5 clicked middle box
  and the square moved there (4x4 block at 28-31,27-30), bottom area cleared to 2x2 green remnant.
Tested scope: 1 object (cyan) moved from bottom to middle box.
Open: can objects be moved between top boxes? What's the winning condition?
  Does the middle box need a specific arrangement?


## Goal hypothesis
Claim: Arrange 4 bottom pieces in middle box matching top box left-to-right order (9, E, B, F).
Evidence: Top shows 9,E,B,F. I placed E at pos2, F at pos4. Green dots at 4 positions 
  (cols ~22, 28, 34, 40) seem to be slots.
Tested scope: 2 of 4 pieces placed.
Open: Does placement order matter? Is there a check when all 4 are placed?


## H1: Level requires all 4 pieces placed + some confirmation action (SPACEBAR?)
Evidence: All 4 placed (steps 5,7,9,11) but NOT_FINISHED at step 11.
Untested: Whether SPACEBAR or clicking a specific location triggers level completion.

## H2: Pieces need to be in specific positions matching top reference
Evidence: Middle box pieces at cols 21-24,27-30,33-36,39-42 vs top at 19-22,24-27,29-32,34-37.
Untested: Whether position mismatch is the issue.


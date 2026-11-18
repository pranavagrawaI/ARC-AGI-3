## H1: player = 2x2 blob at (31-32, 21-22) in the 3-floor chamber
Evidence: colors 0,1 only appear at rows 31-33, cols 20-22, forming a plus-shaped
cluster in open floor (color 3) inside a color-5 bordered box.
Scope: untested — no action executed yet. Open: whether arrows 9/12/c are player,
keys, goals, or decoration.

## H2: objective = traverse maze (walls color 5, floor 3) to some exit/gate
Untested. The 5x5 sprite at (8-16, 31-37) looks like a door/gate icon.

## Open questions (answer by acting)
- Q1: which cell does an arrow move, if anything? (probe: RIGHT)
- Q2: does hitting wall color 5 block movement?

## H3: arrow-pattern object at rows 45-49 is the movable agent
Evidence: step 1 RIGHT -> it shifted cols +5; blob did not move.
Open: step size (5?), what stops it (walls? bounds?), what the blob/door/HUD are.


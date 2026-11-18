# Game mechanics (evidence-based)

## General
- 64x64 grid. Colors: 4 = dark background/wall, 3 = gray floor (walkable), 5 = dark panel fill,
  9 = blue glyph / player lower body, 12 = orange player head (rows top), 11 = yellow (HUD bar + rings),
  8 = red (bar end markers), 0/1 = the white "+" pickup.
- Player is a 5x5 sprite (2 rows orange 12 on top of 3 rows blue 9). One action = one 5x5 tile step.
- Movement is tile-quantized: tile centers are at rows ≡2 (mod 5), cols ≡1 (mod 5) (level 1); derive
  the phase from the player's top-left each level.
- A blocked move produces NO change and does NOT consume budget.
- HUD: bottom row 61-62 yellow bar starting at col 13; each successful action consumes 1 cell
  (consumed cells turn color 3). Level 0 budget 33, level 1 budget 42. Running out presumably = fail.
- Bottom-left 12x10 box (rows 53-62, cols 0-11) = a 2x-scaled glyph display ("current state").

## Level 0 (SOLVED in 19 actions)
- Layout: tile maze (rows 25-49, cols 14-53) with a shaft leading up to a black glyph panel (rows 8-16).
- Panel glyph (target shape) = XXX/..X/X.X ; the bottom-left HUD glyph started as its MIRROR (XXX/X../X.X).
- White "+" pickup in the maze: picking it up made the HUD glyph become the mirror image
  (i.e. it FLIPPED the HUD shape so it MATCHED the panel target).
- After the flip, walking up into the panel (3 UP from maze top) completed the level.
- So: goal = make HUD glyph match the panel glyph (via "+" = mirror/flip switch), then step into the panel.

## Level 1
- Box with target glyph is INSIDE the maze at tile (row 40, col 14); glyph = XXX/X../X.X.
- HUD glyph currently = XXX/..X/X.X = mirror of it → need the "+" flip.
- "+" at tile (45, 49). Player start tile (40, 29).
- NEW ELEMENT: two yellow rings (3x3 with hollow center) at tiles (15,14) and (50,39). Unknown use:
  possibly step-budget refills or optional collectibles. Distances: S->Y1 12, S->+ 17, +->Y2 3,
  Y1->(35,2 above box) 4, +->(35,2) 19.
- Planned route (37 actions, budget 42): URUUUUURRDRDDDDDD (to +) UUUUUUULLLLLLDL (to Y1) DDDDD
  (Y1 -> (35,2) -> into box). Collects Y1 en route.

[checkpoint: model committed UP with no model-authored beliefs update]

[checkpoint: model committed DOWN with no model-authored beliefs update]

[checkpoint: model committed DOWN with no model-authored beliefs update]

[checkpoint: model committed DOWN with no model-authored beliefs update]

[checkpoint: model committed DOWN with no model-authored beliefs update]

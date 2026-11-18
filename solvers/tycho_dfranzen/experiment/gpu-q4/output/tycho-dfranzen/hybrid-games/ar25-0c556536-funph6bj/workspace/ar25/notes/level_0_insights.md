# Level 0 — SOLVED (win on action 15 of 16, winning action = LEFT)

## Board / palette (64x64, colours are ints)
- 9 = blue background (the whole field).
- 5 = **black = the player shape** (also used for the solid border row 63 and the
  action gauge, see below).
- 0 = white accent dots *inside* the player shape (part of its footprint, not holes).
- 4 = dark grey = **mirror preview** of the player (body cells; its accent dots render
  as 11 when they land on the target).
- 11 = yellow = **goal target** silhouette; also the colour of the unused part of the
  right-edge action gauge.
- 10 = pale cyan band = the **mirror line marker** (a 3-column full-height stripe).

## Mechanics (all verified against every observed transition)
1. Arrows translate the player shape as a rigid body. **One press = 3 cells** (the
   stroke thickness); the shape lives on a 3-lattice. Verified for LEFT and DOWN.
2. The dark preview is the player footprint (body ∪ white dots) reflected as
   `c' = 2A - c` with **A = 31 in level 0** (band drawn at cols 33-35, so the band is
   *not* exactly the axis — measure the axis from the shapes, do not assume the band
   centre). Rows are unchanged: the mirror is a pure left/right reflection.
3. The preview is redrawn every frame and is painted **over** the yellow target
   (yellow cells it covers become 4). So "yellow disappeared" ≠ progress lost.
4. **Win condition = the preview footprint exactly coincides with the target footprint.**
   Partial overlap does NOT win: at turn 14 the preview (cols 48-56) already overlapped
   the target (cols 51-59) by 6 columns and nothing happened; one more LEFT made it
   exact and the level ended.
5. Right edge col 63 is an **action-budget gauge**: black fills downward from row 0,
   one cell per action taken (row k turns 5→11... i.e. black rows = 0..actions_used-1).
   Everything below it is 11. Read it to count actions spent when the header is absent.
6. No obstacles in level 0; the player never touched a wall, so wall/clamping behaviour
   is still unobserved.

## Level 0 route (15 actions; 1 wasted)
Start: player rows 15-23, cols 18-26. Target Γ: rows 45-53, bar cols 51-59 / stem cols 51-53.
Needed preview Δrow +24 (8 DOWN) and Δcol +3 (1 LEFT, since player Δcol = −preview Δcol).
Actual: 1 LEFT at the start (cols 18→15, a wasted press), then 11 DOWN + 3 LEFT.
**Optimal = 9 actions: 8 DOWN then 1 LEFT.** Lesson: compute the whole translation
first, then press exactly Δ/3 times per axis — do not nudge-and-see.

## Method that worked
Represent each shape as a set of (row, col) over its *footprint* (body colour ∪ accent-dot
colour), take the bbox row-run signature to match shapes up to translation, then solve
Δrow, Δcol for the preview and negate Δcol for the player. That is a closed-form plan;
no search needed.

## Next-level variation to expect
Level 1 keeps the same palette but: the player is on the **right** of the band and the
preview on the **left** (sides swapped), the player shape is a 3-segment Z/S (not a Γ),
the target is an S in the lower-left, and the band sits at cols 36-38 (axis measured
A = 37, i.e. `c' = 74 - c`, here the band centre *does* equal the axis).

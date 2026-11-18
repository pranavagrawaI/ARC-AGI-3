# world_model.md — notes for the builder (level 0 solved, level 1 in progress)

## State representation that is sufficient
`grid` 64x64 int. Derive three objects and nothing else:
- `player` = footprint of the moving blob = cells of colour 5 (excluding border row 63,
  col 63) ∪ the white (0) accent dots inside its bbox.
- `preview` = footprint of colour 4 ∪ the accent-dot cells inside its bbox (dots render
  as 0 or 11 depending on what is underneath).
- `target` = footprint of colour 11 excluding col 63 (the gauge).
Plus `axis A` (reflection constant) and `step = 3`.

## Transition law (verified on all 15 level-0 transitions, 100%)
- `UP/DOWN`: player rows ±3, cols unchanged. `LEFT/RIGHT`: player cols ∓/±3.
  Rigid translation; no rotation, no deformation, no gravity, no per-turn timer.
- `preview = {(r, 2A - c) : (r,c) in player}` — recomputed from scratch each frame
  (a *derived* channel, not an independent object). Level 0 A = 31; level 1 A = 37.
- Drawing order: preview is painted over the target (target cells under it read 4).
- Action gauge: col 63, rows 0..(actions_used-1) are 5, the rest 11. Deterministic
  bookkeeping, safe to ignore for planning except as a free action counter.

## Outcome
`win` iff `preview_footprint == target_footprint` (exact set equality, translation-only).
Overlap is not enough (observed: 6-column overlap, no win). Terminal frame = the frame
in which the equality first holds, with the preview's accent dots rendered as 11.

## Planner (closed form, no search)
Δrow_prev = target.top - preview.top ; Δcol_prev = target.left - preview.left
(both must be multiples of 3, and the row-run signatures must already match up to
translation — if not, the level needs rotation and this model is incomplete).
Then player needs Δrow = Δrow_prev, Δcol = **−**Δcol_prev.
Presses = |Δrow|/3 DOWN/UP + |Δcol|/3 LEFT/RIGHT. Level 0: 8 DOWN + 1 LEFT = 9.

## OPEN PROBLEM carried into level 1 (material assumption!)
Level 1 start: player rows 18-29 cols 45-59 (right side), preview rows 18-29 cols 15-29
(left side), target rows 42-53 cols 3-17, A = 37 (`c' = 74 - c`).
Row-run signatures match target↔preview exactly, so translation-only applies:
preview needs (Δrow +24, Δcol −12) ⇒ player needs (Δrow +24, Δcol **+12**), i.e. 8 DOWN
+ 4 RIGHT = 12 presses. But the player's right edge is col 59 and the interior ends at
col 62 (col 63 is the gauge border) ⇒ the player appears to run out of board after
1 RIGHT. So one of these must be false:
  (a) the right border is passable or wrapping,
  (b) `step` is not 3 in this level,
  (c) the reflection axis moves with the player (or the player may cross the band and
      the mirror re-appears on the other side),
  (d) the win condition here is not exact preview/target coincidence.
**Probe:** press RIGHT once from the start frame and record player Δcol, preview Δcol
and whether the player clamps at the border. Then 1-2 more RIGHTs to see what happens
at the wall. Encode the observed clamping/wrapping rule before trusting any plan.

## Rendering gotcha
The inlined display grid is a *lossy* downscale of the real 64x64 frame — trust
`wmlib.frames()` / the on-disk arrays, and compute footprints with numpy, not by eye.

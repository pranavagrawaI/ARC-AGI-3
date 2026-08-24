# World Model Notes - Level 0

## Confirmed Mechanics
- Grid is 64x64. Background 9, wall 10 at cols30-32, right UI border col63 value 11, bottom border row63 value5.
- Two movable rigid bodies:
  1. Shape A: value5 L-shape with interior zeros (value0). Connected component of values {0,5} size 45.
  2. Shape B: value4 solid L-shape. Connected component value4 size 45.
- Vertical moves UP/DOWN: both components translate together vertically by 3 cells per press.
- Horizontal moves LEFT/RIGHT: components move away from the central wall in mirrored fashion.
  - LEFT (dx=-3): Shape A (left of wall) moves left 3, Shape B (right of wall) moves right 3.
  - RIGHT (dx=+3): Shape A moves right 3, Shape B moves left 3.
  This is implemented by mirroring dx for components with mean col > wall_center_col (31).
- Wall interaction:
  - Shape A 5-cells can overwrite wall cells (10 -> 5). Zeros (0) inside Shape A cannot overwrite wall; they are removed if their target cell is wall.
  - Shape B 4-cells cannot overwrite wall; cells whose target is wall are removed.
  - Rigid translation is applied to each component; cells that would land on wall are either overwritten (Shape A 5) or deleted (Shape A 0, Shape B 4). Cells landing on background move normally.
  - When a component moves away, cells it vacated restore underlying terrain: wall columns 30-32 return to 10, elsewhere to background 9. This was the missing rule causing divergence at turn 9 DOWN.
- Zero-cell painting rule (confirmed this pass):
  - Shape A zero cells paint 0 over background (9) and over Shape B cells (4), but do NOT overwrite wall (10) nor target cells (11). Target cells remain visible through zero holes. This explains divergence at turn 11 RIGHT (model previously painted zeros over Shape B) and turn 33 RIGHT (model previously painted zeros over target 11).
- Move counter displayed on right border column63: each arrow press changes the topmost remaining 11 cell to 5, filling downwards. CLICK and other actions do not affect counter.
- SPACEBAR increments move counter by one (topmost 11 ->5) without moving shapes.
- UNDO reverts the most recent shape movement (UP/DOWN/LEFT/RIGHT) to the positions held before that movement, popping a history stack; it does not revert the move counter and it skips non-movement actions (e.g., SPACEBAR) when selecting which move to undo.
- CLICK at (16,19) produced no visible change; selection mechanics not confirmed.
- No level_complete or GAME_OVER observed yet.

## Open Hypotheses
- Objective unknown: possibly align L-shapes, fill zeros, reach target. No terminal evidence.
- Selection: actor beliefs mention cursor-based interaction; current evidence shows arrow keys move components with mirrored horizontal rule regardless of click. Might be that click selects, but no effect observed yet. Need probe with click on different cells before moving.
- SPACEBAR behavior confirmed (counter increment only); UNDO behavior confirmed (revert last movement, counter unaffected).

## Model Representation
- State holds grid only; movable components identified each transition via mask !=9,10,11 excluding row63/col63.
- Components labeled via scipy.ndimage.label and shifted rigidly.
  - dy applied equally to all components.
  - dx applied with mirroring for components right of wall.
- State now stores a history stack of previous shape positions (orig_a_r, orig_a_c, orig_b_r, orig_b_c) for each movement action. UNDO pops the stack to restore prior positions; RESET clears the stack.
- Vacating rule (corrected):
  - Movable cells are removed and underlying terrain restored: cells in wall columns 30-32 revert to 10, all others revert to background 9. This ensures Shape A 5-cells that previously overwrote wall restore wall to 10 when the shape moves away.
- Pasting rules:
  - Write into background cells (9) always.
  - Allow 5-valued cells to overwrite wall cells (10 -> 5); zeros and 4-cells never overwrite wall and are deleted if target is wall.
  - Zero cells paint 0 over background (9) and over Shape B cells (4), but are blocked by wall (10) and target cells (11). Implemented as: if underlying grid cell is not 10 and not 11, write 0.
- Move counter updated by painting first remaining 11 in col63 to 5. SPACEBAR increments counter without movement; UNDO does not change counter.

## Next Probes
- Test CLICK on shape border vs interior to see if selection changes which components move.
- Observe terminal condition after repeated moves; objective still unknown.
- Confirm collision handling when components approach wall from left side (Shape A moving left into empty space).
- Probe SPACEBAR and UNDO effects — confirmed.

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

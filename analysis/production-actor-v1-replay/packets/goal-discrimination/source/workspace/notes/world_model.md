# World Model Notes

## Confirmed Mechanics (Level 0)

- Grid 64x64, background color 9 (blue).
- Static elements:
  - Barrier color 10 (a) columns 30-32 rows 0-62, bottom row 63 is color 5.
  - Right column 63 is color 11 (b) initially, bottom cell 63,63 is 5.
  - Right L shape color 11 at ~rows45-53 cols51-59.
  - Bottom row row63 cols0-62 is color5 (black).

- Movable pieces:
  - Piece A: color 4 L-shape, bounding box rows15-23 cols36-44.
  - Piece B: color 5 L-shape with holes (color0), bounding box rows15-23 cols18-26.
  - Both pieces move together as separate components.
  - Movement step size = 3 cells.
  - UP moves both pieces up 3 rows, DOWN moves down 3 rows.
  - LEFT/RIGHT move pieces horizontally by 3 cols with mirrored direction across barrier:
    - Left-side pieces (centroid col < 32) move in action direction.
    - Right-side pieces (centroid col >= 32) move opposite to action direction.
    - Observed: LEFT moves piece5 left 3, piece4 right 3.
  - Movement preserves entire bounding box content including interior zeros and background 9 cells inside bbox. Old bbox area cleared to background 9 except overlapping region.

- Move counter:
  - Each action increments a monotonic counter.
  - Counter rendered as vertical bar on column 63, top-down, painting cells from row0 downwards to color5 (black), overwriting color11.
  - After first action, (0,63)=5; after second, (1,63)=5, etc.
  - Counter stored in State.move_count.

- Segmentation approach:
  - wmlib.segment groups by color. Movable components identified as color 4 or 5 components with bbox row max !=63 (exclude bottom row) and c1 <63 (exclude column 63 markers), height>=2 width>=2.
  - Move by extracting subgrid, clearing old bbox to 9, pasting at offset.

## Open Questions

- SPACEBAR, CLICK, UNDO effects unknown.
- Objective unknown. Hypotheses: shape matching via symmetry between left pieces (4 and 5) and right L shape (11). Possibly need to align shapes using moves.
- Win condition not observed; outcome() currently returns ongoing.
- No GAME_OVER observed.

## Probes

- LEFT/RIGHT horizontal mirrored movement confirmed on 1 transition.
- Test SPACEBAR for rotation or selection.
- Test CLICK to see if piece selection changes movement.
- Observe terminal state to infer objective.

Last updated: simulation_accuracy=1.00 on 3 observed transitions; LEFT/RIGHT mirrored movement encoded.

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

[checkpoint: pass ended with no model-authored update]

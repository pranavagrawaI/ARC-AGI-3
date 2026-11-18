# Level 0 — observations & hypotheses (updated T10)

## Objects
- **Mini-box** (rows3-12, cols3-13): top 5 rows WHITE (3-7), bottom 5 rows PURPLE (8-12). Target pattern.
- **Panel** (rows0-8, cols16+): WHITE swatch rows3-5 x36-38; PURPLE swatch rows3-5 x42-44; underline row7 cols35-39 (white) or cols41-45 (purple) marks selection.
- **Tray**: light-grey(2) box rows24-32, cols25-38 (open bottom at row32). Holds paint block 7x12 at rows25-31, cols26-37.
- **Canvas**: white 10x10, rows34-43, cols27-36.
- **Bar**: row63 charcoal(4), 64 at start, shrinks from the RIGHT end. Currently 58 (rightmost fill col57).

## Confirmed mechanics
- Swatch click: selects color, repaints TRAY block, moves underline. (T2 white→block white; T3 purple→block purple.)
- **ACTION5 = pour**: animation = block falls from home (25-31) 7 rows to (32-38) then springs back; paints canvas rows it covers = rows 34-38 cols 27-36, with selected color. Drop verified identical in anim 004 (T4), 005 (T5), 009 (T9): top of block 25→32.
- **H_A REFUTED (T8/T9)**: click (43,33) then ACTION5 (T9) still dropped to 32-38 and painted 34-38. Canvas clicks do NOT set drop depth (when ungrabbed).
- **Arrows (T6/T7) do not move block directly** (no visible change; T9 stamp unchanged ⇒ arrows didn't offset the drop either).
- Bar deltas per turn (start 64): T1 -1 (canvas click), T2 0, T3 -1, T4 -1, T5 0, T6 -1, T7 0, T8 -1 (click 43,33), T9 -1 (ACTION5), T10 0 (click block 28,31). Irregular pattern (1,0,1,1,0,1,0,1,1,0) — treat as cost rule unknown, ≥0 per action; 58 left, ample budget.

## State now (end T10)
- Block: PURPLE, home rows 25-31. Underline: under purple swatch. Canvas: rows34-38 PURPLE, rows39-43 WHITE. Bar 58.
- Goal (strong): canvas must become top5 WHITE + bottom5 PURPLE (mini-box). Currently INVERTED.

## Key problem
Stamp footprint is FIXED at rows 34-38 (bottom of canvas unreachable) → need a way to shift the pour +5 rows. Block home position in tray must be the lever.

## T11 result
- ACTION2 after T10 block-click: NO block movement. Bar 58→57.
- ⇒ "click block + arrows" does NOT move the block (grab hypothesis via arrows refuted).

## New leading hypothesis: two-click DRAG (ACTION6 press on block → ACTION6 release on destination)
- T12: ACTION6 (28,31) grab block.
- T13: ACTION6 (43,33) release at canvas bottom → observe where block lands (expect 37-43 if bottom-aligned to click row) and whether rows 39-43 paint PURPLE.
  - If drag paints bottom half purple → canvas fully purple → then: white swatch (4,37), grab (28,31), release (38,33) → top half white → WIN.
  - If drag only moves block (no paint): follow up with ACTION5 to pour from new position.
  - If nothing moves on T13: drag dead; try double-click block, or click block then swatch, or canvas click while block "armed".


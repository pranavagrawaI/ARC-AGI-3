
## Hypotheses

| # | Hypothesis | Status | Evidence |
|---|-----------|--------|----------|
| H1 | The 12+9 fused block (rows 45-49, 5x5) is the controllable entity | **Confirmed** | RIGHT moved it 5 cells right (34-38 → 39-43); top 9-sprite and 0/1 glyph did not move |
| H2 | The 0/1 glyph (rows 31-33, cols 20-22) is the goal | **Untested** | It's a tiny 5-cell marker in the middle of the maze; no other obvious target |
| H3 | The top 9-sprite (rows 11-13, cols 35-37) is a distractor or NPC | **Untested** | It didn't move with RIGHT |
| H4 | The 12-bar on top of the 9-sprite is a "roof" or "carry" indicator | **Untested** | It moves with the 9-sprite as a fused unit |
| H5 | Walls (color 3) block movement | **Confirmed** | Entity surrounded by walls on all 4 sides at (45-49, 39-43) |
| H6 | Below the entity (row 50) is open floor | **Confirmed** | Row 50, cols 39-43 are all color 4 (floor) |

## Deciding tests
- H2: Navigate to the 0/1 glyph and see if level completes
- H3: Try to move the top 9-sprite (can't — we can only control one entity)
- H4: Watch if 12-bar changes/disappears at goal


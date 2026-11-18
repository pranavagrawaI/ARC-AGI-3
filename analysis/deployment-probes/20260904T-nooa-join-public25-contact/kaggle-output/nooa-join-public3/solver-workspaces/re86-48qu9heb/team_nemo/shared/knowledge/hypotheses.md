# Hypotheses

## H1: Cross = "eye" glyph, 9 = "target/donut" glyph, 0 = black dot
Status: observation (level 0 start)
- 3x3 crosses: 4-arms + 11 center at (3,15), (9,6), (9,24), (17,15), (16,48), (24,40), (24,53), (35,48)
- 9-colored objects: 3x3 cross at (24,40) and (24,53); vertical bar 9 (rows 32..58, col 36); horizontal bar 9 (row 45, cols 23..49) with black(0) at (45,36)
- 5 = background, f = bottom border row 63

## H2: The 11 column at col 21 rows 16..38 + row 27 cols 10..32 form a "+"-shaped entity (the player?)
- Intersects the 9-structure at (32..44, 36)? No — the 11 cross at (35,48) is inside the 9 cross region? (35,48) center is inside 9bar? No.
- Test: press UP/DOWN/LEFT/RIGHT and see what moves. If the 11-plus moves, that's the player.
- Deciding test: single LEFT, diff.

## H3: Goal mechanic — cross the player onto the 9-objects / align crosses?
- The row-45 9-bar with black dot at (45,36) looks like a "key slot" or a target line.
- Test after movement discovery.

## Plan
1. Probe LEFT (single action, predicted_path: move — the 11-plus entity is expected to move 1 cell left; if nothing moves, try SPACEBAR).
2. Build encode/predict helpers once movement semantics confirmed.


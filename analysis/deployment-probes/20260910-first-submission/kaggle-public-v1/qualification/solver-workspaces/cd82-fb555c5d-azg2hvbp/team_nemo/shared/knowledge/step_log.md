# Step log

## Turn 0 (step 0, level 0) — initial inspection
Grid layout (measured, rows 0-43 non-background):
- Big background = color 5 (3236 cells)
- Top-right block: color 3 region rows 0-8, cols 18-63, with internal color-4 frame and
  3x3 color-f hole at rows 3-5, cols 42-44 (a "door"?)
- Left block rows 3-12: color 0 hollow box rows 3-7 cols 3-12 (top/bottom walls),
  color-f filled box rows 8-12 cols 3-12 (solid f block)
- Color-4 vertical wall col 16-17 rows 0-17; color-4 rows 16-17 cols 0-17 (L-shape wall)
- Middle: color-2 block with f core: rows 24-32, cols 25-38 (f core rows 25-31 cols 26-37)
- Bottom: color 0 rectangle rows 34-43, cols 27-36 (empty box 10x10)
- Bottom edge row 63 all color 4.

No visible "player" marker. Hypothesis: the player may be one of the f regions or 0 regions;
arrows may move something. Unknown which object moves, and what SPACEBAR/CLICK do.

Carry-forward: first action should be a single cardinal move to identify what moves.

## Turn 1 (step 1) — UP result
- Measured: exactly ONE cell changed: [63,63] 4→5. Row 63 (bottom wall, all 4s) now has a hole at col 63.
- Nothing else on the whole grid changed (rows 0-62 identical).
- Hypothesis H1 (arrows move a player object): NOT supported — no visible player moved.
- New hypothesis H5: arrow keys erode/remove wall cells near a boundary (e.g., UP clears a bottom-row cell from the right end; or clears the cell at the player's row/col against the wall it pushes toward).
- Need: repeat UP, then probe DOWN/LEFT/RIGHT to map the pattern.


## Turn 2 (step 2) — UP again
- Measured: ZERO cells changed. outcome="no_impact", label="blocked".
- So the first UP cleared exactly [63,63] (4→5), and the second UP was BLOCKED (no change).
- H5 (arrows erase one boundary wall cell per press): only partially supported — one cell cleared, then blocked.
- Reinterpretation: UP moved something/changed something internally? But no other pixels changed. 
  Perhaps the "player" is at the bottom row and UP steps the player up INTO the wall cell [63,63],
  leaving it empty (the player cell becomes background 5 = invisible when it's just 'a player on background').
  Then the second UP is blocked by the row above (row 62 is 5s though...). Actually row 62 is all 5s, so a player at [62,63] could move up freely — so blocked is odd unless there's an invisible blocker.
  ALTERNATIVE: UP "digs" one cell from the bottom wall at the current column, then is blocked because it can't dig through... no.
  BEST NEW HYPOTHESIS H6: There is an invisible player at [63,63]. UP moves the player up, leaving its vacated cell as background (5). Player now at [62,63] (invisible on 5). Second UP: blocked?? row 61 is also 5s. Unless player motion leaves a TRAIL and the trail blocks? No trail seen.
  OR H7: UP = "remove one wall cell adjacent to an edge cursor" — first removal [63,63]; second blocked because removal only allowed once per... something.
- Decisive next test: DOWN. If player at [62,63], DOWN should restore... no, it would move player to [63,63] and if it's now a hole (5), grid might show player? We can't see the player.
  Test RIGHT/LEFT instead: if a cursor is at col 63 bottom, LEFT should clear [63,62]? Let's just test DOWN first (cheap, tells us if [63,63] refills or another cell changes).


## Turn 3 (step 3) — DOWN
- Measured: exactly ONE cell changed: [63,62] 4→5. Now row 63 has 5s at cols 62,63.
- So DOWN did NOT refill [63,63]. Instead it carved a NEW cell [63,62] in the bottom wall, adjacent to the existing hole.
- Pattern so far:
  - UP → hole at [63,63] (rightmost of bottom row)
  - UP → blocked (no change)
  - DOWN → hole at [63,62]
- Refined hypothesis H8: There is a "cursor"/player along the bottom row at column c. Each arrow press carves the wall cell at the bottom row in the direction pointed... but UP carved [63,63] and DOWN carved [63,62].
  - Maybe: the player is AT [63, c]. UP tries to move the player up (into row 62); since row 62 is 5 (empty), player moves up and the cell it LEAVES becomes 5. But then [63,63] would be where the player started. Then UP again is blocked (row 61 is also 5, so not blocked...). UNLESS the player can only be in the bottom row? No.
  - Alternative H9: arrows shift a "carver" position. UP = carver stays at col 63 and carves bottom cell; then UP blocked because already carved. DOWN moves carver to col 62 and carves.
  - Or H10: This is a mario-like "bump head" thing: player is a 1-cell entity rendered as... hmm, we never saw any non-wall pixel appear.
- KEY TEST: press LEFT or RIGHT to see which cell gets carved next. If RIGHT carves [63,61] (continuing leftward pattern) or [63,64] (impossible, out of bounds) or something else, the pattern becomes clear.
  Actually current hole region is cols {62,63}. If the "player" is at the rightmost hole edge (col 63) and moves:
  - UP carved 63, DOWN carved 62 → maybe player is at col 63 now, and UP/DOWN move it along the bottom row horizontally? That's weird.
  - Most consistent: each press carves the cell just in front of the player along the bottom. Let me test LEFT: if it carves [63,61], then the carving position decrements with LEFT. If RIGHT carves [63,63] again (already 5) → no_impact.
- DECISIVE: test LEFT next.


## Turn 4 (step 4, level 0) — action LEFT
- Carved [63,61] 4->5 (bottom wall; holes now cols 61,62,63).
- CENTRAL 2/15 BLOB MOVED LEFT ~12 cols and GREW: 114->128 cells, frame(color2) 30->43, fill(color15) 84->91... 
  (counts: before 2:30/15:84, after 2:43/15:85).
  BEFORE bbox rows 24-32 cols 25-38; AFTER bbox rows 21-37 cols 14-30.
- Best rigid translation IoU only 0.45 at dc=-12 -> blob RESHAPES while moving (not rigid).
- 0-rooms (164 cells) UNCHANGED. Top blocks unchanged. Only blob + 1 bottom cell changed.
- Frozen model (world_model.py) MISMATCHED on LEFT (only predicted carve). batch closed.
- NEW H10: horizontal keys move central 2/15 blob by ~12 cols toward direction, growing it.
  UP/DOWN so far: only bottom-row carve, no blob change (turns 1-3). Need to re-verify verticals don't move blob.
- TEST NEXT: RIGHT. Expect blob back ~12 right (grow more?) and carve [63,62] or no carve (62 already 5? no, 62 is 5 now -> carve 63? already 5). Actually holes {61,62,63} all 5; RIGHT may carve nothing new or carve 60? Decide by observation.


## Turn 5 (step 5, level 0) — action RIGHT
- Blob region (rows 15-45, cols 5-49) after RIGHT == blob region BEFORE LEFT, cell-identical.
  => RIGHT is the EXACT INVERSE of LEFT (blob 128->114, position restored). No growth on the way back.
- Bottom row: NO change. Holes still {61,62,63}. RIGHT did not carve a new hole.
- Total diff: 200 cells, all in blob region. 0-cells unchanged.
- REFINED MECHANIC: directional keys move the 2/15 blob ~12 px in that direction (left/right tested).
  RIGHT after LEFT = perfect inverse => motion is stateless / position-based, not a trail.
  LEFT's blob grew (114->128) but RIGHT restored exactly => blob's SHAPE depends on POSITION
  (e.g. it deforms to fit between fixed 0-walls/obstacles, or it's the set of reachable cells
  from an origin, or it 'fills' space toward the wall it's moving away from).
- Carve rule (bottom row): DOWN carved 63, LEFT carved 61, RIGHT carved nothing (62 already hole? no...).
  Holes after: {61,62,63}. DOWN carved 63 (first, was 4). LEFT carved 61. RIGHT: no carve.
  Hypothesis: each direction carves the hole at (cursor + step)? or carves where a "cursor" is,
  and cursor position differs. DOWN:63, LEFT:61 -> cursor moved. Need model: maybe carving happens
  at column = f(direction, existing holes). PROVISIONAL.
- Frozen model mismatched again (expected; world_model only knows carves). batch closed.
NEXT: test UP and DOWN for blob effect (do verticals move the blob too?). Then test LEFT from fresh
position to see if shape growth is deterministic on position.


## Turn 0 (L0, step 0) — first look
- Only actions: RESET, CLICK. No arrow keys / USE / UNDO.
- Grid inventory (non-bg above HUD row 63):
  - color 3 (walls): 16 cells — 3 vertical/horizontal segments: col 27 rows 9-11; col 42 rows 19-23; row 27 cols 9-11; row 41 cols 22-26. (row 63 is a full-3 HUD band)
  - color 2/4: two boxes: B1 r18-24 c36-48 (7x13, interior 4s), B2 r35-47 c21-27 (13x7, interior 4s)
  - color 14 (e, hollow 3x3 rings): one ring r9-11 c28-32 (14 cells) touching wall seg (col27); two rings r20-22 c38-40 and c44-46 INSIDE box B1 (7 cells each = ring + 1? actually 7 cells each)
  - color 11 (b, near-solid): block r28-35 c9-11 (23 cells, hole at (34,10)); two small 3x3 rings inside box B2 at r37-39 and r43-45 c23-25 (7 each)
  - color 13 (d): 10 cells in 4 clusters: (10,31) center of ring at r9-11c28-32; diamond (9,52)(10,51)(10,53)(11,52); single (34,10); diamond (51,10)(52,9)(52,11)(53,10)
- HUD row 63: 64 cells of color 3.
- Reconstruction: encode/grid_from_z round-trips with 0 diff.
- Submitted RESET, expected no_impact (calibration).

## Turn 1 (L0, step 1)
- RESET from fresh start => no_impact (grid unchanged, step advanced). H5 confirmed.
- Next: calibrate CLICK coordinate order at (10,31) — the 13-dot in the e-ring.

## Turn 2 (L0, step 2)
- `CLICK 31 10` => outcome labeled "move"; ONLY change: HUD cell (63,63): 3 -> 4. All 10 13-cells unchanged; all objects unchanged.
- Confirms coordinate order (col, row).
- The 13-dot at (10,31) is inside the e-ring r9-11c28-32; clicking it scored 1 point (HUD +1, new color 4).
- New question: is the score per 13-dot? per ring? Why color 4 in HUD? (4 = box-interior color.)
- Next: click another 13-location to see if HUD converts again and what color.

## Turn 3 (L0, step 3) — CLICK 39 21
- Clicked cell (21,39) = center of B1 left e-ring.
- Effects (vs initial):
  * Top e-ring (r9-11 c28-32): DISSOLVED. Marker d at (10,28). (Was armed by turn-2 click at its center.)
  * B1 left e-ring (r20-22 c38-40): center 4→e, now SOLID 3x3 e.
  * B1 right e-ring (r20-22 c44-46): also became SOLID 3x3 e.
  * B2 top b-ring (r37-39 c23-25): became SOLID 3x3 b.
  * B2 bottom b-ring (r43-45 c23-25): became SOLID 3x3 b.
  * HUD: +2 fours (total 3).
- Interpretation: clicking a ring center triggers a "round" that resolves ALL rings:
  clicked ring → solid, previously-clicked ring → dissolve+marker, other rings → solid.
  HUD +1 per ring resolved? (2 HUD for 5 rings doesn't add up.)
- NEED: RESET + single clean experiment.

## Carry-Forward (turn 3)
- CLICK x y = (column x, row y) per spec.
- Turn 2: clicked top e-ring center (10,31) → HUD+1, ring intact.
- Turn 3: clicked B1 left e-ring center (21,39) → top e-ring shrank (right half gone, d moved to (10,28)), 
  B1/B2 rings filled solid, HUD+2.
- Model: "arm then resolve" — click arms a ring, next click resolves it + fills all others.
- Next: RESET, then clean single-ring experiment.

## Turn 6 (L0, step 6) — CLICK 45 21 (B1 right e-ring center)
- vs initial: top e-ring UNCHANGED (full ring + d at (10,31)).
- B1 e-rings: solid. B2 b-rings: solid. HUD: +3 (cols 61,62,63).
- vs turn-5: top e-ring RESTORED (was shrunk, now full). HUD +2.
- Interpretation: top e-ring is a PULSING indicator (changes on tick 1, reverts on tick 2).
  B1/B2 rings: one-time fill (solid after tick 1, stay solid).
  HUD: +1 (tick 1) + +2 (tick 2) = +3 total.
- Next: click top e-ring center (10,31) to see if it completes the level.

## Turn 7 (L0, step 7) — CLICK 31 10 (top e-ring d-center)
- vs turn-6: B1/B2 rings UNFILLED (solid→hollow). HUD+1 (col 60). Top e-ring unchanged.
- BREAKTHROUGH: clicking d-center UNFILLS all solid rings (reverse of 4-center click).
- Model: 
  * Click 4-center (hollow ring): FILL ring (solid), HUD+1.
  * Click d-center: UNFILL all solid rings, HUD+1.
  * Top e-ring d-center is the "collect" button.
  * B1/B2 rings are "batteries" that charge (fill) and discharge (unfill).
- Goal: fill HUD to 64? By repeatedly charging and collecting?
- Next: click a 4-center again to re-fill, then click d-center to collect. Watch HUD.


## L0 — hypotheses (updated turn 2)
| H | status | evidence |
|---|--------|----------|
| H-spec: `CLICK x y` = column x, row y, 0-indexed | confirmed | `CLICK 31 10` changed HUD cell (r63,c63) 3->4 and only HUD; (10,31) is a 13-dot inside the e-ring — a cell that was hit. |
| H-RESET: RESET from fresh start is no_impact | confirmed | step1 RESET: zero diff. |
| H-HUDmeter: row 63 is a 64-cell meter, initially all color 3; each successful CLICK converts one 3 into the color of... something (first: 4) | probable | HUD went 63x3 -> 63x3+1x4, fill grows right-to-left (last cell). |
| H-objective: fill the meter (64 cells?) to win the level | open | needs more conversions to see target/end condition |
| H-mechanic: clicking a special cell (the 13-dots?) scores 1 and turns it into a 4-marker | open | dot at (10,31) did NOT change to 4 (still 13). Only HUD changed. |
| H-alt: clicking the e-ring center dot "completes" that ring (scores), and rings are the collectibles | open | ring at r9-11c28-32 with center dot (10,31) |
| H-13clusters: 13-cells are all "player-ish" markers (4 diamond clusters) | weak | (34,10) sits in the hole of the b-block; (52,10) diamond; (10,52) diamond; (10,31) dot-in-ring |

## Key objects (L0 initial)
- e-ring (color 14) r9-11 c28-32, center (10,31) = 13. Wall segment col 27 rows 9-11 just left of it.
- e-rings inside box B1 (r18-24 c36-48): r20-22 c38-40 and r20-22 c44-46, centers (21,39),(21,45) are 4s (NOT 13!).
- b-block r28-35 c9-11 (solid 3x8, hole (34,10)=13).
- b-rings inside box B2 (r35-47 c21-27): r37-39 and r43-45 c23-25, centers (38,24),(44,24) are 4s.
- 13-diamond at (9,52)/(10,51)(10,53)(11,52); 13-diamond at (51,10)/(52,9)(52,11)(53,10); single 13 at (34,10) and (10,31).
- Walls (color 3): col27 rows9-11, col42 rows19-23, row27 cols9-11, row41 cols22-26.
## Turn 3 re-analysis (careful)
Turn 2: CLICK 31 10 → (row=10, col=31) = center of top e-ring. Result: HUD+1 only. Ring intact.
Turn 3: CLICK 39 21 → (row=21, col=39) = center of B1 left e-ring. Result:
  - Top e-ring DISSOLVED (marker d at (10,28))
  - B1 left e-ring: center 4→e (now solid 3x3)
  - B1 right e-ring: center 4→e (now solid 3x3)  
  - B2 top b-ring: center 4→b (now solid 3x3)
  - B2 bottom b-ring: center 4→b (now solid 3x3)
  - HUD +2 (total 3 fours)

Hypothesis H-wave: clicking a ring center triggers a "resolution wave":
  - Clicked ring → solid (+1 HUD)
  - Previously-clicked rings → dissolve+marker (+1 HUD each)
  - All other rings → solid (cascade, +0 HUD)
  - Total: 2 HUD for this turn ✓

H-cascade-scope: ALL rings on the board are affected, regardless of box/color.
  - e-rings dissolve (if previously clicked) or fill (if not)
  - b-rings fill
  - Need to test: does clicking a b-ring also affect e-rings?

H-HUD-target: 64 fours = level complete? Or some other target?


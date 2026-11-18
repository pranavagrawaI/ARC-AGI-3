# Step log

## Turn 0 (step 0, level 0) — initial survey
Actions available: RESET, UP, DOWN, LEFT, RIGHT (no USE/CLICK/UNDO).
Grid 64x64. Measured colors: 0(3),1(2),3(892),4(2609),5(439),8(12),9(45),b/11(84),c/12(10).

Map structure (from RLE):
- 4 (navy) = open field, most of map; 5 (purple) = outer frame (cols 0-3 top, rows 52-63 bottom bands, left box).
- 3 (olive) walls form: top box (rows 8-16, cols 32-40, interior 5-floor with 9s inside);
  vertical sliver rows 17-24 cols 34-38; big horizontal bar rows 25-29 cols 14-53;
  vertical corridor hole cols 29-33 rows 30-39; left strip cols 19-23 rows 31-49;
  right mass rows 34-49 cols 34-53 (with 5x5 12/9 block at rows 45-49 cols 34-38).
- Candidate PLAYER: cross glyph of 0/1, 5 cells: (31,21)0 (32,21)0 (32,22)0 / (32,20)1 (33,21)1,
  center (32,21), surrounded on all 4 sides by 3 (cols 14-19 & 23-28 left/right, rows 30 & 34 up/down).
- 9 (maroon) items, 45 cells, 3 clusters:
  A) inside top box rows 11-13 cols 35-37 (6 cells, sparse),
  B) bottom-left 5-box rows 55-60 cols 3-8 (24 cells, box border 5, open to right via col 11 at rows 55-56?),
  C) solid 5x3 block rows 47-49 cols 34-38 (15 cells), topped by 12 (fuchsia) rows 45-46.
- Bottom-right: b (azure) strip rows 61-62 cols 13-54 + 8 (silver) stripes cols 56-63 rows 61-62.
- 12 (fuchsia): only the 5x2 band above cluster C.

Carry-forward: player=cross@(32,21) [untested]; 3=breakable wall [untested]; 9=collectable [untested];
goal hypothesis: collect all 9s (maybe then reach 8/b zone). No UNDO -> every action counts.

## Turn 1 (step 1) — RIGHT probe result
- cross@(32,21) UNCHANGED (cross not the player, or 3 solid to it — but 3 was crossed by the block, so cross likely static decoration).
- 5x5 block (12/9) cols 34-38 -> 39-43 (jumped +5 = own width, overwrote 3s; vacated -> 3).
- b strip rows 61-62: (61,13),(62,13) 11->3: erosion 2 cells from LEFT end.
- Total 52 changed cells; nothing else. 3 is traversable to the block (not a solid wall).
- Frozen model mismatched (predicted cross move) — model must be rebuilt around block as mover.


## turn 2 (after step 2 LEFT)
- Measured LEFT: 5x5 block (top 2 rows c, bottom 3 rows 9) jumped (45,39)->(45,34), 50 cells; b-strip col 14 eroded (2 cells 11->3). Total 52.
- world_model.py v1 written and VALIDATED EXACT on both step1 (RIGHT) and step2 (LEFT): 0 bad cells.
  Rule: block slides exactly 5 cells (1 arena tile) in action direction; destination 5x5 must be all '3'; leaves '3'.
  Every action ALSO erodes 1 col from the LEFT of b(11) strip (rows 61-62), erased->3. Strip was 42 cols (13-54); now 40 -> ~40 actions of "fuel" if erosion is a timer.
- z now: br=45 bc=34 L=15.
- BFS over '3' arena w/ 5x5 block: 34 reachable positions. 0/1 glyph (rows31-33, cols20-22) itself unreachable (blocks overlap); nearest: (40,19) d4, (35,19) d5, (30,34) d3, (30,24) d7.
- PLAN: UP UP UP -> (30,34), then LEFT LEFT -> (30,24) (block right-adjacent to glyph). Watch for level_up / glyph reaction / strip effects.
- CARRY-FORWARD: 5-jump on '3' arena; strip erosion 1 col/action, left end, direction-independent; glyph 0/1 static in both transitions; 9-clusters & 8-pattern & 4/5 borders untouched so far (scenery? untested).


## turn 3 (after step 3 UP) — measured: model v1 MATCHED (fb status=matched, batch_allowed=true)
- Block now (40,34), L=16. Arena tile map computed (top-left rows mult of 5, cols 5t+4):
  open tiles in arena; player P=(40,34); glyph-only tile G=(30,19) contains the 5 0/1 glyph cells (31,21)0 (32,20)1 (32,21)0 (32,22)0 (33,21)1.
- (30,29) and (35,29) are BLOCKED tiles -> direct LEFT path along row 30/35 is cut; must detour via row 25.
- 9-clusters (top box rows11-13 c35-37; bottom-left rows55-60 c3-8) are OUTSIDE the reachable tile arena -> scenery.
- KEY: no USE/CLICK in available_actions -> goal must be achieved by MOVEMENT only. Only target = glyph tile G.
- PLAN (7 actions): UP UP UP LEFT LEFT LEFT DOWN = land ON G (30,19). Model v1 says that last DOWN is BLOCKED (glyph cells not '3') -> predicts block stays (25,19), strip erodes only. If game actually allows landing => level_up likely and model mismatch at step 9 = the signal.
- b-strip fuel: L 16 -> 23 after batch (32 cols left); not a constraint.


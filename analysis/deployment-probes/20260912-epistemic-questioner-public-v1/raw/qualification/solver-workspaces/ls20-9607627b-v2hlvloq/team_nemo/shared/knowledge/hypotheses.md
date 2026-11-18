# Hypotheses (level 0, step 0)

## Measured grid structure (step 0)
- Colors: 4 (2609 cells, background/right region), 5 (439, left band cols 0-3 + top-left 4x4 block),
  3 (892, walls/rooms), 9 (45, orange dots), 11 (84, bottom bar rows 61-62 cols 13-54),
  8 (12, rows 61-62 cols 56-63 blocks), 12 (10, rows 45-46 cols 34-38),
  0/1 (5 cells, rows 31-33 cols 20-22 — odd cluster inside room, maybe a character).
- Top: a box (rows 8-16, cols 28-36, wall=3, inside=5) containing 9-dot pattern rows 11-13.
- A corridor of 3 descends rows 17-24 cols 26-30, into a big room (rows 25-39, cols 14-46)
  with a vertical 4-wall at cols 25-26; 9-dots at rows 45-49 cols 34-38 below the room;
  12s at rows 45-46 cols 34-38 (two-row block).
- Bottom: 9-dots rows 55-60 cols 3-8 (L-shape), bar of 11s rows 61-62 cols 13-54, 8-blocks bottom-right.
- No obvious "player" sprite yet: 0/1 cluster at (31-33, 20-22) is the only non-structural 5x5-ish oddity.

## Working hypotheses (untested)
- H1: 0/1 cluster is the player; arrows move it on the 4s/open space, walls (3) block.
- H2: 9-dots are collectibles; 12s a goal/portal; 11/8 are static decor.
- H3: some other mechanic (e.g. sliding, pushing).

Next: one exploratory move to see what moves and what changes.


## Measured: step 1, action UP
- 52 changed cells, bbox rows 40-62 cols 13-38. NOT a global scroll (best roll mismatch = 0 at k=0 on all axes).
- 12-block (10 cells) MOVED from rows 45-46 cols 34-38 to rows 40-41 (shifted 5 rows up).
- 9-block (15 cells) moved from rows 47-49 to rows 42-44 (5 up) = same 5x3 region.
- Two 11 cells disappeared at (61,13),(62,13); net +2 walls (3) elsewhere. Counts: 12 stayed 10, 9 stayed 45.
- 0/1 cluster (rows 31-33, cols 20-22) UNCHANGED — if it is the player, it did not move on UP.
- So: a 5-row object (12+9 stack) advanced UP toward a 9-group at rows 38-40 cols 34-38 (adjacent after move: 9 at 42-44? no: the 9s sit at 42-44).

## Refined mechanics hypothesis
- H2 (revised): UP moves some object(s) upward 5 cells at a time, or the "player" is the 12/9 cluster, not the 0/1 cluster.
- Open: does DOWN reverse it? Does the object stop on collision? Is 0/1 cluster inert?


## Answer to host question (step0 vs step2, measured)
- UP and DOWN motions are each other's inverse EXCEPT each step also converts
  two 11-cells to 3, column-by-column: UP took col 13, DOWN took col 14.
- step0 vs step2 diff = exactly {(61,13),(61,14),(62,13),(62,14)}: 11->3.
- Conclusion: block motion is reversible; bar erosion is CUMULATIVE (a one-way
  counter that advances with each move, direction-independently).
- New objective hypothesis H4: the 11 bar (rows 61-62, cols 13-54, 84 cells,
  42 columns-pairs) depletes by one column-pair per step; level completes when
  fully depleted or at some threshold. OR the goal is to trigger the depletion
  at a specific spot.
- Note: the 11 bar starts at col 13 and currently spans cols 14-54 (82 cells = 41 col-pairs).
  If each step erodes 2 cells from the left edge, ~41 steps to empty.
- Budget alert: model_admission_seconds_remaining ~482s. Avoid model-building;
  keep using single exploratory actions.


## Step 3 (UP) confirmed
- Bar erosion as predicted: (61,15),(62,15) 11->3. Erosion column sequence so far: 13 (UP), 14 (DOWN), 15 (UP).
- 12 block back at rows 40-41, 9 at 42-44 (same as after step 1). Motion is exact inverse, bar still erodes.
- Erosion rate: 1 column-pair per step, advancing left->right, direction-independent.
- Bar started cols 13-54 (42 col-pairs); 3 consumed; 39 left if linear.
- Open: what happens when bar fully depleted? Level-up at col 54+? Budget note: skip model, keep probing.


## Step 4 (LEFT) — key discovery
- Only 2 cells changed: (61,16),(62,16) 11->3. NO block motion.
- So LEFT/RIGHT do NOT move the 12/9 block but DO erode the bar.
- Revised rule: EVERY arrow action erodes one column-pair from the 11 bar (left->right),
  and only UP/DOWN move the 12/9 block (5 rows each).
- Bar: started 42 col-pairs (cols 13-54); consumed 4 (13,14,15,16); now cols 17-54 (38 pairs, 76 cells).
- Objective still open: is level completion = bar fully depleted (38 more steps), or
  something the UP/DOWN block motion does (e.g., reaching the top box with 9-dots)?
- The 12/9 block currently at rows 40-44 cols 34-38. There is a 9-group at rows 38-40? and
  the top box (rows 8-16, cols 28-36) with its own 9-dots at rows 11-13.


## Step 5 (UP) confirmed
- Block at rows 35-36 (12) / 37-39 (9); bar col 17 eroded. All consistent.
- Block trajectory with UP: 45-49 → 40-44 → 35-39. Moves 5 rows per UP, no wall blocking so far.
- Next UP would put 12 at 30-31, 9 at 32-34. The room's top area (rows 25-29 are open 3-floor? actually rows 25-29 cols 14-46 are 3 = floor).
- Objective still open. Bar: 74 cells = 37 col-pairs left (cols 18-54).
- Plan: continue UP; watch for (a) block hitting the top box area or stopping at a wall,
  (b) bar emptying → level completion. If the block reaches the top box (rows 8-16)
  via the corridor at cols 26-30, that's ~6 more UPs.


## Step 6 (UP) confirmed — prediction matched
- 12 block rows 30-31, 9 rows 32-34; bar col 18 eroded; 52 cells. All as predicted.
- Bar now cols 19-54: 72 cells (36 col-pairs).
- Block will next reach rows 25-26/27-29 (top of the big room). Then corridor rows 17-24.


## Step 7 (UP) confirmed
- 12 block rows 25-26, 9 rows 27-29 (top of big room); bar col 19 eroded. Prediction matched.
- Next UP: 12 at 20-21 / 9 at 22-24 → still inside the corridor band (rows 17-24 cols 26-30 are 3).
  Above that: rows 8-16 top box at cols 28-36 (walls 3, floor 5, 9-dots at 11-13).
- Bar: cols 20-54 left (35 pairs). 7 steps done (4 eroded by: 13,14,15,16,17,18,19).
- Watch: does the block get absorbed into the top box (level up?), stop at a wall, or pass through?


## Step 8 (UP) confirmed
- 12 block rows 20-21, 9 rows 22-24; bar col 20 eroded. Prediction matched.
- Block is now INSIDE the corridor band (rows 17-24, cols 26-30 are 3). Block at cols 34-38.
- Next UP: 12 at 15-16 / 9 at 17-19. Row 16 is the BOTTOM WALL of the top box (cols 28-36 are 3 at row 16).
  So the block (cols 34-38) overlaps the top-box wall region at cols 34-36 row 16.
  This may be where it stops, merges, or triggers something.
- Bar: cols 21-54 left (34 pairs).


## Step 9 (UP) confirmed
- 12 block rows 15-16, 9 rows 17-19; bar col 21 eroded. Prediction matched.
- Block now overlaps top box bottom-wall (row 16 cols 34-36 are 3) and its 9-dots (rows 11-13).
- Next UP: 12 at 10-11 / 9 at 12-14 → inside the box interior (rows 9-15, cols 29-35 are 5),
  adjacent/overlapping the box's own 9-dots (rows 11-13, cols 32-34).
  This is the likely trigger zone for level completion.
- Bar: cols 22-54 left (33 pairs).


## Step 10 (UP): BLOCKED — key finding
- action label 'blocked', outcome no_impact, 0 cells changed, bar did NOT erode (cols still 22-54).
- Block sits at 12@15-16 / 9@17-19, immediately below the top box (rows 8-16, cols 28-36).
- So the block CANNOT enter/overlap the top box: the box's bottom wall (row 16, cols 28-36 = 3)
  stops it. Walls DO block after all — earlier "passing through 3" was floor, not blocking walls.
- Also: blocked actions do NOT erode the bar. Erosion = 1 col-pair per SUCCESSFUL move (step 1-9).
- Block position vs top box: block cols 34-38, box interior cols 29-35 → overlap only cols 34-35.
  Box bottom wall spans full width cols 28-36. Blocked regardless.
- REVISION: the objective is probably NOT "push block into box". Candidates now:
  - H5: erode the 11 bar fully (33 col-pairs left) → level completes at bar empty.
  - H6: collect the 9-dots somewhere (block's 9s vs the scattered 9-groups).
  - The 9-dot groups (top box rows 11-13, bottom-left rows 55-60) may be the real collectibles,
    and the block carries 9s; maybe when the block's 9s reach/merge with a 9-group something happens.
- Budget note: skip helper/model; keep probing. Best next test: RIGHT? LEFT only eroded.
  Actually LEFT/RIGHT erode but don't move the block — they are cheap erosion steps.
  If H5 (bar depletion = win), just keep pressing any action. Bar left: cols 22-54 = 33 pairs.


## Step 11 (RIGHT) confirmed
- Bar eroded col 22 (61,22),(62,22); no block motion. Bar now cols 23-54 (64 cells, 32 col-pairs).
- So: blocked actions do NOT erode; any successful (even 2-cell) action erodes 1 col-pair.
- LEFT/RIGHT = 2-cell erosion only. UP/DOWN = 52-cell moves + erosion.
- Remaining erosion count if H5 (empty bar = win): 32 successful actions needed (min, using LEFT/RIGHT).
- Block is parked against top box. DOWN would un-park it.
- Strategy: if H5 is right, just spam LEFT (2 cells each, no block displacement to track) 32 times.
  But each step is one submission turn; batching is not earned. That's slow but safe.
  Budget check: model_admission_seconds remaining — must not build a model.


## Step 12 (LEFT) confirmed
- Bar eroded col 23 → now cols 24-54 (62 cells, 31 col-pairs). 2 cells changed, no block motion.
- H5 (bar depletion = win) still open; 31 successful actions remain to empty the bar.
- Strategy: keep alternating LEFT/RIGHT (or either) one per turn; each erodes one col-pair.
- Budget: model_admission_seconds remaining — do NOT build a helper/model; keep probing.
- If bar empties without level-up, H5 is wrong and I need a new objective theory.


## Step 13 (RIGHT) confirmed
- Bar eroded col 24 → now cols 25-54 (60 cells, 30 col-pairs). 2 cells, no block motion.
- H5 (bar depletion = win): 30 successful actions remaining.
- Block parked at 12@15-16 / 9@17-19 against top box. Bar erosion continues via LEFT/RIGHT.


## Step 14 (LEFT) confirmed
- Bar eroded col 25 → now cols 26-54 (58 cells, 29 col-pairs). 2 cells, no block motion.
- H5 (bar depletion = win): 29 successful actions remaining.
- Block parked at 12@15-16 / 9@17-19 against top box.


## Step 15 (RIGHT) confirmed
- Bar eroded col 26 → now cols 27-54 (56 cells, 28 col-pairs). 2 cells, no block motion.
- last_retrodiction: 1/1 matched (batching gate still requires frozen model check).
- H5 (bar depletion = win): 28 successful actions remaining.
- Block parked at 12@15-16 / 9@17-19 against top box.


## Step 16 (LEFT) confirmed
- Bar eroded col 27 → now cols 28-54 (54 cells, 27 col-pairs). 2 cells, no block motion.
- H5 (bar depletion = win): 27 successful actions remaining.
- Block parked at 12@15-16 / 9@17-19 against top box.


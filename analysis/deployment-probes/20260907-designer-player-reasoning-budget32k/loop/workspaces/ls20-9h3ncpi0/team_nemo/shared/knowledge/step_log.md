
# Turn 0 — initial frame (step 0, level 0, NOT_FINISHED)

## Layout facts (measured, 64x64; row0=top)
- Colors present: 0,1,3,4,5,8,9,11,12 (counts 3,2,892,2609,439,12,45,84,10)
- 4 = background/void (2609). 3 = floor (892). 5 = wall/block (439).
- Left-edge wall: rows 0-49, cols 0-3 are 5 (rows 0-16 full, rows 17-49 also 0-3).
- Big floor region: rows 17-49, roughly cols 14-53, with a LEFT staircase
  (rows 17-24 floor only at cols 32-36; row 25+ opens to cols 14-53) and
  some interior 4-pockets (e.g. row 30 cols 31-34).
- Upper box (rows 8-16, cols 32-40): border 3, interior 5, with 6 '9' cells
  at (11,35-37),(12,37),(13,35),(13,37).
- Object on floor: 12-block rows 45-46 cols 34-38 (n=10) directly above a
  9-block rows 47-49 cols 34-38 (n=15).
- Bottom HUD band (rows 52-63): 5-frame box lower-left (rows 53-62, cols 1-11)
  containing 9s at rows 55-60 cols 3-8; long 11-band rows 61-62 cols 13-54;
  8s at rows 61-62 cols 56-63.

## Candidate player sprite (5 cells, inside floor)
- 0 cells: (31,21),(32,21),(32,22)   1 cells: (32,20),(33,21)
- 3x3 at rows31-33/cols20-22: `.0. / 100 / .1.`
- Hypothesis H1: this 5-cell cluster is the player (directional arrow).

## Hypotheses
- H1: 0/1 cluster = player. Status: untested. Test: press a direction, see which cells move.
- H2: 3=floor (walkable), 5=wall, 4=void; arrow keys move player on floor. untested.
- H3: goal involves the 9/12 objects (collect or reach). untested.
- H4: upper box (rows8-16) + bottom band are HUD/minimap, not walkable. untested.


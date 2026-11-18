# Hypotheses
- H1: player is the 0/1 cluster at rows 31-33, cols 20-22; 4-direction move. | status: untested (UP changed nothing there)
- H2: something else moves with UP: the 9 (white) / 12 (gray) block pair near (45,34-38) shifted upward by ~3-5 rows. | status: under test
- H3: colors meaning: 5=green border? (5 appears as big border block AND inside the 4 region); 3=green region; 4=blue background; 9/12=an object that slides up; 8=orange bar; 11=gray bar. | status: unknown
- H4: goal = move the 9-block into the green 3 region (rows 8-49)? | status: untested
- H5: the 0/1 cluster is a "you" marker drawn over the moving object's previous position? | untested

## Updated after DOWN (step 2):
- H2 CONFIRMED (partially): DOWN moved the 9-block from rows 42-44 back to 47-49 (down 3 rows). So vertical movement is 3 rows/step.
- NEW: 12-block (2x5) moves 5 rows/step vertically (45-46→40-41 on UP, 40-41→45-46 on DOWN). 9-block (3x5) moves 3 rows/step. They overlap at step 0 but separate after movement.
- H6: 12 and 9 may be separate objects with different velocities, OR 12 is a "shadow"/"ghost" trailing ahead. | status: under test
- H7: Need LEFT/RIGHT probes to determine horizontal movement pattern.


## After LEFT (step 3):
- 9-block: rows 47-49, cols 34-38 -> cols 29-33 (moved 5 cols left)
- 12-block: rows 45-46, cols 34-38 -> cols 29-33 (moved 5 cols left)
- Both moved 5 cols. Vertical positions unchanged.
- H8: horizontal movement = 5 cols/step for both objects. Vertical: 12 moves 5 rows/step, 9 moves 3 rows/step. | status: needs RIGHT confirmation
- H9: Maybe the "speed" depends on the object's dimension in that axis: 12 is 2 rows tall -> moves 5 rows?? No, 2≠5.
- H10: The objects might have fixed velocities: 12=(5,5), 9=(3,5) in (row,col) units per step. | untested


## H11: The "move" outcome does NOT guarantee the object actually moved — it can be blocked by walls. | CONFIRMED (step 7-9: 3x LEFT, object stayed at (35,34))
## H12: The blue (4) regions are walls that block movement. | CONFIRMED
## H13: The wall at cols 29-33 has gaps at rows 25-29 and 45-49. | CONFIRMED from grid inspection


## H14: The 0/1 target is a collectible item. | CONFIRMED (disappeared when object moved over it)
## H15: The 11-bar is a countdown timer (~1 col/step). | LIKELY (consistent shrinkage)
## H16: Win condition = collect target + reach a specific location. | UNTESTED
## H17: Win condition = collect target + the target removes a wall. | UNTESTED
## H18: The 8-marks (3 pairs) represent item slots / score. | UNTESTED


## H14: The 0/1 target is a collectible item. | CONTRADICTED (reappears when object moves away)
## H14b: The 0/1 target is a STATIC goal marker. | CONFIRMED
## H16: Win condition = collect target + reach a specific location. | REVISED: touch target + reach exit
## H19: Win condition = object center aligns with target center. | CONTRADICTED (was at (30,19), center (32,21) = target center, no trigger)


## H20: The bottom-left 9-cell pattern is a "base" or "home" that the object needs to reach. | UNLIKELY (outside playfield)
## H21: The win condition is to move the object to a specific position that I haven't identified. | UNTESTED
## H22: The win condition is to collect all 3 "8-marks" in the status bar. | UNTESTED (how?)


## H23: The 9-cell pattern at the top (rows 11-13, cols 35-37) is a "key" or "gate" that triggers level completion. | UNTESTED
## H24: The win condition involves both the 0/1 target AND the 9-cell pattern. | UNTESTED


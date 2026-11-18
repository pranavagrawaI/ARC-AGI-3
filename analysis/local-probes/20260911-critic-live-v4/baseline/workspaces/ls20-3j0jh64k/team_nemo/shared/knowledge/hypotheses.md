# Hypotheses — level 0, step 0 (turn 1)

## Measured facts (step 0)
- 64x64 grid. Background 0x4. Left wall 0x5, cols 0-3 (some rows).
- Band region rows 40-49, cols 34-38: 12s (0xC, top 2 rows), 9s (0x9, bottom 3 rows); 0x3 field fills rest of 10x10 (rows 40-49, cols 34-43). Band is 5 wide, 5 tall.
- Rows 40-43 cols 34-43 all 0x3; rows 45-48 cols 34-43: 12/9 in cols 34-38, 0x3 in cols 39-43. Rows 44? measured: row44 = "455555555554444444444444444444444444444444444444444444444444444444" — all 0x4 except left wall. (need to recheck row 44 in band)
- Bar: rows 61-62, cols 20-54 are 0xB (0xBBB...), i.e. 35 cols wide, 2 rows tall.
- 0x8 cells: rows 61-62, cols 56-57, 59-60, 62-63 (8 cells).
- 0x5 wall left edge; some rows rows 61-63 start "4555...54" pattern? row61: "45555555555453333333bbbb..."

## Live hypotheses (objective)
- H1: the 0x9 (yellow) band segment needs to reach the 0x8 (gray) cells in the bottom bar — i.e. push the band right/down until it contacts the 8-cells.
- H2: the 0xB bar is an obstacle or a goal zone; 12s must touch/merge with it.
- H3: 12/9 band is a single "player" block; moving moves the whole band; the goal is to fill the 0x3 region or reach a target.
Open conditions: need to test what each action moves (band vs player vs camera).

## Action plan
1. RIGHT — does the 12/9 band move? (test H3)
2. If not, LEFT/UP/DOWN to find the movable object.


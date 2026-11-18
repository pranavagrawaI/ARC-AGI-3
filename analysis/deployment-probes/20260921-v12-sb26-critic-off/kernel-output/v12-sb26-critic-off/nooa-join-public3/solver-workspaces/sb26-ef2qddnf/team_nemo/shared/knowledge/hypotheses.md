
# Hypotheses (level 0, step 0)

## Scene description (measured, step 0)
- Background color 4 everywhere except the objects below.
- **Top box** rows 0-7, cols 17-45: border color 5, interior 4.
  - Slot A cols 18-24: 3x3 ring of color 9 (red), center 4.
  - Slot B cols 26-32: 3x3 ring of color 14 (purple), center 4.
  - Slot C cols 34-40: 3x3 ring of color 11 (cyan), center 4.
  - Slot D cols 42-48(41-47): 3x3 ring of color 15 (pink), center 4.
- **Middle container** rows 24-35, cols 17-46: border color 8 (orange),
  corner cells color 0 (black). Inside:
  - 3x3 squares of color 2 (blue) centered at (29-30? rows 29-30, cols 22-23) etc —
    actually measured: 4x2 blocks? Let me recheck: comps 3-6 rows 29-30, cols 22-23,
    28-29, 34-35, 40-41. These are 2x2 squares of color 2? Wait comp 5 also has color 0.
    Actually the printed crop shows rows 29-30 have "22444422..." at cols 22-23,28-29,34-35,40-41
    — 4 squares of 2x2 color-2 (blue). One (cols 34-35) touches a corner-0? No, it was
    near cols 34-35 which is central. Recheck: comp 5 has color 0 — maybe a corner-0 cell
    got included? Actually no — the bbox rows 29-30 cols 34-35 with colors 0,2,4,8 means
    some of those cells are 0. Let me verify precisely.
- **Legend** rows 57-60: four 4x4 solid squares at cols 18-21 (14/purple),
  26-29 (9/red), 34-37 (11/cyan), 42-45 (15/pink).
- **Full-width band** row 53: color 2 (blue) across cols 0-63.

## Live hypotheses
H1: This is a "pick the matching color" puzzle. The 3x3 rings at top have colors 9,14,11,15
    (slots A,B,C,D). The legend squares at bottom are 14,9,11,15. The 2x2 blue squares in the
    middle are targets? The band at row 53 is blue (2).
H2: Goal: click something to complete level 0. Actions available: RESET, SPACEBAR, CLICK, UNDO.
    Need to discover which object is interactive.

## Next probe
Click center of a top-slot ring (e.g. red ring center at row 3-4, col 21) or a legend square
to see what changes.


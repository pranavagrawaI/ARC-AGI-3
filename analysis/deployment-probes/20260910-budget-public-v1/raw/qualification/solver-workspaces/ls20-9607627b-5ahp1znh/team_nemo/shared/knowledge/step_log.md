## Turn 0 - initial observation
Grid 64x64. Palette: 4(bg/sky), 5(walls), 3(blocks), 9(entities), 11(floor strip row61-62), 8(small strips), 12(2 rows strip), 0/1 (small dots near (32,21)).
Layout:
- Big 3-block room rows 8-49, cols 14-53. Contains:
  - rows 8-16 cols 32-40: closed 5-box with 9's inside (11,35-37),(12,37),(13,35),(13,37) - box1
  - rows 17-21 cols 34-38: open ##### box (no floor)
  - rows 22-44: various 5-walled rooms with 3 floor inside; there are 9-entities scattered (rows 47-49 cols 34-38 are a 3x5 block of 9s under a cccc strip at rows 45-46)
- rows 45-46 cols 34-38: color 12 strip
- rows 53-62 col 1-10: 5-walled box with 9 entities inside (rows 55-60)
- rows 61-62 cols 13-54: color 11 floor strip; row 61 has 8-entities at cols 56,57,59,60,62,63 (dashed)
- row 63: all 5 (bottom wall)
- top rows 0-7: 5-walls cols 0-3 then 4s
- cols 0-3 all rows: 5-wall
Player location unknown - no unique player marker visible. The 9 entities are candidates for the player or for 'items'. No player icon distinct from 9s.
Actions available: RESET, UP, DOWN, LEFT, RIGHT (no USE/CLICK).


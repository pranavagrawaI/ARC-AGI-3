## Turn 0 (fresh start, step 0, level 0)
Initial observation. Actions: RESET/UP/DOWN/LEFT/RIGHT only.
Key measured objects (step 0 grid, parsed):
- 11-bar: rows 61-62, cols 13-54 (42 cols). 8-blocks at rows 61-62, cols 56-57/59-60/62-63.
- Scrolling band window: rows 40-49, cols 34-43 (10x10). Band = 5x5 block, 12 on top 2 rows, 9 on bottom 3 rows, currently at rows 45-49, cols 34-38.
- Static 9-dot patterns: top box (rows 8-16, cols 32-40, ~6 dots at rows 11-13 cols 35-37 + extras at 12-13), bottom-left (rows 55-60, cols 3-8).
- 0/1 tiny dots at (31-33, 20-22) inside big 3-region.

## Turn 1 — LEFT (step 1)
Observation sought: does the 5x5 band scroll LEFT (untested direction), or is LEFT bar-only?
Predicted: band 12-block cols 34-38 -> 29-33 (rows 45-46), 9-block cols 34-38 -> 29-33 (rows 47-49); old cols 34-38 -> 3; bar 11->3 at col 13.
Alternate (bar-only): only 2 cells (61,13),(62,13) 11->3.

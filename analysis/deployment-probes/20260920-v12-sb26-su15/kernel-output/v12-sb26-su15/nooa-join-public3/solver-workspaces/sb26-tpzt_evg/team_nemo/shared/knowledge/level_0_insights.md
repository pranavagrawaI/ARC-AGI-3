# Level 0 Insights

## Successful Conditions
- Starting: 4x4 target box (rows 25-34, cols 18-45) empty; 4 reference squares at top (9, E, B, F); 4 source squares at bottom (E, F, 9, B shuffled).
- 12 actions total: 3 SPACEBAR (initial), then 6 CLICK select/place pairs.

## Sequence That Worked
1. SPACEBAR x3 (steps 1-3) — unclear purpose, possibly init or count
2. CLICK bottom E (19,58) -> select; CLICK middle pos2 (31,29) -> place E
3. CLICK bottom F (27,58) -> select; CLICK middle pos4 (40,29) -> place F
4. CLICK bottom 9 (35,58) -> select; CLICK middle pos1 (22,29) -> place 9
5. CLICK bottom B (43,58) -> select; CLICK middle pos3 (34,29) -> place B
6. SPACEBAR (step 12) -> level completed

## Supported Rules
- Select a 4x4 square from bottom row by clicking its center
- Place selected square into a slot in the middle box by clicking target position
- Placement: 16 cells of color fill + 20-cell border removal + source cleared + 4-cell remnant left at source
- Each placement = 53 changed cells
- SPACEBAR after all pieces placed triggers level completion
- 4 green 2x2 remnant squares appear at bottom where pieces were selected from

## Tested Scope
- Only level 0. Whether same mechanics apply to level 1 is untested.

## Open Questions
- What were the 3 initial SPACEBARs for? (Steps 1-3 showed 1 cell change each)
- Why are pieces placed in specific positions? (pos1=9, pos2=E, pos3=B, pos4=F — matching top reference order?)
- Are the top reference squares just decorative or do they define required positions?

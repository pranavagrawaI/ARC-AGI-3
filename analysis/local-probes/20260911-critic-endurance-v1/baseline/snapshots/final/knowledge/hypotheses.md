# Hypotheses

## Grid colors (measured)
- 5 = outer border/shell (static)
- 4 = open/void corridors (static)
- 3 = maze floor/walls (static)
- 0,1 = character sprite at (31-33, 20-22) (static, decorative?)
- 9 = collectible dots
- 12 (c) = top part of the "player block"
- 11 (b) = countdown timer bar (decreases 1 col/step from left)
- 8 = static block bottom-right

## Player object: c/9-block
- 5x5 structure: c-block (2 rows) on top, 9-block (3 rows) below
- Moves 5 cells per action in the action direction
- Current position (step 3): rows 45-49, cols 29-33
- Initial position (step 0): rows 45-49, cols 34-38

## Countdown timer: b-bar
- Rows 61-62, cols 13-54 initially (42 cells wide)
- Decreases by 1 column per step (from the left)
- At step 3: 39 columns left
- When it runs out: GAME OVER?

## Static 9-cells (collectibles?)
- Top-box: (11,35-37), (12,37), (13,35,37) - 6 cells
- Bottom-left: (55-60, 3-8) - 18 cells
- These are OUTSIDE the maze area, in 5-walled regions

## Mechanics (tested steps 1-3)
- UP: c/9-block moves up 5, b-bar -1
- DOWN: c/9-block moves down 5, b-bar -1
- LEFT: c/9-block moves left 5, b-bar -1
- RIGHT: untested
- Character sprite: does NOT move on screen
- Maze walls: do NOT change

## Objective (hypothesis)
- Move c/9-block to collect/overlap with static 9-cells before b-bar runs out
- OR: move c/9-block to a specific target position

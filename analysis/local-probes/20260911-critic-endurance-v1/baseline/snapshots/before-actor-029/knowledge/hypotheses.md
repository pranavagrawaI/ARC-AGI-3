# Hypotheses

## Grid colors (measured, step 0)
- 5 = outer border/shell
- 4 = background/void (large areas)
- 3 = maze structure (floor or walls?)
- 0,1 = character sprite (3x3 at rows 31-33, cols 20-22)
- 9 = collectible dots (3 clusters: top-box ~(11,35-37), middle ~(47-49,34-38), bottom-left ~(55-60,3-8))
- 12 (c) = block at (45-46, 34-38) - 5x2
- 8 = block at (61-62, 56-63) - 8x2
- 11 (b) = bar at (61-62, 13-54) - 42x2

## Mechanics (untested)
- Character moves with UP/DOWN/LEFT/RIGHT
- Need to determine: does character walk on 3-cells (floor) or 4-cells (corridors)?
- Need to determine: goal (collect all 9s? reach a specific location?)

## Objective (untested)
- Possibly collect all 9-cells
- Or reach the c-block / 8-block

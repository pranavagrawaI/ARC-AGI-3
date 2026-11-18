# Hypotheses

## H1: The white 5x10 block at rows 8-12, cols 3-12 is the player
- Status: Untested
- Evidence: It's a distinct white shape on the playfield, common in grid games
- Test: Try UP/DOWN/LEFT/RIGHT and see if this block moves

## H2: The 10x10 black rectangle at (34-43, 27-36) is an obstacle or goal
- Status: Untested
- Evidence: Large dark region on the playfield
- Test: Try to walk into it or see if the game responds

## H3: The 7x12 white block at (25-31, 26-37) is another entity or collectible
- Status: Untested
- Evidence: Distinct white shape
- Test: Move near it and observe

## H4: Color 4 (blue) are walls
- Status: Untested
- Evidence: They form borders and enclosures
- Test: Try to walk through them

## H5: Color 5 (gray) is the walkable background
- Status: Untested
- Evidence: Fills most of the playfield

## H1: The white 5x10 block at rows 8-12, cols 3-12 is the player
- Status: CONTRADICTED
- Evidence: UP and RIGHT did not move this block; it's stationary
- The actual player is the medium white diamond component

## H6: The player is a rolling ball that grows as it moves
- Status: Confirmed (1 data point)
- Evidence: RIGHT moved the diamond from (25-31, 26-37) to (23-35, 35-47)
  - Grew from 7x12 to 13x13, n stayed ~85
- Test: Move more and see if it keeps growing

## H7: Black (0) regions are obstacles/walls
- Status: Partially confirmed
- Evidence: UP was blocked by black region at rows 3-7
- Test: Try to walk through other black regions

## H8: Green (2) is a trail/shadow that follows the ball
- Status: Confirmed
- Evidence: Green border appears around the diamond


## H9: SPACEBAR fills the black region with white
- Status: Confirmed (1 data point)
- Evidence: SPACEBAR turned top 5 rows of black (10x10) into white
- Test: Press SPACEBAR again - expect bottom 5 rows to turn white

## H10: The goal is to fill the entire black region with white
- Status: Untested
- Evidence: SPACEBAR is filling the black region
- Test: Keep pressing SPACEBAR until all black is gone, then see if level completes


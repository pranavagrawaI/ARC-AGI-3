# World Model Scratchpad

## Level 0 observations

- Grid 64x64, colors 0-15.
- Movable block composed of color 12 (orange) top rows and color 9 (blue) bottom rows, size 5x5.
- Initial position: rows 45-49, cols 34-38. Background in that region is color 3 (dark-grey).
- Action UP moves block up by its own height (5 rows). Old cells cleared to background 3 (3). New cells placed with same pattern.
- Step counter / yellow bar: color 11 bar on rows 61-62, cols 13-54. Each action shrinks bar from left by one column (cells become 3). This suggests a move budget / life counter.
- No level_complete / game_over observed yet.

## Model decisions

- State holds grid directly.
- transition():
  - Find movable component by flood fill from orange cells (12) through 9/12 adjacency.
  - Compute bounding box, extract pattern, move by height/width in direction of action.
  - Clear old bbox to 3, write pattern at new location.
  - Decrement yellow bar leftmost column on rows 61-62 for any arrow action.
- render() returns grid.
- outcome() remains ongoing pending terminal evidence.

## Open hypotheses

- Does DOWN/LEFT/RIGHT move block by height/width? Likely yes, same rule.
- Is yellow bar decrement always 1 column per action, or does it depend on direction? Observed only UP.
- Objective unknown: likely move block to target location before bar depletes. Could be Sokoban-like.
- Game over condition unknown: maybe bar depletion causes GAME_OVER.

## Next probes

- Test DOWN/LEFT/RIGHT to confirm movement rule.
- Observe when bar reaches zero.
- Look for target cells (maybe color 8?).

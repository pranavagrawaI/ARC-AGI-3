# Hypotheses

## H1: Game is a sliding/pushing maze with an agent entity
- Grid split vertically: b (11) is a "block/wall" occupying left portion in many rows; 5 is background floor; c (12) is right-side region; a (10) appears as two 5x5 squares near bottom (rows 49-53, cols 19-23 and 39-43).
- Candidate: agent is one of the two 'a' blocks or is a single pixel; goal may be to move both a-blocks or stack them.
- Untested. Deciding test: submit UP/RIGHT and see which pixels move; if an 'a' block translates, entity is the a block.

## H2: b is a wall and a is a movable block (sokoban-like)
- c region to the right may be a goal zone or just background. Untested.

## H3: a-blocks are two "keys/boxes" needing to reach two targets
- Untested.


## H1 UPDATE: Two 'a' blocks (5x5 each) are the entities
- CONFIRMED: UP moved both a-blocks from rows 49-53 to rows 44-48 (delta -5 rows).
- Both moved identically, suggesting they're either one unit or two synced agents.
- The '5' color is the walkable corridor; 'b' is wall; 'c' is right-side region.
- Movement magnitude: exactly 5 rows for one UP = block height. Possibly "jump to next gap" or fixed step.

## H4: The a-blocks "slide" until hitting an obstacle or edge of corridor
- If so, UP should move them until they hit the top of the '5' corridor.
- At rows 44-48, the corridor continues up to row 9 (where it narrows). 
- If they slid, they should have gone further up. They only went 5 rows, so maybe NOT sliding.
- Deciding test: press UP again. If they move another 5 rows (to 39-43), it's fixed step.
  If they move to the top of the corridor, it's sliding.


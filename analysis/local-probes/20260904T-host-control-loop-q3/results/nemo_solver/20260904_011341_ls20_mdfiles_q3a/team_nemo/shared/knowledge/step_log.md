
        ## Turn 9 (step 15) - RESET
        - RESET returned entity to (45, 34). Confirmed.
        - Entity has been to positions: 45, 40, 35, 30, 25, 20, 15. None triggered level completion.
        - UP from (15, 34) is blocked by static 9-patch at (11-13, 35-37).
        - LEFT/RIGHT are no_impact (entity confined to corridor cols 34-38).
        - Stuck at level 0. Need to reconsider the game mechanics.


        ## Turn 10 (step 21) - Entity at (15, 34)
        - Confirmed: minimap is a STEP COUNTER, not a minimap. 2 cells per step.
        - Entity at (15, 34), UP blocked by static 9-patch, LEFT/RIGHT no_impact.
        - Need to find the goal mechanic. Trying RESET from (15, 34) to see if it changes the static patch.


        ## Turn 16 (step 27) - BREAKTHROUGH!
        - LEFT from (25,34) WORKED! Entity moved to (25,29).
        - Key insight: entity can move LEFT/RIGHT through 3-walls, but NOT through background (4s).
        - At (25,34), cols 25-33, rows 25-29 are all 3s, so LEFT works.
        - At (20,34), cols 25-33, rows 20-24 are all 4s, so LEFT is blocked.
        - The maze is a 3D structure of 3-walls. The entity can navigate through them.
        - The 0/1 patch room (rows 30-34, cols 20-24) is now reachable!
        - Next: LEFT to (25,24), then DOWN to (30,24), then LEFT to (30,19) to enter the room.


        ## Turn 19 (step 30) - 0/1 patch absorbed!
        - Entity at (30, 19) in the 0/1 room. 0/1 patch absorbed (all 0s and 1s gone).
        - diff=58 (50 entity + 6 patch cells + 2 HUD), 1 animation frame.
        - Level still NOT_FINISHED.
        - Route to (15,34): RIGHT, UP, RIGHT, RIGHT, UP, UP = 6 moves.
        - Trying to reach (15,34) to see if the level completes after absorbing the patch.


        ## Turn 20 (step 36) - Entity at (15, 34)
        - Entity at (15, 34) after 6 moves from (30, 19).
        - Level still NOT_FINISHED.
        - 0/1 patch absorbed. 3x3 9-patch still in Room 1.
        - Trying DOWN then UP to test if a "bounce" triggers something.


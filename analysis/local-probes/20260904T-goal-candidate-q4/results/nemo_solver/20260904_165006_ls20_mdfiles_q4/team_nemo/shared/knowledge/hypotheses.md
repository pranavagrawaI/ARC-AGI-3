# Hypotheses

        ## H1: 3=walkable floor, 4=wall/outside, 5=walkable (different floor)
        - Status: untested
        - Evidence: player (9-spiral) sits on 5 in top room; 3 is the main corridor
        - Test: move DOWN, see if player moves

        ## H2: 9-spiral at top room = player, 9 at bottom-right = goal
        - Status: untested
        - Evidence: two 9s inside playfield, one at start position, one at end of corridor
        - Test: navigate to bottom-right, see if level completes

        ## H3: 0/1 at (31-33, 20-22) = obstacle or collectible
        - Status: untested
        - Evidence: small 3x3 sprite in the middle of the wide band

        ## H4: 12 (green) at (45-46, 34-38) = goal marker above the 9
        - Status: untested

        ## H5: 9 at (55-60, 3-8) outside playfield = HUD/decoration
        - Status: untested

        ## H6: 11 bar at (61-62, 13-54) and 8 blocks at (61-62, 56-63) = HUD
        - Status: untested
        
        ## H7: 9-spirals are STATIC decorations/goals, not a movable player
        - Status: CONFIRMED (turn 1)
        - Evidence: DOWN changed only 2 HUD cells (61,13) and (62,13) b->3; playfield unchanged

        ## H8: HUD bar (rows 61-62, cols 13-54) is a progress/step counter
        - Status: untested
        - Evidence: DOWN filled col 13 with color 3
        - Test: submit RIGHT, expect col 14 to fill

        ## H9: The game is about filling the HUD bar, not navigating a maze
        - Status: untested
        - Evidence: playfield is static; only HUD changes


        ## H10: Player moves 5 grid-cells per action
        - Status: confirmed (turn 2)
        - Evidence: RIGHT moved player from cols 34-38 to 39-43 (5-cell shift)

        ## H11: Path to goal: LEFT then UP x7
        - Status: untested
        - Evidence: fit map shows all positions valid; goal at (11-13, 35-37)
        - Plan: LEFT to (45,34), then UP x7 to (10,34)


        ## H12: HUD bar is a step budget that resets when full
        - Status: CONFIRMED (turn 13)
        - Evidence: HUD filled to 42/42 at step 42, no level completion.
          After step 44 (LEFT), HUD reset to 3/42. The HUD tracks steps mod 42.

        ## H13: The 0/1 entity is a collectible that disappears when player gets close
        - Status: CONTRADICTED (turn 13)
        - Evidence: Player moved adjacent to 0/1 entity (diff=138, animation),
          but 0/1 entity is still present in settled grid.

        ## H14: The goal is to reach the bottom-left 9-spiral (rows 55-60, cols 3-8)
        - Status: untested
        - Evidence: It's the only 9-spiral in a room that the player CAN reach
          (the wide band connects to the bottom-left area via a pipe at cols 19-23,
          rows 40-49). The top room 9-spiral is unreachable (room too small).

        ## H15: The player needs to reach the bottom-left room through the left pipe
        - Status: untested
        - Evidence: Left pipe at cols 19-23, rows 40-49 connects the wide band to
          the bottom area. The bottom-left room is at rows 53-63, cols 1-9.


        ## H16: The goal is to reach the top room (position 15,34)
        - Status: untested
        - Evidence: The top room is the only "destination" in the maze. The 9-spiral in the
          top room is the only 9-spiral that the player can approach (though not reach exactly).
          The player can reach (15,34) which is adjacent to the 9-spiral.
        - Test: Navigate to (15,34) via RIGHT, RIGHT, UP x6. If the level completes, confirmed.

        ## H17: The 0/1 entity is a wall that blocks the left pipe
        - Status: CONFIRMED
        - Evidence: The 0/1 entity at (31-33, 20-22) blocks the left pipe (cols 19-23).
          The player cannot go UP through the left pipe.
          The player must use the right pipe (cols 34-38) to reach the top room.


        ## H18: When HUD is full (42/42), the next action teleports player to start position (45,34) and resets HUD
        - Status: CONFIRMED
        - Evidence: Step 43: player at (25,29), HUD=42/42. Step 44 (LEFT): player at (45,34), HUD=0/42.
          The player teleported 20 rows down and 5 cols right. The 0/1 entity was unchanged.
          The HUD reset from 42/42 to 0/42.
        - Implication: The game has 42-step "phases." Each phase starts at (45,34).
          The goal must be achievable within 42 steps from (45,34).

        ## H19: The goal is to reach a specific position within 42 steps of the phase start
        - Status: untested
        - Evidence: The player starts at (45,34) each phase. The maze has specific destinations:
          top room (15,34), 0/1 entity area (31-33, 20-22), wide band positions.
        - Test: In the next phase, navigate to the top room (15,34) within 42 steps and
          see if the level completes.

        ## H20: The goal is to be at a specific position when the HUD resets
        - Status: untested
        - Evidence: The teleport happens when the HUD is full. Maybe the player needs to
          be at a specific position (like the 0/1 entity) when the HUD resets for the
          level to complete.
        - Test: Navigate to be at/near the 0/1 entity when the HUD reaches 42/42.


        ## H21: RESET might fix a corrupted game state
        - Status: untested
        - Evidence: 55 steps of exploration with no level completion. The game may be
          in a bad state due to my exploration.
        - Test: Submit RESET and see if the game returns to the initial state.


        ## H22: The 0/1 entity is a COLLECTIBLE that the player pushes into
        - Status: CONFIRMED
        - Evidence: Player at (35,19) pushed UP into the 0/1 entity at (31-33, 20-22).
          The entity DISAPPEARED, the player moved to (30,19), and an animation frame
          was produced (event_frame_count=1, diff=58 instead of normal 52).
          The left pipe (cols 19-23) is now CLEAR.

        ## H23: The 0/1 entity is a KEY that unlocks the left pipe
        - Status: CONFIRMED
        - Evidence: Before collection, the left pipe was blocked (player couldn't go UP
          from (35,19) to (30,19)). After collection, the player is at (30,19) and
          can continue UP to (25,19).

        ## H24: Collecting the 0/1 entity is the GOAL of Level 0
        - Status: untested
        - Evidence: The 0/1 entity was the only interactive object. Collecting it
          cleared the left pipe. The level might complete on collection, or there
          might be a second objective after collection.


        ## H25: Collecting the 0/1 entity rotated/advanced the bottom-left 9-spiral
        - Status: CONFIRMED
        - Evidence: Before collection, 9s at (57,3),(57,4),(58,3),(58,4). After collection,
          those became 5s and new 9s appeared at (57,7),(57,8),(58,7),(58,8).
          The spiral pattern shifted clockwise.
        - Implication: The 9-spiral is a dynamic object that changes when the 0/1 entity
          is collected. The level goal might be related to the spiral's state.

        ## H26: The level goal is to complete the 9-spiral rotation
        - Status: untested
        - Evidence: The 9-spiral changed after collecting the 0/1 entity. If the spiral
          needs to complete a full rotation, the level might complete when the spiral
          returns to its original shape.
        - Test: Navigate to the top room and see if the 9-spiral there also changes.
          If so, the goal might be to "activate" both spirals.


        ## H27: The top room 9-spiral did NOT change after collecting the 0/1 entity
        - Status: CONFIRMED
        - Evidence: The top room 9-spiral pattern is identical to before the collection.
          Only the bottom-left 9-spiral changed.
        - Implication: The 0/1 entity is specific to the bottom-left room/spiral.

        ## H28: The level goal involves the 9-spiral completing a rotation
        - Status: untested
        - Evidence: The bottom-left 9-spiral rotated/shifted after collecting the 0/1 entity.
          If the spiral needs to complete a full rotation (return to original state), the
          level might end when that happens.
        - Test: Find and collect more items to advance the spiral further.


        ## H29: The 0/1 entity is a STATIC, PASSABLE object (not collectible)
        - Status: CONFIRMED
        - Evidence: The 0/1 entity was present at step 0 (game start). When the player
          moved to (30,19), the entity was temporarily hidden by the player's body.
          After the player moved away, the entity reappeared at the same position.
          The entity blocks pathfinding (values 0,1 are not standard floor) but is
          passable by the player.

        ## H30: The bottom-left 9-spiral change was triggered by player position, not collection
        - Status: CONFIRMED
        - Evidence: The 9-spiral changed at step 62 when the player moved from (35,19)
          to (30,19), overlapping with the 0/1 entity. The change persisted after the
          player moved away. The 0/1 entity was NOT destroyed.
        - Implication: The 9-spiral is "activated" when the player reaches a specific
          position or zone. The activation is permanent.

        ## H31: The 9-spiral is a progress indicator that must complete a full rotation
        - Status: untested
        - Evidence: The 9-spiral pattern shifted (9s moved from left to right side)
          after the player triggered it. If the spiral needs to complete a full
          rotation, the level might end when the spiral returns to its original state.
        - Test: Navigate the player to other trigger zones to advance the spiral further.


        ## H32: The HUD counts DOWN from 42 to 0 (step budget, not progress bar)
        - Status: CONFIRMED
        - Evidence: Step 0 shows all 11s (full). Each move decreases the count.
          When HUD reaches 0, player teleports to (45,34) and HUD resets to 42.

        ## H33: The goal is to be at the entity zone (30,19) when HUD reaches 0
        - Status: untested
        - Evidence: The entity zone is the only interactive area. Maybe the level
          completes when the player is at the entity zone with 0 steps remaining.
        - Test: Navigate to (30,19), then oscillate until HUD=0.

        ## H34: The goal involves the 88588588 pattern at bottom-right
        - Status: untested
        - Evidence: The 8-8-5-8-8-5-8-8 pattern at (61-62, 56-63) is static and
          unexplained. It might be a timer, counter, or indicator.


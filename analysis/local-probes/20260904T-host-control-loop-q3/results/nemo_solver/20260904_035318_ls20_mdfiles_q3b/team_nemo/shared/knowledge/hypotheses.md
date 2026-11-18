# Hypotheses

        | Hypothesis | Status | Evidence |
        |---|---|---|
        | 3 = walkable floor | untested | Player is inside 3-block; 4 is background |
        | 4 = wall (inside 3-block) | untested | 4-strips at rows 30-39 cols 29-33 and rows 40-44 cols 19-23 |
        | 5 = frame/border (not walkable) | untested | 5-frame at rows 8-16 cols 32-40 |
        | 9 = goal/target | untested | 5x5 9-patch in HUD (rows 47-49), 9s inside 5-frame (rows 11-13) |
        | 0/1 = player (5-cell diamond) | confirmed | Player cells: (31,21),(32,20),(32,21),(32,22),(33,21) |
        | 12 (c) = HUD element | confirmed | hud_rows includes 45,46 |
        | 11 (b) = HUD element | confirmed | rows 61-62 |
        | 8 = HUD element | confirmed | rows 61-62 |
        | Player moves 1 cell per action | untested | Force-advanced RIGHT: player didn't move (blocked) |
        | Goal = reach 5-frame at top | untested | 9s inside 5-frame |
        | Goal = reach 9-patch in HUD | untested | 5x5 9-patch at rows 47-49 |
        | HUD rows 45-49, 61-62 are status display | confirmed | hud_rows = [45,46,47,48,49,61,62] |
        
        | 5x5 12/9 block moves 5 cells in action direction | confirmed | RIGHT: +5 cols; UP: -5 rows |
        | Player (0/1 diamond) is static | confirmed | No movement in 2 actions |
        | Goal = position 5x5 block over 5-frame | untested | 5-frame at rows 8-16, cols 32-40 |
        | 5-frame center = target for 5x5 block center | untested | 5-frame center (12,36); block center now (42,41) |


        | 5-frame is a closed box - block can't enter | confirmed | Frame walls (5s) block all entry; block entered from bottom but wall closed behind |
        | Goal = put block inside 5-frame | contradicted | Frame is closed; block can't get inside |
        | Goal = cover player with block | untested | Player at (32,21); block 5x5 centered there would be at (30,19) |
        | Goal = position block at specific playfield location | untested | Need to figure out which location |
        | Block can move in 4-background area | confirmed | Block moved from (45,34) to (20,34) through 4-area |
        | Block blocked by 5-frame walls | confirmed | UP at (15,34) blocked by frame walls at row 15 |
        | Block blocked by 4-walls | confirmed | LEFT at (15,34) blocked by 4-walls at cols 29-33 |


        ## Turn 13 (post-RESET) findings
        | Hypothesis | Status | Evidence |
        |---|---|---|
        | 4-walls at rows 40-44, cols 19-23 | CONTRADICTED | Always floor (3s); no such walls exist |
        | Actual 4-walls: block A (rows 30-39, cols 29-33) | confirmed | 10x5 wall block |
        | Actual 4-walls: block B (rows 40-44, cols 14-18) | confirmed | 5x5 wall block |
        | Actual 4-walls: block C (rows 40-44, cols 24-33) | confirmed | 5x10 wall block |
        | Frame is fully closed (5-walls rows 9-15, cols 33-39) | confirmed | Block can't enter from any side |
        | Frame 9s pattern (cross): (11,35-37),(12,37),(13,35),(13,37) | confirmed | Static decoration |
        | HUD 9s pattern (6-wide) at rows 55-60 | confirmed | Different from frame 9s |
        | Covering player with block does NOT complete level | confirmed | Block at (30,19) covered player, no level_up |
        | Block passes THROUGH player (player hidden, reappears in animation) | confirmed | Event frame showed player diamond |
        | Goal = position block at specific playfield location | untested | Need to find the target |
        | Goal = park block at top of playfield near 5-frame | untested | Testing now |
        | Goal = block must reach row 25 (top corridor) | untested | Next test |


        ## Turn 37: CRITICAL WALL MODEL CORRECTION
        | Hypothesis | Status | Evidence |
        |---|---|---|
        | 5-cells (frame) are walls | CONTRADICTED | Block at (15,34) overlaps row 15 5-cells; frame interior has 3-cells at row 8 |
        | 4-cells are the ONLY walls | CONFIRMED | 37 reachable positions with 4-only walls; (10,34) now reachable |
        | Block can pass through 5-frame | CONFIRMED | Block overlapped frame bottom wall at (15,34) |
        | Goal = block 9s overlap 9-cross in frame | untested | Block at (10,34): 9s at rows 12-14 cols 34-38 overlaps cross at (12,37),(13,35),(13,37) |


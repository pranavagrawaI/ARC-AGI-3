# Level 0 Insights

        ## Winning Strategy
        1. Navigate the 5x5 tile from start position to the items (0/1 cross) in the room.
        2. Overlap the items to pick them up (items disappear, diff=58).
        3. Navigate around the 4-wall obstacle (go below it via rows 45+).
        4. Return to the shaft (cols 34-38).
        5. UP into the box - items unlock box entry.
        6. Tile enters box, level completes.

        ## Key Mechanics
        - 5x5 tile (12-cells top 2 rows + 9-cells bottom 3 rows) = PLAYER
        - 6-cell 9-pattern in box = GOAL (disappears on win)
        - Items (0/1 cross) = KEY to unlock box entry
        - 4-walls = impassable obstacles
        - Tile moves 5 cells per action
        - Tile can overlap box border (3s) but can't fully enter box WITHOUT items
        - With items: tile can enter box and win

        ## Grid Structure
        - Box: rows 8-16, cols 33-41 (3-border, 5-interior)
        - Shaft: rows 17-24, cols 34-38 (4-walls at cols 33, 39)
        - Room: rows 25-49, cols 14-53
        - 4-wall in room: rows 30-39, cols 29-33 + rows 40-44, cols 24-33 + rows 40-49, cols 14-18

        ## What Generalizes
        - Tile movement: 5 cells per action, blocked by 4-walls
        - Items unlock areas
        - Goal is in a restricted area requiring items
        - Navigation around obstacles is key
        
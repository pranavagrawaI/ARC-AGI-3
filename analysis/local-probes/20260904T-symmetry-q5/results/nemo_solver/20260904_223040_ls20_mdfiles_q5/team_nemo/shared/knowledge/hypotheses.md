# Hypotheses

        ## H1: Player is the 0/1 cross at (31-33, 20-22)
        - Status: untested
        - Evidence: 4-cell cross shape (0/1) surrounded by 3s in the maze
        - Test: submit UP/DOWN/LEFT/RIGHT, see if the cross moves

        ## H2: 3 = walkable floor, 4 = wall/background
        - Status: untested
        - Evidence: player is on 3s, maze corridors are 3s
        - Test: move player, see if 4s block movement

        ## H3: 9 = collectible items
        - Status: untested
        - Evidence: 3 clusters of 9s in the maze
        - Test: walk into a 9-cluster, see if it disappears/collects

        ## H4: 12 = goal/target block
        - Status: untested
        - Evidence: 5x2 block at (45-46, 34-38)
        - Test: walk into it, see if level completes

        ## H5: 5 = impassable dark walls (boundary)
        - Status: untested
        - Evidence: 5s form the outer boundary of the maze
        - Test: try to walk into 5, expect no_impact

        ## H6: Maze structure — 3-corridors with 4-pockets as walls
        - Status: untested
        - Evidence: BFS with 3=floor, 4=wall shows maze layout
        - Test: trace the maze, find path from player to goal


        ## H1: 0/1 cross is the player
        - Status: CONTRADICTED — DOWN action did not move the cross
        - Evidence: after DOWN, 0/1 cells unchanged; instead bottom band cells changed

        ## H7: Bottom band (11-color, rows 61-62) is the player
        - Status: untested
        - Evidence: DOWN changed 2 cells at the left edge of the band (b->3)
        - Test: try RIGHT/LEFT/UP, see if the band moves or changes

        ## H8: 0/1 cross is a static marker/object
        - Status: untested
        - Evidence: didn't move on DOWN
        - Test: try more actions, see if it ever moves


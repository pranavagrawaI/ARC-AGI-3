# Hypotheses — game-820258

        | # | hypothesis | status | evidence |
        |---|---|---|---|
        | H1 | player is colors 0+1, moves on 3 (platforms) | contradicted | 0/1 static after 6 actions |
        | H6 | arrow keys shift 12/9 in that direction by 5 cells | confirmed | 6 actions all consistent |
        | H7 | 11 bar erodes from left by 2 per action | confirmed | 84→82→80→78→76→74→72 |
        | H8 | game goal: position 12/9 stack at 0/1 target | untested | plausible; 0/1 is static marker in left corridor |
        | H9 | 11 bar is a timer — game over when it runs out | untested | 84 cells / 2 per action = ~42 actions max |
        | H10 | stack can cross gap (cols 26-28) via wide bar (rows 25-29) | untested | 3-region structure allows it |

        ## Confirmed dynamics
        - Stack (12+9, 5×5) moves 5 cells per arrow in that direction
        - 11 bar: starts at 84 cells (2×42), erodes 2 from left per arrow action
        - 0/1 marker: static at rows 31-33, cols 20-22
        - Stack position tracked as top-left of 12 blob
        - 3-region corridors: left (cols 14-25), gap (cols 26-28), right (cols 29-53)
        - Wide bar at rows 25-29 spans cols 14-53 (stack can cross gap here)
        - Top box at rows 8-16, cols 28-36 (9×9, 3-border, 5-interior with some 9s)
        
        | H8 | game goal: position 12/9 stack at 0/1 target | confirmed | 0/1 collected when stack overlapped |
        | H11 | after 0/1 collection, next goal is top box or another location | untested | 0/1 gone, game not finished |
        | H12 | 0/1 collection animation (0-cells at top box + bottom-left) is a score effect | untested | animation showed transient 0-cells |


        | H13 | shaft is at cols 34-38 (NOT 29-33) — doesn't align with top box interior | confirmed | grid analysis |
        | H14 | top box interior (cols 29-35) is NOT accessible from shaft (cols 34-38) | confirmed | shaft overlaps box right border |
        | H15 | goal is to collect 9-clusters or reach 8-grid | untested | need to explore |


        | H16 | Level 1: left box border is solid (9-cells at top block entry) | confirmed | stack blocked at row 38 |
        | H17 | Level 1: right box border is solid (no gap) | confirmed | grid analysis |
        | H18 | Level 1: goal is NOT the left box or right box | untested | both have solid borders |
        | H19 | Level 1: goal might be to collect the 0/1 from outside the right box | untested | 0/1 is inside right box |
        | H20 | Level 1: goal might be a specific stack position (not a box) | untested | need to explore |


        | H21 | Level 1: 0/1 is a HAZARD (resets level when stack overlaps) | confirmed | level reset when stack overlapped 0/1 |
        | H22 | Level 1: goal is to reach the right block (rows 39-49, cols 44-56) | untested | only accessible area not yet explored |
        | H23 | Level 1: goal is to reach the bottom bar (rows 50-54, cols 39-54) | untested | connects to right block |
        | H24 | Level 1: 9-cells in left box border are a "door" that opens under specific conditions | untested | stack blocked at row 38 |


        | H25 | 11 bar running out triggers a LEVEL RESET | confirmed | DOWN from wide bar with empty 11 bar → reset to start position, 11 bar restored to 84 |
        | H26 | 11 bar is a "lives" mechanism: when empty, next action resets the level | confirmed | same as H25 |
        | H27 | Level 1: goal is to enter the left box (rows 38-46, cols 12-20) | untested | left box border appears solid but 9-cells in top border might be passable |
        | H28 | Level 1: goal is to navigate to a specific position (not a box) | untested | no clear "box" goal identified yet |


        | H29 | Wide bar at rows 10-14, cols 14-18 is a DEATH TRAP | confirmed | 7th LEFT from cols 19-23 triggered DEATH (reward -1.0) |
        | H30 | Left arm (rows 25-37, cols 14-18) is INACCESSIBLE from the wide bar | confirmed | death trap at cols 14-18 blocks access |
        | H31 | Left box (rows 38-46, cols 12-20) is INACCESSIBLE | confirmed | left arm is inaccessible |
        | H32 | Level 1: goal is to reach the middle section (rows 15-24, cols 29-38) | untested | accessible in 9 moves |
        | H33 | Level 1: goal is to reach the right section (rows 15-24, cols 44-58) | untested | accessible in ~24 moves |
        | H34 | Level 1: 11 bar erodes at 2 cells/move, max 42 moves per attempt | confirmed | observed consistently |


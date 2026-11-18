# Hypotheses (turn 0)
Game = click-based. available_actions = [RESET, CLICK]. Player = bbb (color 11) cluster at rows 13-16, cols 29-34 (top-center mass ~ rows 13-15, cols 30-33, plus a tail at rows 33-37).
Goal = 9-diamond centered (col 36, row 55), radius 6 (max manhattan dist).
Board = 0-filled interior rows 9..46, cols 14..47, framed by 5. Obstacles: 4444 blocks (5x5) and 5-pixels inside.

H1: clicking a cell that is part of the player cluster (color 11) moves the player DOWN by 1 row (cluster shifts +1 row; vacated cells become 0). Deciding test: CLICK 31 14 (center top) -> expect player cluster shifted down.
H2: clicking a 9-cell grows the diamond by +1 radius layer (radius 6->7), overwriting cells it covers with 9. Deciding test: CLICK on a 9 cell, e.g. 36 51 -> expect 9-layer added around, radius 7.
H3: clicking empty 0 or 5 => no_impact.
H4: Win condition: player reaching/entering goal diamond? Untested.


## Turn 1 update
H1 (click player 11 cell moves player down): **CONTRADICTED**. After CLICK 31 14, player cluster
identical; only status-band cell (1,60) changed 9->3, labeled "move".
NEW H5: status band (row 1) is a counter/timer; clicking player increments it (digit shown at col60).
NEW H6: "move" label = any playfield-or-status change (even band-only).
Next test: click player again -> expect band digit to advance (3->4?).


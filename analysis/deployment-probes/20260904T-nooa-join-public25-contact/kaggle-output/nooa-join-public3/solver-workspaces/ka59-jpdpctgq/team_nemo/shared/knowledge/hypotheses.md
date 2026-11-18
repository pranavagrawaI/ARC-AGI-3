# Hypotheses (level 0, first frame)

| id | hypothesis | status | evidence |
|----|-----------|--------|----------|
| H1 | 5 (at (31,28)) is the player, moved by UP/DOWN/LEFT/RIGHT | untested | single pixel, interior of arena |
| H2 | 0 (at (31,19)) is the goal/exit hole; stepping on it completes the level | untested | single pixel, aligned with player row |
| H3 | 1, 14 (e), 15 (f) are impassable walls | untested | f-block fills a 6-wide column region; arena border is 1 |
| H4 | 4 cells (rows 26-30, cols 44-48; row 63) are floor / status band, not walls | untested | row 63 is full-width bottom border (status band) |
| H5 | 'e' (14) cells form movable or destructible obstacles; CLICK may affect them | untested | e-cells appear in 3 clusters |
| H6 | The white f-strip (cols 33-38) is a static wall column blocking the arena into two halves | untested | solid 6x21 block |

Deciding tests: move LEFT/UP/RIGHT and see which cells the player can enter;
try CLICK on an e cell or the goal.


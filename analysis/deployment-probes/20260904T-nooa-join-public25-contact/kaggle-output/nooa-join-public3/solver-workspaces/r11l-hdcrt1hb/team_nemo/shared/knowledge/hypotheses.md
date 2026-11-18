# Hypotheses

| hypothesis | status | evidence |
|---|---|---|
| H1: game has exactly 2 actions: RESET and "CLICK x y" (no movement) | untested | available_actions=['RESET','CLICK'] at turn 0 |
| H2: grid has 2 isosceles triangles of bg color 5 (top, bottom), black 0 margins at top/bottom corners (2), and 4 "object" glyphs: (a) X of color f at (18..24, 36..42), (b) X of color 3 at (57..61, 25..29), (c) 5x5 block of f with a 6 at center (45..49, 15..19), (d) thin diagonal line of 1's from (36,7) to (57,27) with a hole (0) square at (34..38, 6..8) | untested | turn-0 grid |
| H3: objects move when CLICKed | untested | no data |
| H4: goal involves aligning objects or filling holes | untested | no data |

Deciding test: CLICK each special cell, observe what moves where.


## Turn 1 findings (CLICK 12 41)

| hypothesis | status | evidence |
|---|---|---|
| H1: CLICK x y moves the snake HEAD to (row=y, col=x) | **CONFIRMED** | f-head moved (36,7)->(41,12) = exactly the click position |
| H2: The snake is one connected entity: f-head + black diamond + 1-cell body + 5x5 f-block tail | **PARTIALLY CONFIRMED** | All objects shifted together; 1-cell count changed 17->12 |
| H3: Body 1-cell count = Manhattan(head, tail) - 4 | untested (n=2) | 21-4=17 ✓, 16-4=12 ✓ |
| H4: Tail moves by a fixed rule when head jumps | untested | tail (47,17)->(50,19), delta (+3,+2); head delta (+5,+5) |
| H5: f-X (top) and 3-X (bottom) are static landmarks/goals | untested | unchanged after click |

### Click semantics
- "CLICK x y" → head teleports to (row=y, col=x).
- Body re-routes between new head and tail.
- Tail also moves (rule unknown yet).
- Black diamond (5x5 hole pattern) is centered on the head and moves with it.

### Structure inventory (g1, current)
- f-head: (41, 12)
- Black diamond: rows 39-43, cols 10-14 (centered on head)
- 1-cells (body): 12 cells between head and tail
- 5x5 f-block (tail): rows 48-52, cols 17-21, 6-center at (50,19)
- f-X (static): rows 18-24, cols 36-42
- 3-X (static): rows 57-61, cols 25-29


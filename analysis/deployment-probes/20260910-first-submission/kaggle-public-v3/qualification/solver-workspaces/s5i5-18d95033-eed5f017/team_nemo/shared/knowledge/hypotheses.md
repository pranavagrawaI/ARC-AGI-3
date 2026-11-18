# Hypotheses (level 0)

Grid: 64x64, bg=5, bottom row=3 (full row 63), no movement actions, only RESET + CLICK.

| Hypothesis | Status | Evidence / deciding test |
|---|---|---|
| H1: Game is a puzzle: click colored shapes to make them disappear/transform; level done when playfield is clean (only bg + frame row 63) | untested | initial layout: 6 'letter' shapes in bg area 1-62 (rows 9-47): F-frame(9-11,27-32 + F col39 rows 9-11), B-frame box(18-24,36-48) w/ inner C+3+4 pattern, F-frame(27,9-11), D-slab w/ F(28-35,9-11), B-frame box(35-47,21-27) w/ inner C+D+3 pattern, F-frame(51-53,10-12) |
| H2: Clicking center of a shape triggers something specific (removal / morph / swap) | untested | test: CLICK 10 32 (center of box1) |
| H3: RESET restarts level (no cost in score? still counts as action) | untested | avoid until needed |

First experiment: click dead center of box 1 (36-48, 18-24) -> (32, 32), expecting shape removal or morph.


| H4: bottom row 63 is a per-click counter filling RIGHT-TO-LEFT with color 4 (1 cell per click, from col 63 down); full row (64 fours) = level complete | TESTING | step1: (63,63) 3->4 after 1 click. step2: CLICK 1 1 (empty bg) added (63,61),(63,62) -> 2 cells in ONE click! So not 1/click: maybe +1 per click plus… |
| H5: amount added = f(click position/shape)? step1 added 1 (click 32,32 on shape), step2 added 2 (click 1,1 bg) | untested | next: same click twice (1,1) to measure per-repeat delta |


MEASURED click->bar deltas: click1 (32,32): +1; click2 (1,1): +2; click3 (1,1) repeat: +1.
Pattern: +1, +2, +1 — candidate 1/2/1 periodic. Row63 fills right-to-left with 4; 4 fours so far.
H6: counter is periodic with period 3: 1,2,1,1,2,1,... (next should be +1)
H7: increments = f(position parity) — but repeat at (1,1) broke pure-position theories... unless position+step interacts.
Test: click (1,1) again; if +1 (pattern 1,2,1,1...) H6 survives; if +2 the period is 2 with phase (step%2).


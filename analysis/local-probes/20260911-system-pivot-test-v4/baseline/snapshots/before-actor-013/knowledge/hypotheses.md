## Mechanics
| claim | evidence | tested scope | open |
|---|---|---|---|
| player = 5x3 color-9 block, moves with UP/DOWN/LEFT/RIGHT | static at step 0 | none | step size (1 vs 2) |
| 12 trail stays behind player | static at step 0 | none | regenerates? |
| 11 bar = timer/budget | 70 cells at step 0 | none | does it change on move? |

## Objective (unresolved)
- H1: reduce 11 bar to zero.
- H2: reach the 0/1 cross.
- H3: reach the 8-block.

| claim | evidence | tested scope | open |
| 9-block step size: 5 cols horizontal | step 1 LEFT | one direction | vertical step size (2 rows?) |
| 11 bar: -2 cells per move from left end | step 1 (70→68) | one move | what at 0 (game over?) |
| 12 trail: 2x5 left behind vacated spot | step 1 | one move | persistence, overwrites? |


# Hypotheses
| hypothesis | status | evidence / deciding test |
|---|---|---|
| SPACEBAR moves each lever's 2-stripe down by 1 cell; lever stops when stripe reaches row 32 (bottom cell) | untested | first SPACEBAR test turn |
| Win = all four stripe cells on same row (a "line"), after which level completes | untested | observe when stripes align; level up? |
| Palette (9,e,b,f) vs targets (e,f,9,b) is cosmetic, lever identity follows column | untested | does win depend on matching colors? |
| STOP/UNDO are no-ops or undo a step | untested | test later |

## turn 1 update
| hypothesis | status | evidence / deciding test |
|---|---|---|
| SPACEBAR moves levers down | CONTRADICTED | step 1: levers unchanged at rows 29-30; only row 53 col 63 changed 2->3 |
| SPACEBAR increments a counter (grid[53,63]) | confirmed x1 | 2->3 after one press; predict +1 per press |
| Win condition unknown — maybe counter must reach a target, or CLICK on a target block | untested | keep pressing SPACEBAR; watch for level completion / other changes |
| CLICK is the interaction; maybe click target blocks to select them | untested | test CLICK on a bottom target square |


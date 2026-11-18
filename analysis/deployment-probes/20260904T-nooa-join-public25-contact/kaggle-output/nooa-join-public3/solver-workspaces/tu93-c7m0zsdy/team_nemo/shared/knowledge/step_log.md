
## Turn 0 (repeat state) — world model built, initial grid parsed
- Grid = one maze component rows/cols 15-47. Player (4) at (16,17) on a 3x3 pad of 9s
  (r15-17,c15-17, center replaced by player). Goal = 3x3 pad of 14s at r45-47,c45-47.
  Walls = color 2 (288 cells), floor = color 0 (261). Corridors are 3 cells wide.
  Status band = row 63 all 6s.
- BFS shortest path (single-step model): 61 actions.
  Pattern: DOWN×16 → RIGHT×10 → DOWN×16 → RIGHT×12 → UP×1 → RIGHT×6.
  Endpoint (47,45) is inside the goal pad.
- Turn-0 DOWN probe: I submitted it, harness marked submitted_this_turn=True, but the
  turn-0 notification is the SAME first state (step 0). Player still at (16,17),
  trajectory empty. Likely the probe's result state hasn't been flushed yet, or the
  previous turn's submit was consumed without executing. Need to resubmit a probe.

### Carry-Forward
- z-schema: {player:(r,c), goal:3x3 bbox, walls:frozenset(2-cells)}. predict = single step,
  wall-blocked. Helper maze.py written and loaded.
- Plan: submit ONE probe DOWN (expect move to (17,17)). Once batching is earned, blast
  the BFS path (or chunks of it). Watch for slide mechanics: if DOWN moves player more
  than 1 cell, re-derive predict as slide-until-wall.


## Turn 0 — first frame
- Arena: rows 21..41, cols 9..53, border color 1.
- Player (5) at (31,28). Goal (0) at (31,19). Wall f-strip cols 33..38 rows 21..41.
- e (14) clusters: (30-32,18-20), (30-32,27-29), (32,11-15),(33-36,11),(33-36,15),(36,11-15).
- 4 (green) 5x5 open cell rows 26-30, cols 44-48; bottom row 63 is status band.
- Plan: probe moves LEFT, RIGHT, UP, DOWN one at a time; expect "move" or "no_impact".
- Helper arc.py written (encode/predict).

## Turn 1 — CLICK on player
- Action: CLICK 28 31 (clicked on player at row=31, col=28)
- Prediction: no_impact — WRONG (outcome was "move")
- Changes:
  - player(5) at (31,28) → GONE
  - goal(0) at (31,19) → 4 (floor)
  - goal(0) appeared at (31,28) (where player was)
  - goal(0) appeared at (63,63) (was 4)
- Interpretation: clicking the player "activated" it. Player vanished, became a goal marker.
  The old goal position became floor. A new goal appeared at (63,63).
- This is very strange. Maybe the game has a different mechanic than I assumed.
- Carry-Forward: player is gone, two goals exist at (31,28) and (63,63). Need to figure out
  what to do next. The 14-tiles (e) form two 3x3 rings around (31,28) and (31,19).


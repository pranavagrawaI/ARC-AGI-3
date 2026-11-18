# Step log

## Turn 0 (initial observation)
Level 0 grid layout (color codes):
- 5 = background (black), 4 = dark-gray frame, 3 = mid-gray band, 2 = light-gray/olive, 0 = white, f = purple
- Frame: bottom row 63 all 4; top-left panel border: col 17 rows 0-17, rows 16-17 cols 0-17.
- Mid-gray band rows 0-8, cols 18-63 (ceiling/HUD area) containing icons at rows 2-7:
  - icon A: white 3x3 at (3-5, 36-38) in 5x6 frame
  - icon B: purple 3x3 at (3-5, 42-44) in 5x6 frame; white underline row 7 cols 41-45 (looks "selected")
- Top-left portrait: 10x10 at rows 3-12 cols 3-12 = white rows 3-7, purple rows 8-12.
- Center: light-gray (2) 14-wide top bar row 24 cols 25-38 + corner pixels (32,25),(32,38); purple (f) 13x7 fill rows 25-31 cols 26-37.
- Center-bottom: white (0) 10x10 at rows 34-43, cols 27-36.
Next: test movement keys (UP first).

## Turn 0 plan (experiment)
Test if white 10x10 square is the player: LEFT, LEFT (clear space), UP, UP (may hit purple tray at rows 24-31).
Predictions: if player, each action diffs ~100 cells (10x10 move); if blocked, 0 diff for UPs.


## Turn 1 (re-entry)
- Received a delayed/duplicate notification of the INITIAL state (step 0, "first state"); my [LEFT,LEFT,UP,UP] batch (submitted prev turn, submitted_this_turn=True) has NOT executed yet — env_step still 0, game_states queue empty, game-state tail job running (state in flight).
- Polled game_states queue for ~10s: no new state.
- Decision: WAIT for the post-action state; do not queue blind extra actions that would contaminate the experiment.


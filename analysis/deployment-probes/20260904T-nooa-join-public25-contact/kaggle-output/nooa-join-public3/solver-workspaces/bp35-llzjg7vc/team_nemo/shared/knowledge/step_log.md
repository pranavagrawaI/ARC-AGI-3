# Step Log

## Turn 0 (step 0) — initial state, no actions yet executed
- Level 0. available: RESET, LEFT, RIGHT, CLICK, UNDO (no UP/DOWN — 1-DOF player).
- Grid: border=5 (with 3-pixel X notches every 4 rows/cols on the inner edge),
  playfield=10, two 5x5 'e' squares at (1-5,13-17) and (1-5,21-25)? — actually rows 1-5
  cols 13-29 and 13-17 cols 31-53, plus scattered 3-pixels (178 of them),
  a 3x4 '9' blob at (37-40, 20-21) with two 'b' pixels at (38-39,22).
- Bottom row 63 is all 0 — status band.
- Plan: probe LEFT to see what moves (is the 9-blob the cursor? do e-squares scroll?).


## Turn 1 (step 0) — resubmitting probe (harness dropped the first)
- State still level 0 step 0, no action_results. Game-state tail job is running.
- Submitting LEFT as probe; predicting the 9-diamond cursor at (37-40,20-21) shifts
  one column left (or the e-bit band scrolls).


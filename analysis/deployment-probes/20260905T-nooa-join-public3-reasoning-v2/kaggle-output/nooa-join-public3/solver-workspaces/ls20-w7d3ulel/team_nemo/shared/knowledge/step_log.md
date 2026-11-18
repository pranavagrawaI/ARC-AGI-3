# Step log
## Turn 0 (step 0)
- Parsed initial grid: sprite center (32,21); goal patch 9 at rows47-49,c34-38 (orange 12 above it rows45-46); HUD bar/lives bottom; head-box icon top.
- Wrote worldmodel.py helper: encode->z{base,pos}, predict one-step slide/step unknown.
- Submitted LEFT to disambiguate 1-step vs slide (H1/H2).

## Turn 1 (still step 0)
- Woken by the first-state line (tail producer delivered it late); LEFT batch from turn 0 is
  registered (submitted_this_turn=True) but env hasn't stepped yet (trajectory empty).
- Waiting for post-LEFT state to score H1 (1-step) vs H2 (slide): sprite center should be
  (32,20) if 1-step, or (~32,15) if it slides to the torso left wall (col 14 boundary).


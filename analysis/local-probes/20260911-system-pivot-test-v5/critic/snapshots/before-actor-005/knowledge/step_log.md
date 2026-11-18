
## Turn 0 (step 0) — initial inspection
Grid layout (64x64):
- 4 = background; 5 = outer frame + boxes; 3 = floor/walls (inner region).
- 9-block A: rows 11-13, cols 35-37 — 3x3 cluster (with 5-pixel 'B' shape) inside a 3-box
  (rows 8-16, cols 35-43). This looks like a START marker.
- 9-block B: rows 56-62, cols 1-9 — a larger 'B' shape lower-left (rows 56-60 cols 1-7,
  plus rows 60-62 cols 4-9). Also color-9; possibly a second start/decoration.
- 0/1 cross: rows 31-33, cols 20-22 — small target-looking cross (0 center, 1 L-shape).
- 8-block: rows 61-62, cols 56-63 — 2x8, bottom-right corner.
- 11-bar: rows 61-62, cols 16-54 (39 cols x 2 rows = 78... measured 70 cells) — long bar
  across the bottom. Possibly a timer/decrementing counter.
- 12-trail: rows 35-36, cols 39-43 — 2x5 strip just below center. Possibly a "spawn"/teleport
  marker or a trail left by a previous actor.
- No visible player marker distinct from the color-9 clusters yet.

Hypotheses:
- H1: 11-bar decrements each move (timer); goal = reach target before zero.
- H2: 12-trail marks where something must go (goal = place/step-on).
- H3: 0/1 cross is the target (goal = reach it).
- H4: 8-block is a target/teleporter.
- H5: 9-clusters are player(s); 9-A (inside box) may be start, 9-B lower-left decoration.

Probe: UP (reversible; should move a player if one exists and disambiguate which
color-9 cluster is the player, and whether 11-bar decrements).


# Hypotheses (Level 0, turn 0)

Grid: color 5 = walls/borders (a frame with X-pattern notches), color a = playfield,
color e = "bits" (two 8x8-like bitmaps: (1-5,13-29) and (13-17,31-53)),
color 3 = scattered single pixels, color 9 = small diamond (37-40, 20-21),
bottom row 63 = color 0 (status band?).

| Hypothesis | Status | Evidence / Deciding test |
|---|---|---|
| Game is a "click/merge" puzzle: player moves a cursor with LEFT/RIGHT (no UP/DOWN!) and CLICK toggles/selects cells to transform the e-bits | untested | available_actions has only LEFT, RIGHT, CLICK (no UP/DOWN) → player is 1-DOF; test with LEFT then grid diff |
| The 9-diamond is the player/cursor | untested | watch where it goes on LEFT/RIGHT |
| Goal: assemble/transform e-bits (e.g. light up a pattern), maybe e's are "on" pixels | untested | |
| e bits are 7-segment-digit-like glyphs | untested | 8x5 blocks: rows 1-5 cols 13-29 (width 17 ≈ 2 digits of 8 + gap?) |
| Win = make two bitmaps match / form a letter | untested | |


## Turn 1 result (LEFT)
- Sprite 4x5 at (37-40, 15-17) [9-diamond + 2 b-pixels] moved LEFT by 5 cols → (37-40, 10-12)
  Actually: pre-LEFT 9s at cols 20-21, b at col 22 → post-LEFT 9s at cols 15-16, b at col 14. Moved -5 cols.
- e-bits (147 total, two 5x5 squares at (1-5,13-17) and (1-5,21-25) approx) did NOT move.
- New f-pixel appeared at (63, 0) — bottom status band.
- Prediction "move" was CORRECT.

| Hypothesis | Status | Evidence / Deciding test |
|---|---|---|
| LEFT/RIGHT moves the 9-diamond sprite horizontally by 5 cols | CONFIRMED | 9s went col 20-21 → col 15-16, b went col 22 → col 14, delta = -5 |
| The e-squares are fixed "bit" targets that the sprite aligns under | untested | sprite center col after LEFT = 15; e-square 1 is cols 13-17, center 15 — ALIGNED |
| CLICK selects/toggles the cell under the sprite, changing e-bits | untested | try CLICK when aligned under e-square |
| Goal: light up / fill the e-squares by clicking under them | untested | |
| f-pixel at (63,0) = step counter or score | untested | watch it after next action |


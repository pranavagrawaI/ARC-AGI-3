# Actor beliefs — Browser/Yields game

## Board layout (64x64) level 0
- Background (empty space) = 9 (value 9 theta-computed). Orange wall markers = 10 ('a').
- Floor = row 63 all green 5. Left wall of green 5 = col 0 (col3 has a stray).
- Bottom-right RED PAD = b (11), a rigid shape at rows45-53 approx cols51-59 (45 cells).
- Top-right: move counter on col 63 (green 5 cells filling down from row0, one per turn).
- TWO MOVER BLOCKS in "play area" upper region:
  - GREEN block (5): at origin rows15-23, cols18-26 (40 cells). Has holes (0).
  - GRAY block (4): at origin rows15-23, cols36-44 (45 cells).
- The two blocks are mirror-images positioned symmetrically about a vertical axis.
  - mirror(green) about axis ~31.5 lands on gray: green col18..26 -> 37..45; gray is at 36..44 (off by ~1).
  - gray shape ≈ congruent (colours ignored) to the right RED pad. green (40) != pad (45).
  - green mirror axis 30.5 gives small intersection; axis 31 mirror is an exact SUBSET of gray (misses 5 gray cells at rows16,19,22).

## Mechanics (from observed transitions, model builder confirms high confidence)
- Two rigid blocks share a row offset. LEFT/RIGHT move COLUMNS ±3 apart/together; UP/DOWN move ROWS ±3.
  - Actually LEFT pushed GREEN left & GRAY (block cols apart); they move symmetrically away/toward each other.
  - Turning the pair: LEFT/RIGHT advances each block's col by ±3 toward/away; DOWN/UP moves rows +3/-3.
- Every turn-taking action (direction + SPACEBAR + presumably CLICK) advances a step counter filling col63 from top (green 5).
  - At t11 RESET -> no col63 fill. After 3 actions (t12 DOWN,t13 LEFT,t14 SPACE) col63 rows0,1,2 green.
- RESET rewinds to initial pose. BLOCKS and pads return to origin.
- Not yet probed: wall collision / clamping, CLICK semantics, UNDO semantics, falling at bottom.

## Open hypotheses (outcome)
- H1 (builder): blocks converge to a mirror-symmetric pose around the orange axis => win.
- H2: gray block must align/insert onto the right RED pad (target socket).
- H3: move-budget col63 exhaust => GAME_OVER.
None verified. No terminal/GAME_OVER observed yet.

## Next steps
- Probe collision/clamp: move blocks far apart until wall, and to extremes.
- Determine whether win triggers when blocks reach symmetric extreme pose or when aligned.
# Hypotheses — level 0

H1 | 4-directional movement: arrow key moves the player sprite (cross of colors 0/1) by 1 cell | untested | test: submit DOWN, expect sprite shifts to row 33, col 21, rest unchanged
H2 | color 4 = impassable wall (blocks movement); other colors (3,5,8,9,11,12) passable until disproven | untested | decide at first wall contact (e.g., LEFT ~7 steps hits color 4)
H3 | Goal: reach/step on the color-12 pad (2x5, rows 45-46, cols 34-38) | untested | reach pad; observe level_completed or trap
H4 | color-9 blocks are static decoration/obstacles (3 clusters) | untested
H5 | colors 5/8/11 outside the 3-floor are out-of-bounds area (rows 52+ is a separate band w/ 11=dark band, 8=three 2x2 dots) | untested

## turn 1 update
H1 (sprite is movable player) | CONTRADICTED | DOWN left cross at (32,21) unchanged; diff = (61,13),(62,13): 11->3 only
H6 | each arrow press converts one 11-band slot (rows61-62, cols13-54) to 3, leftmost-empty first | partial (1 sample: DOWN) | decide with RIGHT: expect col14 slot filled next
H7 | level completes when band fully filled (42/42 slots = 3) | untested | watch for level_completed as filled -> 42
H8 | 0/1 cross is a static icon/decoration, not the player | supported (1 sample)
H9 | 9-clusters & 12-pad & 8-dots are static decoration until shown otherwise | untested



"""World model for the arc grid game.

Level 0 (first frame):
- Arena: wall color 1, rows 21..41, cols 9..53. Interior rows 22..40, cols 10..52.
- Player: color 5, single pixel at (31,28), fully surrounded by 14 ("e").
- Goal: color 0 at (31,19).
- 14 ("e") = destructible/interactable tiles (form clusters around player and goal).
- 15 ("f") = tall block cols 33..38, rows 21..41 (static wall hypothesis).
- 4 = open floor (5x5 at rows 26..30, cols 44..48); row 63 = status band.

z = {
  "player": (r, c) | None,   # color-5 pixel
  "goal":   (r, c) | None,   # color-0 pixel
  "walls":  frozenset((r,c)),# colors {1,15} cells
  "e":      frozenset((r,c)),# color-14 cells (destructible?)
  "arena":  (21, 41, 9, 53),
  "level":  int,
}
"""
import numpy as np

ARENA = (21, 41, 9, 53)
BG = 2
PLAYER_C = 5
GOAL_C = 0
E_C = 14
WALL_COLORS = (1, 15)
DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def _cells(A, c):
    pos = np.argwhere(A == c)
    return {(int(r), int(cc)) for r, cc in pos}

def encode(grid, level=0):
    A = np.asarray(grid)
    pc = _cells(A, PLAYER_C)
    gc = _cells(A, GOAL_C)
    z = {
        "player": sorted(pc)[0] if pc else None,
        "goal": sorted(gc)[0] if gc else None,
        "walls": frozenset(_cells(A, 1) | _cells(A, 15)),
        "e": frozenset(_cells(A, E_C)),
        "arena": ARENA,
        "level": level,
    }
    return z

def passable(z, r, c):
    r0, r1, c0, c1 = z["arena"]
    if not (r0 <= r <= r1 and c0 <= c <= c1):
        return False
    return (r, c) not in z["walls"]

def predict(z, action):
    """z' , expect  (expect in move/no_impact/level_up/unknown).
    Assumptions (to be validated):
      - movement blocked by walls AND by e-tiles
      - entering goal cell -> level_up
      - CLICK not modeled (returns z, unknown)
    """
    r, c = z["player"]
    if action in DIRS:
        dr, dc = DIRS[action]
        nr, nc = r + dr, c + dc
        if passable(z, nr, nc) and (nr, nc) not in z["e"]:
            n = dict(z, player=(nr, nc))
            return n, ("level_up" if (nr, nc) == z["goal"] else "move")
        return z, "no_impact"
    return z, "unknown"


"""World model for the grid game.

Grid facts (level 0, turn 0):
- 64x64. 0/1 = white plus-shaped SPRITE at center (32,21).
- 3 = gray floor (the big "body" region), 4 = dark background / walls,
  5 = light-gray border (head box, HUD icon), 9 = blue, 12 = orange,
  8 = red HUD dots, b = yellow HUD bar.
- Static map = grid with sprite cells restored to 3.
"""
import numpy as np

SPRITE_COLORS = (0, 1)
# colors considered walkable (untested: 9/12 assumed walkable, 4/5 assumed wall)
WALKABLE = (3, 9, 12)

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def encode(grid: np.ndarray) -> dict:
    """grid: 64x64 int array -> latent z dict."""
    spr = np.isin(grid, SPRITE_COLORS)
    ys, xs = np.where(spr)
    # sprite center = bbox center
    r = int((ys.min() + ys.max()) // 2)
    c = int((xs.min() + xs.max()) // 2)
    base = grid.copy()
    base[spr] = 3  # restore floor under sprite
    return {"base": base, "pos": (r, c)}

def render_sprite(base: np.ndarray, pos) -> np.ndarray:
    """Return grid with sprite plus-shape drawn at pos."""
    g = base.copy()
    r, c = pos
    g[r, c] = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < 64 and 0 <= cc < 64:
            g[rr, cc] = 1 if (dr, dc) in ((1, 0), (0, -1)) else 0
    return g

def predict(z: dict, action: str) -> dict:
    """One-step dynamics. Returns (z', outcome) where outcome is
    'move' | 'no_impact' (| 'unknown' if base changed unexpectedly)."""
    base, (r, c) = z["base"], z["pos"]
    if action not in DIRS:
        return z, "unknown"
    dr, dc = DIRS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < 64 and 0 <= nc < 64):
        return z, "no_impact"
    if base[nr, nc] in WALKABLE:
        return {"base": base, "pos": (nr, nc)}, "move"
    return z, "no_impact"

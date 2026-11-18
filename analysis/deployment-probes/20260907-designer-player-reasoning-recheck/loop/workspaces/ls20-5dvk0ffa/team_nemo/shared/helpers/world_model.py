"""World model for the 64x64 grid game (v3).

z fields (JSON-safe; the harness roundtrips z through JSON, so NO tuples):
  player   -- list [row, col] = topmost '0' cell of the sprite (or None)
  sprite   -- list of [row, col], all 0/1 cells
  walk     -- list of [row, col], walkable cells (provisional colors 0,1,3,9,12)
  items9   -- list of [row, col], color-9 cells
  blocks12 -- list of [row, col], color-12 cells
  hud      -- list of [row, col], colors 8/11 (bottom band rows 61-62)

Dynamics (v3, untested): moving translates the 5-cell sprite by the direction
vector; blocked if the target (checked at the reference '0' cell) is off-grid
or not walkable. UP = row+1 (screen-down): confirmed once by step 1 (diff 52px
sprite-only change, sprite shifted +1 row). Pickup/goal model: still unknown.
"""
import numpy as _np

WALK_COLORS = (0, 1, 3, 9, 12)

DIRS = {"UP": (1, 0), "DOWN": (-1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def _top0(grid):
    ys, xs = _np.where(grid == 0)
    if len(ys) == 0:
        return None
    r = int(ys.min())
    return [int(r), int(xs[ys == r].min())]

def _cells(mask):
    return [list(map(int, p)) for p in _np.argwhere(mask).tolist()]

def encode(grid):
    return {
        "player": _top0(grid),
        "sprite": _cells((grid == 0) | (grid == 1)),
        "walk": _cells(_np.isin(grid, WALK_COLORS)),
        "items9": _cells(grid == 9),
        "blocks12": _cells(grid == 12),
        "hud": _cells(_np.isin(grid, (8, 11))),
    }

def predict(z, action):
    z2 = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in z.items()}
    d = DIRS.get(action)
    p = z.get("player")
    if d is None or p is None:
        return z2  # RESET / unknown / no player: no change (provisional)
    r, c = int(p[0]), int(p[1])
    r2, c2 = r + d[0], c + d[1]
    walkset = {tuple(q) for q in z.get("walk", [])}
    if 0 <= r2 < 64 and 0 <= c2 < 64 and (r2, c2) in walkset:
        z2["player"] = [r2, c2]
        z2["sprite"] = [[r + d[0], c + d[1]] for (r, c) in z.get("sprite", [])]
    return z2

def render(z):
    return (f"player={z.get('player')} n_sprite={len(z.get('sprite', []))} "
            f"n9={len(z.get('items9', []))} n12={len(z.get('blocks12', []))} "
            f"n_hud={len(z.get('hud', []))}")

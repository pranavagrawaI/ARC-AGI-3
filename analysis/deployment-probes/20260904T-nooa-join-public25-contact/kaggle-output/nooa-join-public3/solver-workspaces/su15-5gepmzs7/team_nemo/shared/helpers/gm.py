"""World model for the click-grid game (turn 0 baseline)."""
import numpy as np

BG = 5  # background field color

def parse(grid_rows):
    """64 hex rows -> 64x64 int array (row 0 = top)."""
    return np.array([[int(c, 16) for c in row] for row in grid_rows], dtype=int)

def diff(a, b):
    """(row, col, old, new) for every differing cell."""
    d = np.argwhere(a != b)
    return [tuple(int(v) for v in p) for p in d]

def changed(a, b):
    """{(r,c): (old,new)} for every differing cell."""
    return {(int(r), int(c)): (int(a[r, c]), int(b[r, c]))
            for r, c in np.argwhere(a != b)}

def blobs(g, colors, bg=BG):
    """Largest connected component (8-conn) of `colors`; bbox + pixel count."""
    mask = np.isin(g, colors)
    if not mask.any():
        return None
    seen = np.zeros_like(mask, dtype=bool)
    best = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                comp = []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < g.shape[0] and 0 <= nx < g.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                if len(comp) > len(best):
                    best = comp
    if not best:
        return None
    rs = [p[0] for p in best]
    cs = [p[1] for p in best]
    return {"count": len(best), "r0": min(rs), "r1": max(rs), "c0": min(cs), "c1": max(cs),
            "center": ((min(rs) + max(rs)) / 2, (min(cs) + max(cs)) / 2)}

def encode(grid):
    """z = latent state. grid: 64x64 array.  Baseline (dynamics unknown yet).
    Z_SCHEMA:
      blob        bbox of the 9/3 clump (the 'ball'): count/r0/r1/c0/c1/center
      dots        list of 3-color dots along the anti-diagonal (the 'track')
      top_block   static white 3x3 rows 4-6 cols 30-32 (HUD?)
      bottom_block static white 3x3 rows 58-60 cols 3-5 (HUD?)
      crosshair   small 0 crosshair near (53,10)
    """
    g = parse(grid) if isinstance(grid, (list, tuple)) else grid
    return {
        "blob": blobs(g, [3, 9]),
        "dots": [tuple(int(v) for v in p) for p in np.argwhere(g == 3)],
        "top_block": [4, 30, 6, 32],
        "bottom_block": [58, 3, 60, 5],
        "crosshair": (53, 10),
    }

def predict(z, action):
    """No dynamics known yet; returns a copy. Mismatch is the learning signal."""
    import copy
    return copy.deepcopy(z)

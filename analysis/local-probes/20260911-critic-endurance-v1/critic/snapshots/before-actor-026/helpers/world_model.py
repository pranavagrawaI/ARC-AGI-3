"""World model: pink band erosion + yellow/teal block sliding by 5."""
import numpy as np

def _extents(rows, cols):
    if len(rows) == 0:
        return None
    return [int(min(rows)), int(max(rows)), int(min(cols)), int(max(cols))]

def encode(grid):
    """grid: 64x64 int array. Latent: band left/right, and 12- and 9-cluster bboxes
    limited to the big central field (rows 35..55, cols 20..45)."""
    g = np.asarray(grid)
    pink_cols = [int(c) for c in range(64) if g[61, c] == 11 and g[62, c] == 11]
    band = [min(pink_cols), max(pink_cols)] if pink_cols else None
    sub = g[35:56, 20:46]
    y_r, y_c = np.where(sub == 12)
    t_r, t_c = np.where(sub == 9)
    yellow = _extents(list(y_r) + [0], list(y_c) + [0])  # placeholder to satisfy sig
    # redo properly with offsets
    if len(y_r):
        yellow = [int(y_r.min()) + 35, int(y_r.max()) + 35, int(y_c.min()) + 20, int(y_c.max()) + 20]
    else:
        yellow = None
    if len(t_r):
        teal = [int(t_r.min()) + 35, int(t_r.max()) + 35, int(t_c.min()) + 20, int(t_c.max()) + 20]
    else:
        teal = None
    return {"band": band, "yellow": yellow, "teal": teal}

def _shift(bbox, dr, dc):
    if bbox is None:
        return None
    r0, r1, c0, c1 = bbox
    return [r0 + dr, r1 + dr, c0 + dc, c1 + dc]

def predict(z, action):
    out = dict(z)
    b = z.get("band")
    out["band"] = [b[0] + 1, b[1]] if b else None
    dr = dc = 0
    if action == "UP":
        dr = -5
    elif action == "DOWN":
        dr = 5
    elif action == "LEFT":
        dc = -5
    elif action == "RIGHT":
        dc = 5
    out["yellow"] = _shift(z.get("yellow"), dr, dc)
    out["teal"] = _shift(z.get("teal"), dr, dc)
    return out

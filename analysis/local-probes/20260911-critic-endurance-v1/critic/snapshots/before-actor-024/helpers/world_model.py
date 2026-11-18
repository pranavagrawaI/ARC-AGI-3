"""World model: pink band erosion + yellow/teal block sliding by 5."""
import numpy as np

def _extents(mask_rows, mask_cols):
    if not mask_rows:
        return None
    return [int(min(mask_rows)), int(max(mask_rows)), int(min(mask_cols)), int(max(mask_cols))]

def encode(grid):
    """grid: 64x64 int array. Latent: band left/right, and 12- and 9-cluster bboxes
    limited to the big central field (rows 35..55, cols 20..45) so other 9-clusters
    elsewhere do not pollute the object readout."""
    g = np.asarray(grid)
    pink_cols = [int(c) for c in range(64) if g[61, c] == 11 and g[62, c] == 11]
    band = [min(pink_cols), max(pink_cols)] if pink_cols else None
    sub = g[35:56, 20:46]
    y_r, y_c = np.where(sub == 12)
    t_r, t_c = np.where(sub == 9)
    yellow = _extents(y_r + 35, y_c + 20)
    teal = _extents(t_r + 35, t_c + 20)
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

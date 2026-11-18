"""World model: pink band erosion + yellow/teal block sliding by 5 (full-grid scan)."""
import numpy as np

def encode(grid):
    """grid: 64x64 int array. Full-grid scan for the block."""
    g = np.asarray(grid)
    pink_cols = [int(c) for c in range(64) if g[61, c] == 11 and g[62, c] == 11]
    band = [min(pink_cols), max(pink_cols)] if pink_cols else None
    # Find the 5x5 block: look for 12-cells (yellow top of block)
    # The block has exactly 10 yellow cells (2 rows x 5 cols) and 15 teal cells (3 rows x 5 cols)
    # BUT teal also appears in the goal pattern and in the bottom-left box.
    # Strategy: find the yellow 12-cells; the block is always there.
    y_r, y_c = np.where(g == 12)
    if len(y_r):
        yellow = [int(y_r.min()), int(y_r.max()), int(y_c.min()), int(y_c.max())]
    else:
        yellow = None
    # For teal: find the teal cluster that's directly below the yellow (within 1 row)
    if yellow:
        yr0, yr1, yc0, yc1 = yellow
        # Teal should be at rows yr1+1 to yr1+3, same cols
        t_mask = (g == 9) & (np.arange(64)[:, None] >= yr1 + 1) & (np.arange(64)[:, None] <= yr1 + 3) \
                 & (np.arange(64)[None, :] >= yc0) & (np.arange(64)[None, :] <= yc1)
        t_r, t_c = np.where(t_mask)
        if len(t_r):
            teal = [int(t_r.min()), int(t_r.max()), int(t_c.min()), int(t_c.max())]
        else:
            teal = None
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

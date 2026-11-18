"""World model: pink band erosion + yellow/teal block sliding."""
import numpy as np

def encode(grid):
    """grid: 64x64 int array. Returns JSON-safe dict with the moving features."""
    g = np.asarray(grid)
    pink_cols = [int(c) for c in range(64) if g[61, c] == 11 and g[62, c] == 11]
    band = [min(pink_cols), max(pink_cols)] if pink_cols else None
    y = np.where(g[45:47] == 12)
    if len(y[1]):
        obj_cols = [int(min(y[1])), int(max(y[1]))]
        obj_rows = [int(min(y[0])) + 45, int(max(y[0])) + 45]
    else:
        obj_cols = None
        obj_rows = None
    return {"band": band, "obj_cols": obj_cols, "obj_rows": obj_rows}

def predict(z, action):
    out = {"band": None, "obj_cols": z.get("obj_cols"), "obj_rows": z.get("obj_rows")}
    b = z.get("band")
    if b:
        out["band"] = [b[0] + 1, b[1]]
    oc = z.get("obj_cols")
    orow = z.get("obj_rows")
    if oc is not None:
        if action == "LEFT":
            out["obj_cols"] = [oc[0] - 5, oc[1] - 5]
        elif action == "RIGHT":
            out["obj_cols"] = [oc[0] + 5, oc[1] + 5]
        elif action == "UP":
            out["obj_rows"] = [orow[0] - 5, orow[1] - 5]
        elif action == "DOWN":
            out["obj_rows"] = [orow[0] + 5, orow[1] + 5]
    return out

"""World model: pink band erosion + yellow/teal block sliding."""
import numpy as np

def encode(grid):
    """grid: 64x64 int array. Returns JSON-safe dict with the moving features."""
    g = np.asarray(grid)
    # pink band rows 61-62
    pink_cols = [c for c in range(64) if g[61, c] == 11 and g[62, c] == 11]
    band = [int(min(pink_cols)), int(max(pink_cols))] if pink_cols else None
    # yellow/teal object in rows 44-50
    y = np.where(g[45:47] == 12)
    t = np.where(g[47:50] == 9)
    if len(y[1]):
        obj_cols = [int(min(y[1])), int(max(y[1]))]
    else:
        obj_cols = None
    obj_rows = [int(min(y[0])+45), int(max(y[0])+45)] if len(y[0]) else None
    return {"band": band, "obj_cols": obj_cols, "obj_rows": obj_rows}

def predict(z, action):
    """Apply the two local rules. band erodes 1 col from left each action.
    LEFT/RIGHT move obj_cols by 5; UP/DOWN move obj_rows by 5 (untested)."""
    out = dict(z)
    out["band"] = [b[0] + 1, b[1]] if z.get("band") else None
    oc = z.get("obj_cols")
    orow = z.get("obj_rows")
    if oc is not None:
        if action == "LEFT":
            out["obj_cols"] = [c - 5 for c in oc]
        elif action == "RIGHT":
            out["obj_cols"] = [c + 5 for c in oc]
        elif action == "UP":
            out["obj_rows"] = [r - 5 for r in orow]
        elif action == "DOWN":
            out["obj_rows"] = [r + 5 for r in orow]
    return out

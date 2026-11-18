
import numpy as np

def encode(grid):
    g = np.asarray(grid, dtype=np.int64)
    h, w = g.shape
    flat = g.flatten()
    s1 = int(np.sum(flat * 31) % (10**9))
    s2 = int(np.sum(flat**2) % (10**9))
    # row 53 fill count (color 3 in row 53)
    r53 = g[53]
    n3 = int((r53 == 3).sum())
    # rightmost unfilled (color 2) index in row 53
    idx2 = np.where(r53 == 2)[0]
    next_col = int(idx2.max()) if len(idx2) else -1
    return {
        "w": int(w), "h": int(h),
        "fp": [s1, s2],
        "n3": n3,
        "next_col": next_col,
    }

def predict(z, action):
    # Rule: SPACEBAR fills one more cell in row 53 from right to left (2 -> 3).
    nz = dict(z)
    if action == "SPACEBAR" and z.get("next_col", -1) >= 0:
        nz["n3"] = z.get("n3", 0) + 1
        nz["next_col"] = z.get("next_col") - 1
        nz["fp"] = [z["fp"][0] + 31, z["fp"][1] + 5]  # approx; will not match exactly
    return nz

import numpy as np

# Position transitions:
# 14 -> LEFT: 14, RIGHT: 25
# 25 -> LEFT: 14, RIGHT: 33
# 33 -> LEFT: 25, RIGHT: 33
_TRANS = {
    (14, "LEFT"): 14, (14, "RIGHT"): 25,
    (25, "LEFT"): 14, (25, "RIGHT"): 33,
    (33, "LEFT"): 25, (33, "RIGHT"): 33,
}

def _blob_mask(g):
    m = np.isin(g, [2, 15])
    z = np.zeros_like(m, bool)
    z[15:46, 5:50] = m[15:46, 5:50]
    return z

def encode(g):
    z = _blob_mask(g)
    if not z.any():
        return {"blob_cmin": 25, "holes": []}
    rs, cs = np.where(z)
    cmin = int(cs.min())
    row63 = g[63]
    holes = [c for c in range(64) if row63[c] == 5]
    return {"blob_cmin": cmin, "holes": holes}

def predict(z, action):
    cmin = z["blob_cmin"]
    holes = list(z["holes"])
    new_cmin = cmin
    if action in ("LEFT", "RIGHT"):
        new_cmin = _TRANS.get((cmin, action), cmin)
    # Carve min-1 when cmin changes to non-anchor
    if new_cmin != cmin and new_cmin != 25:
        if holes:
            new_hole = min(holes) - 1
            if 0 <= new_hole < 64 and new_hole not in holes:
                holes.append(new_hole)
        else:
            holes.append(63)
    return {"blob_cmin": new_cmin, "holes": sorted(holes)}

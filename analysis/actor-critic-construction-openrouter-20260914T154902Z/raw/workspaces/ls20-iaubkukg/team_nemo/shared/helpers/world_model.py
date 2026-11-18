
import numpy as np

STATIC_COLORS = set([3, 4, 5, 8, 11])

def _blob(grid, colors):
    rr, cc = np.argwhere(np.isin(grid, colors)).T
    if len(rr) == 0:
        return None
    return [int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())]

def _strip(grid):
    # 12-block is a 2-row horizontal run in rows 45-49
    m = (grid == 12)
    rr, cc = np.argwhere(m).T
    if len(rr) == 0:
        return None
    return [int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())]

def encode(grid):
    g = np.array(grid)
    z = {"w": int(g.shape[1]), "h": int(g.shape[0]),
         "walls": g.tolist(),
         "player": _blob(g, [0, 1]),
         "strip": _strip(g)}
    return z

def predict(z, action):
    import copy
    g = np.array(z["walls"])
    nz = dict(z)
    nz["walls"] = g.tolist()
    pl = z.get("player")
    st = z.get("strip")
    if pl is not None:
        r0, r1, c0, c1 = pl
        if action == "UP":
            if g[r0-1, c0] != 5 and g[r0-1, c1] != 5:
                pl = [r0-1, r1-1, c0, c1]
        elif action == "DOWN":
            if g[r1+1, c0] != 5 and g[r1+1, c1] != 5:
                pl = [r0+1, r1+1, c0, c1]
        elif action == "LEFT":
            if g[r0, c0-1] != 5 and g[r1, c0-1] != 5:
                pl = [r0, r1, c0-1, c1-1]
        elif action == "RIGHT":
            if g[r0, c1+1] != 5 and g[r1, c1+1] != 5:
                pl = [r0, r1, c0+1, c1+1]
    if st is not None and action in ("LEFT", "RIGHT"):
        r0, r1, c0, c1 = st
        d = 1 if action == "RIGHT" else -1
        if g[r0, c1+d] != 5 and g[r1, c1+d] != 5:
            st = [r0, r1, c0+d, c1+d]
    nz["player"] = pl
    nz["strip"] = st
    return nz

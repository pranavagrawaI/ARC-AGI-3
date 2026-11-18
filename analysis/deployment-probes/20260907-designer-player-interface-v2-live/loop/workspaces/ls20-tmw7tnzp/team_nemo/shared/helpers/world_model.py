import numpy as np

Z_SCHEMA = {
 "p": "list[int,int] top-left (row,col) anchor of the player sprite bbox",
 "sprite": "list[list[int,int]] sprite cells (dr,dc) relative to the anchor",
 "map_hex": "str 4096 hex digits, whole grid with sprite cells filled by PATH_FILL",
 "WALLS": "list[int] colors that block the sprite",
}
PATH_FILL = 3
WALLS = [5]
_DELTAS = {"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1)}

def _pack(g):
    return ''.join(f"{v:x}" for v in g.flatten().tolist())

def _unpack(s):
    return np.array([int(ch,16) for ch in s]).reshape(64,64)

def encode(grid):
    g = np.asarray(grid, dtype=np.int64).copy()
    mask = (g==0)|(g==1)
    rows, cols = np.where(mask)
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())
    sprite = sorted((int(r-r0), int(c-c0)) for r,c in zip(rows.tolist(), cols.tolist()))
    g[r0:r1+1, c0:c1+1][mask[r0:r1+1, c0:c1+1]] = PATH_FILL
    return {"p":[r0,c0], "sprite":sprite, "map_hex":_pack(g), "WALLS":list(WALLS)}

def predict(z, action):
    nz = dict(z); nz["p"]=list(z["p"])
    if action in _DELTAS:
        dr, dc = _DELTAS[action]
        nr, nc = z["p"][0]+dr, z["p"][1]+dc
        g = _unpack(z["map_hex"])
        walls = set(z["WALLS"])
        ok = all(0 <= nr+sr < 64 and 0 <= nc+sc < 64 and g[nr+sr, nc+sc] not in walls
                 for sr, sc in z["sprite"])
        if ok:
            nz["p"] = [nr, nc]
    return nz

def expect_class(z, action):
    return "move" if predict(z, action)["p"] != z["p"] else "no_impact"

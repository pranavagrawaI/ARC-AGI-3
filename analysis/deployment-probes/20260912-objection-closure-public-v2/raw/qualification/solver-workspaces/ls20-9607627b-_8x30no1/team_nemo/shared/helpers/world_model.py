import numpy as np

def encode(grid):
    g = np.asarray(grid, dtype=int)
    item = {"r0": 45, "c0": 29}
    for r in range(0, 64 - 5):
        for c in range(0, 64 - 5):
            if (g[r:r+2, c:c+5] == 12).all():
                item = {"r0": r, "c0": c}
                break
        else:
            continue
        break
    # walkable = 3
    walk = (g == 3).astype(int).tolist()
    return {"item": item, "walk": walk}

def predict(z, action):
    z = dict(z); z["item"] = dict(z["item"])
    r0, c0 = z["item"]["r0"], z["item"]["c0"]
    walk = z["walk"]
    moves = {"LEFT": (0, -5), "RIGHT": (0, 5), "UP": (-5, 0), "DOWN": (5, 0)}
    if action in moves:
        dr, dc = moves[action]
        nr, nc = r0 + dr, c0 + dc
        # check all 25 cells of the item fit on walkable (3) ground
        ok = True
        for i in range(5):
            for j in range(5):
                r, cc = nr + i, nc + j
                if r < 0 or r >= 64 or cc < 0 or cc >= 64 or walk[r][cc] != 1:
                    ok = False; break
            if not ok: break
        if ok:
            z["item"] = {"r0": nr, "c0": nc}
    return z

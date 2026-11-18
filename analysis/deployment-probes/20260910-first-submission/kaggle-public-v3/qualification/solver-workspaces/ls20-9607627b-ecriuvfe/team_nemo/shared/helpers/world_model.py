
import json
import numpy as np

STONE = (0, 1, 3)

def _player(g):
    m = np.isin(g, (0, 1))
    if m.sum() == 0:
        return None
    rows, cols = np.where(m)
    return (int(rows.mean()), int(cols.mean()))

def _band(g):
    m = (g == 11)
    if m.sum() == 0:
        return None
    rows, cols = np.where(m)
    return (int(rows.min()), int(cols.min()), int(m.sum()))

def _goals(g):
    goals = []
    for c in (12, 8):
        m = (g == c)
        if m.sum() > 0:
            rows, cols = np.where(m)
            goals.append({"color": int(c), "r": int(rows.mean()), "col": int(cols.mean()), "n": int(m.sum())})
    return goals

def _stone_mask(g):
    m = np.isin(g, (0, 1, 3))
    return m.astype(np.uint8).tobytes().hex()

def encode(g):
    return {
        "player": _player(g),
        "band": _band(g),
        "goals": _goals(g),
        "stone_mask": _stone_mask(g),
    }

def predict(z, action):
    z2 = dict(z)
    if action not in ("UP", "DOWN", "LEFT", "RIGHT"):
        return z2
    pm = np.frombuffer(bytes.fromhex(z["stone_mask"]), dtype=np.uint8).reshape(64, 64).copy()
    dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[action]
    cells = np.argwhere(pm > 0)
    if action == "DOWN":
        lead = cells[np.argmax(cells[:, 0])]
    elif action == "UP":
        lead = cells[np.argmin(cells[:, 0])]
    elif action == "LEFT":
        lead = cells[np.argmin(cells[:, 1])]
    elif action == "RIGHT":
        lead = cells[np.argmax(cells[:, 1])]
    lr, lc = int(lead[0]), int(lead[1])
    nr, nc = lr + dr, lc + dc
    if 0 <= nr < 64 and 0 <= nc < 64 and pm[nr, nc] == 0:
        pm2 = pm.copy()
        pm2[lr, lc] = 0
        pm2[nr, nc] = 1
        z2["stone_mask"] = pm2.tobytes().hex()
        if z.get("band"):
            br, bc, bn = z["band"]
            newn = bn - 1
            z2["band"] = (br, bc + 1, newn) if newn > 0 else None
    return z2

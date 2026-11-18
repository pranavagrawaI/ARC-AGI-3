
import numpy as np

BAND_ROWS = (61, 62)
BAND_COLS = range(13, 55)  # 42 slots, each a 2x1 column
DIRS = {"UP", "DOWN", "LEFT", "RIGHT"}

def encode(grid):
    """grid: 64x64 int array -> latent z (JSON-safe)."""
    g = np.asarray(grid)
    filled = 0
    for c in BAND_COLS:
        if g[61, c] == 3 and g[62, c] == 3:
            filled += 1
        else:
            break
    spr = np.argwhere(np.isin(g, [0, 1]))
    r, c = int(spr[:, 0].mean()), int(spr[:, 1].mean())
    return {"filled": int(filled), "sprite": [r, c],
            "Z_SCHEMA": {"filled": "count of leftmost band slots (rows61-62, cols13-54) that are 3 (converted from 11)",
                         "sprite": "centroid of 0/1 cross; static so far (H1 contradicted)"}}

def predict(z, action):
    """Provisional rule: each arrow press converts the next leftmost band slot 11->3.
    Direction-agnostic UNTIL tested (DOWN->+1 slot measured; other directions untested)."""
    z = dict(z)
    if action in DIRS and z["filled"] < 42:
        z["filled"] = z["filled"] + 1
    return z

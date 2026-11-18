"""World model v10: 5x5 assembly slides 5 cells; blocked by yellow(5) cells."""

import numpy as np

Z_SCHEMA = {
    "asm_row": "int - top row of 5x5 assembly",
    "asm_col": "int - left col of 5x5 assembly",
    "yellow": "tuple of int - yellow(5) cells as r*64+c",
    "cross_row": "int - top row of 0/1 cross (static)",
    "cross_col": "int - left col of 0/1 cross (static)",
}

STEP = 5

def encode(grid):
    g = np.asarray(grid, dtype=np.int64)
    ys, xs = np.where(g == 12)
    ar = int(ys.min()) if len(ys) > 0 else -1
    ac = int(xs.min()) if len(ys) > 0 else -1
    yys, yxs = np.where(g == 5)
    yellow = tuple(int(r) * 64 + int(c) for r, c in zip(yys, yxs))
    ys0, xs0 = np.where(np.isin(g, [0, 1]))
    cr = int(ys0.min()) if len(ys0) > 0 else -1
    cc = int(xs0.min()) if len(xs0) > 0 else -1
    return {"asm_row": ar, "asm_col": ac, "yellow": yellow, "cross_row": cr, "cross_col": cc}

_DR = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def _yellow_set(z):
    return set(z["yellow"])

def _clear(z, r, c, yset):
    """True if 5x5 area at (r,c) has no yellow cells."""
    for dr in range(5):
        for dc in range(5):
            if (r + dr) * 64 + (c + dc) in yset:
                return False
    return True

def predict(z, action):
    z2 = dict(z)
    if action in _DR:
        dr, dc = _DR[action]
        nr, nc = z["asm_row"] + dr * STEP, z["asm_col"] + dc * STEP
        yset = _yellow_set(z)
        if _clear(z, nr, nc, yset):
            z2["asm_row"] = nr
            z2["asm_col"] = nc
    return z2

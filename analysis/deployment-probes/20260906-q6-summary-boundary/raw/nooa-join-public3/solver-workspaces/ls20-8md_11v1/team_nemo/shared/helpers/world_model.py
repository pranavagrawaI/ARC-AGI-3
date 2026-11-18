"""
World model v18: c/9 block + b-strip dynamics.
No can_move in z - only positions and b-strip state.
"""
import numpy as np

B_RIGHT = 54
B_WRAP_LEFT = 13

Z_SCHEMA = {
    "c_left": "int, leftmost col of c-block (color 12)",
    "c_right": "int, rightmost col of c-block",
    "c_top": "int, topmost row of c-block (color 12)",
    "b_left": "int, leftmost col of b-strip (color 11) in row 61",
    "b_right": "int, rightmost col of b-strip",
    "level": "int",
    "unknown": "dict",
}


def encode(grid):
    g = np.asarray(grid, dtype=int)
    z = {"level": 0, "unknown": {}}
    mask12 = (g == 12)
    ys, xs = np.where(mask12)
    if len(ys) > 0:
        z["c_left"] = int(xs.min())
        z["c_right"] = int(xs.max())
        z["c_top"] = int(ys.min())
    else:
        z["c_left"] = None
        z["c_right"] = None
        z["c_top"] = None
    b_cols = np.flatnonzero(g[61, :] == 11)
    if len(b_cols) > 0:
        z["b_left"] = int(b_cols.min())
        z["b_right"] = int(b_cols.max())
    else:
        z["b_left"] = None
        z["b_right"] = None
    return z


def predict(z, action):
    zp = dict(z)
    zp["unknown"] = dict(z.get("unknown", {}))
    c_left = z.get("c_left")
    c_top = z.get("c_top")

    # Handle b-strip (always advances)
    if z.get("b_left") is not None:
        new_b_left = z["b_left"] + 1
        if new_b_left > B_RIGHT:
            zp["b_left"] = None
            zp["b_right"] = None
        else:
            zp["b_left"] = new_b_left
            zp["b_right"] = B_RIGHT
    else:
        zp["b_left"] = B_WRAP_LEFT
        zp["b_right"] = B_RIGHT

    # Handle c-block movement (always moves; blocking checked at submit time)
    if c_left is not None:
        if action == "LEFT":
            zp["c_left"] = c_left - 5
            zp["c_right"] = zp["c_left"] + 4
        elif action == "RIGHT":
            zp["c_left"] = c_left + 5
            zp["c_right"] = zp["c_left"] + 4
        elif action == "UP":
            zp["c_top"] = c_top - 5
        elif action == "DOWN":
            zp["c_top"] = c_top + 5

    return zp

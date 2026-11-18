
import numpy as np

Z_SCHEMA = {
    "room_gate": "[r, c] top-left of the 5x5 room block (gate part, color 12)",
    "bblock_left": "int, left column of the B-block (color 11)",
    "level": "int",
}

def encode(grid):
    g = np.asarray(grid)
    z = {"level": 0}

    # Room block: find the 5x5 block of colors 12 and 9
    # Gate (color 12) is always 2 rows x 5 cols
    ys12, xs12 = np.where(g == 12)
    if len(ys12) > 0:
        z["room_gate"] = [int(ys12.min()), int(xs12.min())]
    else:
        z["room_gate"] = None

    # B-block (color 11): leftmost column
    ys11, xs11 = np.where(g == 11)
    if len(ys11) > 0:
        z["bblock_left"] = int(xs11.min())
    else:
        z["bblock_left"] = None

    return z

def predict(z, action):
    z2 = dict(z)
    if z2.get("room_gate") is not None and action in ("UP", "DOWN", "LEFT", "RIGHT"):
        r, c = z2["room_gate"]
        # Room moves 5 cells per press
        if action == "UP" and r > 40:      # top boundary
            z2["room_gate"] = [r - 5, c]
        elif action == "DOWN" and r < 45:  # bottom boundary
            z2["room_gate"] = [r + 5, c]
        elif action == "LEFT" and c > 34:  # left boundary
            z2["room_gate"] = [r, c - 5]
        elif action == "RIGHT" and c < 39: # right boundary
            z2["room_gate"] = [r, c + 5]

    # B-block always shrinks by 1 from the left
    if z2.get("bblock_left") is not None:
        z2["bblock_left"] = z2["bblock_left"] + 1

    return z2

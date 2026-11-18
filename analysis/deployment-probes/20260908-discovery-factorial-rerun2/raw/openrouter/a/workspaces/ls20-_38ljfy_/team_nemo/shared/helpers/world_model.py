
import numpy as np
HEX = "0123456789abcdef"

# Provisional model (turn 1): 
#  - the 0/1 plus sprite is tracked as "player" (harness-tracked) but currently treated as STATIC (unresolved why it did not move on DOWN).
#  - a "yellow bar" (color 11) sits at the bottom; hypothesis: each direction action consumes its leftmost column (11 -> 3).
# Full grid is carried in the latent so encode(predict(...)) can be compared cell-exactly.

def encode(grid):
    g = np.asarray(grid)
    rows = ["".join(HEX[int(v)] for v in row) for row in g]
    ys, xs = np.where(np.isin(g, [0, 1]))
    player = sorted([[int(y), int(x), int(g[y, x])] for y, x in zip(ys, xs)])
    m = g == 11
    ybar = None
    if m.any():
        ys, xs = np.where(m)
        ybar = [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]
    return {"grid": rows, "player": player, "ybar": ybar}

def predict(z, action):
    g = np.array([[int(ch, 16) for ch in row] for row in z["grid"]])
    if action in ("UP", "DOWN", "LEFT", "RIGHT"):
        ybar = z["ybar"]
        if ybar is not None:
            top, bot, left, right = ybar
            if left <= right:
                g[top:bot + 1, left] = 3
    rows = ["".join(HEX[int(v)] for v in row) for row in g]
    ys, xs = np.where(np.isin(g, [0, 1]))
    player = sorted([[int(y), int(x), int(g[y, x])] for y, x in zip(ys, xs)])
    m = g == 11
    ybar = None
    if m.any():
        ys, xs = np.where(m)
        ybar = [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]
    return {"grid": rows, "player": player, "ybar": ybar}

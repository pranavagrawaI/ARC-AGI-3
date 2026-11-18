
"""World model v4: single 5x5 stack = c(12) 2 top rows + 9 3 bottom rows, cols aligned.
It translates by 5 cells per UP/DOWN/LEFT/RIGHT. Walls(3) static. b-bar depletes 1 col/action.
Z_SCHEMA:
  stack: [r, c] top-left (row of c, col of c)
"""
import numpy as np

def encode(grid):
    g = grid
    ys, xs = np.where(g == 12)
    if len(ys) == 0:
        return {"stack": None}
    # c-block top-left
    r, c = int(ys.min()), int(xs.min())
    return {"stack": [r, c]}

def predict(z, action):
    z2 = dict(z)
    if not z.get("stack"):
        return z2
    r, c = z["stack"]
    d = {"UP": (-5,0), "DOWN": (5,0), "LEFT": (0,-5), "RIGHT": (0,5)}[action]
    z2["stack"] = [r+d[0], c+d[1]]
    return z2

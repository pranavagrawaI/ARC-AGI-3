
import json

def encode(grid):
    bar_left = None
    bar_right = None
    for c in range(64):
        if grid[61, c] == 11:
            if bar_left is None:
                bar_left = c
            bar_right = c
    # find 12-block (anywhere)
    r12 = [r for r in range(64) for c in range(64) if grid[r,c]==12]
    c12 = [c for r in range(64) for c in range(64) if grid[r,c]==12]
    r9  = [r for r in range(64) for c in range(64) if grid[r,c]==9 and r > (min(r12) if r12 else 0) and r < (max(r12)+10 if r12 else 64)]
    c9  = [c for r in range(64) for c in range(64) if grid[r,c]==9 and r > (min(r12) if r12 else 0) and r < (max(r12)+10 if r12 else 64)]
    z = {
        "bar_left": bar_left if bar_left is not None else -1,
        "bar_right": bar_right if bar_right is not None else -1,
        "r12": min(r12) if r12 else -1,
        "c12": min(c12) if c12 else -1,
        "r9": min(r9) if r9 else -1,
        "c9": min(c9) if c9 else -1,
    }
    return z

def predict(z, action):
    z = dict(z)
    if z["bar_left"] >= 0:
        z["bar_left"] += 1
    if action == "UP":
        z["r12"] -= 5; z["r9"] -= 5
    elif action == "DOWN":
        z["r12"] += 5; z["r9"] += 5
    elif action == "LEFT":
        z["c12"] -= 5; z["c9"] -= 5
    elif action == "RIGHT":
        z["c12"] += 5; z["c9"] += 5
    return z

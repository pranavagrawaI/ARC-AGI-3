
def encode(grid):
    filled = sorted([int(c) for c in range(64) if grid[63, c] == 4])
    n_filled = len(filled)
    first_empty = None
    for c in range(63, -1, -1):
        if grid[63, c] != 4:
            first_empty = c
            break
    return {"filled": filled, "n_filled": n_filled, "first_empty": first_empty}

def predict(z, action):
    if z["first_empty"] is None:
        return {"filled": z["filled"], "n_filled": z["n_filled"], "first_empty": None}
    fe = z["first_empty"]
    filled = sorted(z["filled"] + [fe])
    n_filled = z["n_filled"] + 1
    ne = fe - 1
    if ne < 0:
        ne = None
    return {"filled": filled, "n_filled": n_filled, "first_empty": ne}


def _find_block(g):
    ys, xs = (g == 12).nonzero()
    if len(ys) == 0:
        return None
    return [int(ys.min()), int(xs.min())]

def encode(grid):
    pos = _find_block(grid)
    bar_left = None
    for c in range(64):
        if grid[61, c] == 11:
            bar_left = c; break
    z = {"block": pos, "bar_left": bar_left}
    return z

def predict(z, action):
    dr = dc = 0
    if action == "UP": dr = -1
    elif action == "DOWN": dr = 1
    elif action == "LEFT": dc = -1
    elif action == "RIGHT": dc = 1
    else:
        return dict(z)
    b = z.get("block")
    if b is None:
        return dict(z)
    nz = dict(z)
    nz["block"] = [b[0] + dr*5, b[1] + dc*5]
    if z.get("bar_left") is not None:
        nz["bar_left"] = z["bar_left"] + 1
    return nz

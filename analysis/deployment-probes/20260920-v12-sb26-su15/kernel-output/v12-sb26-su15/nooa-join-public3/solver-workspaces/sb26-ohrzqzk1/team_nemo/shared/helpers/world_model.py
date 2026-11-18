
def encode(grid):
    """Track the trail boundary position in row 53."""
    row53 = grid[53, :]
    n2 = int(sum(1 for v in row53 if v == 2))
    n3 = int(sum(1 for v in row53 if v == 3))
    last2 = -1
    first3 = -1
    for i in range(64):
        if row53[i] == 2:
            last2 = i
    for i in range(64):
        if row53[i] == 3:
            first3 = i
            break
    return {"n2": n2, "n3": n3, "last2": last2, "first3": first3}

def predict(z, action):
    """Predict next state given action."""
    z = dict(z)
    if action == "SPACEBAR":
        if z["n2"] > 0:
            z["n2"] -= 1
            z["n3"] += 1
            z["last2"] -= 1
            z["first3"] -= 1
    return z

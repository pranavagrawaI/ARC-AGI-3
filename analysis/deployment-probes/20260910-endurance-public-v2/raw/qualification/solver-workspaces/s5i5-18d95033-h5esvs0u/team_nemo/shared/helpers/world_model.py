
Z_SCHEMA = {"n4": "int, count of color-4 cells in row 63 (0..64)"}

def encode(grid):
    row63 = grid[63]
    n4 = int(sum(1 for c in row63 if int(c) == 4))
    return {"n4": n4, "row63": [int(c) for c in row63]}

def predict(z, action):
    z = dict(z)
    row63 = list(z["row63"])
    if action == "RESET":
        return {"n4": 0, "row63": [3]*64}
    if action.startswith("CLICK"):
        n4 = z["n4"]
        if n4 >= 64:
            return {"n4": 64, "row63": [4]*64}
        col = 63 - n4  # next column to flip (bar fills right-to-left)
        row63[col] = 4
        return {"n4": n4 + 1, "row63": row63}
    return {"n4": z["n4"], "row63": row63}

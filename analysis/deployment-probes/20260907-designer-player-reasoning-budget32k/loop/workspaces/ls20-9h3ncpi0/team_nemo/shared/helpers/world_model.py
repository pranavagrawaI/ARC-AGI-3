import numpy as np

# world_model v3 — snapshot latent: full grid rows (hex) + sprite cells.
# sprite = the 5-cell 0/1 cluster at rows31-33/cols20-22 (H1: the player).
# predict: arrow keys shift sprite by (dr,dc) iff every NEW cell is in-range
# and is floor "3" or an old sprite cell; otherwise grid unchanged.
# RESET/others: unknown -> no change (provisional).

Z_SCHEMA = {
    "grid": "list of 64 strings, 64 hex chars each, row0 top",
    "sprite": "sorted list [r,c,v] of sprite cells (colors 0/1), v 0 or 1",
}

def _rows(grid):
    if isinstance(grid, np.ndarray) or (isinstance(grid, list) and grid and isinstance(grid[0], (list, tuple, np.ndarray))):
        g = np.asarray(grid)
        return ["".join(format(int(v), "x") for v in row) for row in g]
    return [str(row) for row in grid]

def encode(grid):
    rows = _rows(grid)
    sprite = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch in ("0", "1"):
                sprite.append([r, c, int(ch)])
    return {"grid": rows, "sprite": sorted(sprite)}

_DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def predict(z, action):
    rows = list(z["grid"])
    sprite = [tuple(t) for t in z["sprite"]]
    if not sprite:
        return {"grid": rows, "sprite": []}
    if action not in _DIRS:
        return {"grid": rows, "sprite": [list(t) for t in sprite]}
    dr, dc = _DIRS[action]
    newpos = [(r + dr, c + dc) for r, c, v in sprite]
    oldset = {(r, c) for r, c, v in sprite}
    ok = True
    for nr, nc in newpos:
        if nr < 0 or nr > 63 or nc < 0 or nc > 63:
            ok = False
            break
        if (nr, nc) in oldset:
            continue
        if rows[nr][nc] != "3":
            ok = False
            break
    if not ok:
        return {"grid": rows, "sprite": [list(t) for t in sprite]}
    for r, c, v in sprite:
        rows[r] = rows[r][:c] + "3" + rows[r][c + 1:]
    for (r, c, v), (nr, nc) in zip(sprite, newpos):
        rows[nr] = rows[nr][:nc] + str(v) + rows[nr][nc + 1:]
    newsprite = sorted([nr, nc, v] for (nr, nc), (r, c, v) in zip(newpos, sprite))
    return {"grid": rows, "sprite": newsprite}

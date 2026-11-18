"""World model v2: player = rigid 5-cell plus sprite (colors 0/1), moves 1
cell per directional action; vacated cells revert to floor 3.
Floor 3; goal pad 12 (walkable; contact = win, untested); items 9 (walkable?
untested, avoided by plan); 4 = wall; 5/8/11 = chrome/HUD."""

Z_SCHEMA = {
    "rows":   "list[str] -- 64x64 grid as hex strings (authoritative frame)",
    "player": "list[[r,c]] -- the 5 cells of the plus sprite",
    "center": "[r,c] -- the sprite hub cell",
    "goal":   "list[[r,c]] -- color-12 cells (goal pad)",
    "items":  "list[[r,c]] -- color-9 cells",
}

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
FLOOR = 3
WALL = 4
GOALC = 12
ITEMC = 9
WALKABLE = (3, 12)

def _cells(grid, colors):
    out = []
    for r in range(64):
        row = grid[r]
        for c in range(64):
            if int(row[c], 16) in colors:
                out.append([r, c])
    return out

def _find_center(player, rows):
    blob = set(map(tuple, player))
    best, bestn = None, -1
    for (r, c) in blob:
        if int(rows[r][c], 16) != 0:
            continue
        n = sum(1 for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)) if (r+dr, c+dc) in blob)
        if n > bestn:
            best, bestn = [r, c], n
    return best if best is not None else player[0]

def encode(grid):
    player = _cells(grid, (0, 1))
    return {
        "rows": list(grid),
        "player": player,
        "center": _find_center(player, grid),
        "goal":   _cells(grid, (12,)),
        "items":  _cells(grid, (9,)),
    }

def predict(z, action):
    if action not in DIRS:
        return z
    dr, dc = DIRS[action]
    rows = z["rows"]
    blob = set(map(tuple, z["player"]))
    new = set()
    ok = True
    for (r, c) in sorted(blob):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 64 and 0 <= nc < 64):
            ok = False
            break
        new.add((nr, nc))
        if (nr, nc) not in blob and int(rows[nr][nc], 16) not in WALKABLE:
            ok = False
            break
    if not ok:
        return z
    g = [list(r) for r in rows]
    colors = {(r, c): rows[r][c] for (r, c) in blob}
    for (r, c) in blob:
        g[r][c] = "3"
    for (r, c) in new:
        g[r][c] = colors[(r - dr, c - dc)]
    nrows = ["".join(r) for r in g]
    return {"rows": nrows, "player": sorted(new),
            "center": _find_center(sorted(new), nrows),
            "goal": z["goal"], "items": z["items"]}

def classify(z0, z1, action):
    if z1 == z0:
        return "no_impact"
    if any((r, c) in set(map(tuple, z1["player"])) for (r, c) in z0["goal"]):
        return "level_up"
    return "move"

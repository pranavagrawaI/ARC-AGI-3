
Z_SCHEMA = {
    "player_pos": "list[int] - [row, col] top-left of the 5x5 9/12 block",
    "walker11": "list[int] - [left_col, right_col] of the 11-strip (rows 61-62)",
    "bg": "dict[str, int] - color_str -> count of cells (keys are stringified ints)",
    "levels_completed": "int",
}

def encode(grid):
    import numpy as np
    from collections import Counter
    g = np.array(grid) if not isinstance(grid, np.ndarray) else grid
    c12 = [(r,c) for r in range(64) for c in range(64) if g[r,c]==12]
    if c12:
        r12 = sorted(set(r for r,c in c12))
        c12c = sorted(set(c for r,c in c12))
        player_pos = [r12[0], c12c[0]]
    else:
        player_pos = [None, None]
    c11 = [(r,c) for r in range(64) for c in range(64) if g[r,c]==11]
    if c11:
        cols = sorted(set(c for r,c in c11))
        walker = [cols[0], cols[-1]]
    else:
        walker = [None, None]
    counts = Counter(int(v) for row in g for v in row)
    return {
        "player_pos": player_pos,
        "walker11": walker,
        "bg": {str(k): v for k, v in counts.items()},
        "levels_completed": 0,
    }

def predict(z, action):
    import copy
    z2 = copy.deepcopy(z)
    # Walker retraction: left edge moves right by 1 col, losing 2 cells (rows 61-62)
    if z2["walker11"][0] is not None and z2["walker11"][0] < z2["walker11"][1]:
        z2["walker11"][0] += 1
        if z2["walker11"][0] <= z2["walker11"][1]:
            bg = dict(z2["bg"])
            bg["11"] = bg.get("11", 0) - 2
            bg["3"] = bg.get("3", 0) + 2
            z2["bg"] = bg
        else:
            z2["walker11"] = [None, None]
            bg = dict(z2["bg"])
            bg["11"] = 0
            z2["bg"] = bg
    # Player movement
    if z2["player_pos"][0] is not None:
        if action == "UP":
            z2["player_pos"][0] -= 5
        elif action == "DOWN":
            z2["player_pos"][0] += 5
        elif action == "LEFT":
            z2["player_pos"][1] -= 5
        elif action == "RIGHT":
            z2["player_pos"][1] += 5
    return z2

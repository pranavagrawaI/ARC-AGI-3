
import numpy as np

Z_SCHEMA = {
    "player_row": int,
    "player_col": int,
    "bar_left": int,
    "grid": list  # full grid snapshot
}

def encode(grid):
    grid = np.array(grid, dtype=np.int64)
    c_mask = (grid == 12)
    c_rows = np.where(c_mask.any(axis=1))[0]
    c_cols = np.where(c_mask.any(axis=0))[0]

    b_mask = (grid == 11)
    b_cols = np.where(b_mask.any(axis=0))[0]

    z = {
        "player_row": int(c_rows[0]),
        "player_col": int(c_cols[0]),
        "bar_left": int(b_cols[0]) if len(b_cols) > 0 else None,
        "grid": grid.tolist()
    }
    return z

def predict(z, action):
    grid = np.array(z["grid"], dtype=np.int64)
    pr = z["player_row"]
    pc = z["player_col"]

    if action not in ("UP", "DOWN", "LEFT", "RIGHT"):
        return z

    dr, dc = 0, 0
    if action == "UP": dr, dc = -5, 0
    elif action == "DOWN": dr, dc = 5, 0
    elif action == "LEFT": dr, dc = 0, -5
    elif action == "RIGHT": dr, dc = 0, 5

    new_pr = pr + dr
    new_pc = pc + dc

    # Check bounds
    if new_pr < 0 or new_pr + 1 >= 64 or new_pc < 0 or new_pc + 4 >= 64:
        return z

    # Find 9-cells that are "attached" (in player's column range, within 5 rows below)
    player_bottom = pr + 1
    attached_9 = []
    for r in range(player_bottom + 1, min(player_bottom + 6, 64)):
        for c in range(pc, pc + 5):
            if grid[r, c] == 9:
                attached_9.append((r, c))

    # Erase old player and attached 9
    for r in range(pr, pr + 2):
        for c in range(pc, pc + 5):
            grid[r, c] = 3
    for (r, c) in attached_9:
        grid[r, c] = 3

    # Write attached 9 at new position (same relative offset)
    for (r, c) in attached_9:
        row_off = r - (player_bottom + 1)  # 0, 1, 2, ...
        col_off = c - pc                    # 0..4
        grid[new_pr + 2 + row_off, new_pc + col_off] = 9

    # Write new player
    for r in range(new_pr, new_pr + 2):
        for c in range(new_pc, new_pc + 5):
            grid[r, c] = 12

    # Shrink b-bar from left
    if z["bar_left"] is not None:
        b_mask = (grid == 11)
        b_rows = np.where(b_mask.any(axis=1))[0]
        for r in b_rows:
            grid[r, z["bar_left"]] = 3
        new_bar_left = z["bar_left"] + 1
    else:
        new_bar_left = None

    return {
        "player_row": new_pr,
        "player_col": new_pc,
        "bar_left": new_bar_left,
        "grid": grid.tolist()
    }

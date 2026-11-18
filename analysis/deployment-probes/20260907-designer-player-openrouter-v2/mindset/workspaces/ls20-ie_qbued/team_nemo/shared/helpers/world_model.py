
"""World model for the 64x64 maze game.

Player: a 3-cell cross of 0s on a field of 3s, with its center cell colored 1.
Movement is one cell per direction; walls block; stepping on a 9-colored object
picks it up (it disappears from the grid and the color is added to the held
set). The level completes when all 9-objects have been collected.
"""
import numpy as np

Z_SCHEMA = {
    "player": "(row, col) of the center cell (color 1)",
    "carrying": "set of hex colors picked up (as sorted list)",
    "level": "int, current level index",
}

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def encode(grid):
    """grid: 64x64 np.ndarray -> z dict."""
    grid = np.asarray(grid)
    center = np.argwhere(grid == 1)
    r, c = int(center[0][0]), int(center[0][1])
    return {"player": (r, c), "carrying": [], "level": 0}


def _player_cells(z):
    r, c = z["player"]
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def predict(z, action):
    """z, action in {UP,DOWN,LEFT,RIGHT} -> z' (player position)."""
    r, c = z["player"]
    dr, dc = DIRS[action]
    nr, nc = r + dr, c + dc
    return {"player": (nr, nc), "carrying": list(z["carrying"]), "level": z["level"]}


def player_free(grid, r, c):
    """True if the player's 5 cells (center (r,c)) fit in non-wall cells.
    Walls are 5 (border) and 3 is the floor the player rides on."""
    grid = np.asarray(grid)
    ok = True
    for rr, cc in _player_cells({"player": (r, c)}):
        if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            ok = False; break
        v = grid[rr, cc]
        if v == 5:  # wall
            ok = False; break
        # 4 is outer floor; 3 is maze floor; 9/12 objects may be standable?
    return ok


def would_move(grid, z, action):
    """True if moving the player to z' puts all 5 cells on walkable (non-5)
    cells inside the grid."""
    np_ = predict(z, action)["player"]
    return player_free(grid, *np_)

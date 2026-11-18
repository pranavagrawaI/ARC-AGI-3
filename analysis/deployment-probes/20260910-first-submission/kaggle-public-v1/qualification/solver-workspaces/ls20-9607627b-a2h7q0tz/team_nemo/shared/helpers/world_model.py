
import numpy as np

Z_SCHEMA = {
    "cube_row": "int, top row of 5x5 cube (range 8-45)",
    "cube_col": "int, left col of 5x5 cube (range 4-49)",
    "hud_cells": "int, number of 11-colored cells in HUD strip (range 0-80)",
}

def encode(grid):
    """Extract the latent state from the grid."""
    rows12, cols12 = np.where(grid == 12)
    if len(rows12) == 0:
        cube_row = -1
        cube_col = -1
    else:
        cube_row = int(rows12.min())
        cube_col = int(cols12.min())

    hud_region = grid[60:64, :]
    hud_cells = int((hud_region == 11).sum())

    return {
        "cube_row": cube_row,
        "cube_col": cube_col,
        "hud_cells": hud_cells,
    }

def predict(z, action):
    """Predict the next state given current state and action."""
    z2 = dict(z)
    step = 5

    if action == "UP":
        z2["cube_row"] = z["cube_row"] - step
    elif action == "DOWN":
        z2["cube_row"] = z["cube_row"] + step
    elif action == "LEFT":
        z2["cube_col"] = z["cube_col"] - step
    elif action == "RIGHT":
        z2["cube_col"] = z["cube_col"] + step
    else:
        pass

    # HUD loses 2 cells on each move
    z2["hud_cells"] = max(0, z["hud_cells"] - 2)

    return z2

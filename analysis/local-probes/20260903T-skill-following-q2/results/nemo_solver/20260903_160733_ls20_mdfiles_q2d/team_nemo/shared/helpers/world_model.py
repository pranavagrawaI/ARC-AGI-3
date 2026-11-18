
import numpy as np

"""World model for game-820258."""

Z_SCHEMA = {
    "stack_r": ("int", "top row of 12 blob"),
    "stack_c": ("int", "left col of 12 blob"),
    "bar_cells": ("int", "remaining cells in 11 bar"),
    "bar_left": ("int", "leftmost col of 11 bar"),
}

STEP = 5

def encode(grid):
    """Extract latent state from a 64x64 numpy grid."""
    z = {}
    rows12, cols12 = np.where(grid == 12)
    if len(rows12):
        z["stack_r"] = int(rows12.min())
        z["stack_c"] = int(cols12.min())
    else:
        z["stack_r"] = None
        z["stack_c"] = None
    rows11, cols11 = np.where(grid == 11)
    if len(rows11):
        z["bar_cells"] = int(len(rows11))
        z["bar_left"] = int(cols11.min())
    else:
        z["bar_cells"] = 0
        z["bar_left"] = None
    return z

def predict(z, action):
    """Predict next latent state after an action."""
    z2 = dict(z)
    if z.get("stack_r") is None:
        return z2
    dr, dc = 0, 0
    if action == "UP":    dr, dc = -STEP, 0
    elif action == "DOWN": dr, dc = STEP, 0
    elif action == "LEFT": dr, dc = 0, -STEP
    elif action == "RIGHT":dr, dc = 0, STEP
    elif action == "RESET":
        return {"stack_r": 45, "stack_c": 34, "bar_cells": 84, "bar_left": 13}
    else:
        return z2
    z2["stack_r"] = z["stack_r"] + dr
    z2["stack_c"] = z["stack_c"] + dc
    z2["bar_cells"] = max(0, z["bar_cells"] - 2)
    if z.get("bar_left") is not None:
        z2["bar_left"] = z["bar_left"] + 1
    return z2

def retrodict(actual_grid, predicted_z):
    """Compare predicted latent to actual grid."""
    actual_z = encode(actual_grid)
    notes = []
    for k in Z_SCHEMA:
        if k in actual_z and k in predicted_z:
            if actual_z[k] != predicted_z[k]:
                notes.append(f"{k}: predicted {predicted_z[k]}, actual {actual_z[k]}")
    match = len(notes) == 0
    return actual_z, match, notes

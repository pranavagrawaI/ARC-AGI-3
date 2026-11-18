
import json
import numpy as np

def encode(grid):
    """Encode the grid into a compact JSON-safe representation."""
    grid = np.array(grid)
    # Find all non-background (non-5) cells
    # Background is color 5 (gray)
    cells = {}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            v = int(grid[r, c])
            if v != 5:
                cells[f"{r},{c}"] = v
    return {"cells": cells, "shape": [int(grid.shape[0]), int(grid.shape[1])]}

def predict(z, action):
    """Predict the next state. For now, just return a placeholder.
    We need to discover the actual game mechanics first."""
    # Placeholder: return same state (no change)
    # This will be wrong but establishes the API
    return dict(z)

def decode(z):
    """Decode latent back to grid for debugging."""
    grid = np.full((64, 64), 5, dtype=int)
    for key, val in z["cells"].items():
        r, c = map(int, key.split(","))
        grid[r, c] = val
    return grid

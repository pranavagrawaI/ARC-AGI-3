
import numpy as np

Z_SCHEMA = {
    "player_row": "int, 0-63",
    "player_col": "int, 0-63",
    "player_color": "int",
    "level": "int",
    "grid_hash": "str, md5 of grid",
}

def encode(grid):
    """Encode grid to latent state z."""
    return {
        "grid": grid,
        "grid_hash": str(grid.tobytes())[:16],
    }

def predict(z, action):
    """Predict next state. Returns z' or None if unknown."""
    return None  # Unknown yet

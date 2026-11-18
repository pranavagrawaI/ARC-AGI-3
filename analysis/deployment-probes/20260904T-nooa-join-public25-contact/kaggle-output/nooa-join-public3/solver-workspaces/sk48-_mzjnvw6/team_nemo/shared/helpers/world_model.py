
import numpy as np

Z_SCHEMA = {
    "player_x": "int, 0-63, center col of player square",
    "player_y": "int, 0-63, center row of player square",
    "level": "int, current level",
}

def encode(grid_rows):
    """Parse grid_rows (64 hex strings) into latent state z (dict)."""
    g = _parse_grid(grid_rows)
    # Player: 6-frame square (border color 6, interior 0)
    # Find the 6-colored cells to locate player
    ys, xs = np.nonzero(g == 6)
    if len(ys) == 0:
        return {"player_x": None, "player_y": None, "level": 0}
    px = int(xs.mean())
    py = int(ys.mean())
    return {"player_x": px, "player_y": py, "level": 0}

def predict(z, action):
    """Predict next latent state given action. Player moves by 1 cell in direction."""
    z2 = dict(z)
    dx, dy = 0, 0
    if action == "UP": dy = -1
    elif action == "DOWN": dy = 1
    elif action == "LEFT": dx = -1
    elif action == "RIGHT": dx = 1
    else:
        return z2
    if z["player_x"] is not None:
        z2["player_x"] = z["player_x"] + dx
        z2["player_y"] = z["player_y"] + dy
    return z2

def _parse_grid(grid_rows):
    return np.array([[int(c, 16) for c in row] for row in grid_rows], dtype=np.uint8)

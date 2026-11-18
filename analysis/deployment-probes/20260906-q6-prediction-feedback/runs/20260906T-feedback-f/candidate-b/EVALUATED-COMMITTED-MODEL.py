
import numpy as np

Z_SCHEMA = {
    "agent_top_row": "int, top row of agent (2-row-tall, 5-col-wide color-12 block)",
    "agent_col": "int, left col of agent",
    "band_left": "int, leftmost col of 11-band (increases by 1 per step)",
    "band_right": "int, rightmost col of 11-band (constant 54)",
    "step": "int, total steps taken this level",
}

def encode(grid):
    g = grid
    m12 = (g == 12)
    if m12.sum() == 0:
        return None
    rs, cs = np.where(m12)
    top = int(rs.min())
    col = int(cs.min())
    m11 = (g == 11)
    if m11.sum() > 0:
        band_left = int(np.where(m11)[1].min())
        band_right = int(np.where(m11)[1].max())
    else:
        band_left = 13
        band_right = 54
    return {"agent_top_row": top, "agent_col": col,
            "band_left": band_left, "band_right": band_right,
            "agent_h": 2, "agent_w": 5}

def predict(z, action):
    """Predict next latent state. Wall-aware: agent doesn't move if blocked."""
    z2 = dict(z)
    z2["step"] = z.get("step", 0) + 1
    # Band: left edge moves right by 1 each step, right edge constant
    z2["band_left"] = z["band_left"] + 1
    z2["band_right"] = z["band_right"]  # stays at 54

    # Agent movement: 5 cells in direction, but blocked by walls (4, 5)
    # Need the grid to check walls, so we use a separate function
    top, col = z["agent_top_row"], z["agent_col"]
    h, w = z["agent_h"], z["agent_w"]

    if action == "UP":
        z2["agent_top_row"] = top - 5
    elif action == "DOWN":
        z2["agent_top_row"] = top + 5
    elif action == "LEFT":
        z2["agent_col"] = col - 5
    elif action == "RIGHT":
        z2["agent_col"] = col + 5
    return z2

def predict_with_walls(z, action, grid):
    """Wall-aware prediction: agent stays put if target area has walls (color 4 or 5)."""
    z2 = dict(z)
    z2["step"] = z.get("step", 0) + 1
    z2["band_left"] = z["band_left"] + 1
    z2["band_right"] = z["band_right"]

    top, col = z["agent_top_row"], z["agent_col"]
    h, w = z["agent_h"], z["agent_w"]

    if action == "UP":
        new_top, new_col = top - 5, col
    elif action == "DOWN":
        new_top, new_col = top + 5, col
    elif action == "LEFT":
        new_top, new_col = top, col - 5
    elif action == "RIGHT":
        new_top, new_col = top, col + 5
    else:
        z2["agent_top_row"] = top
        z2["agent_col"] = col
        return z2

    # Check if target area (h rows x w cols at new position) has walls
    if 0 <= new_top and new_top + h <= 64 and 0 <= new_col and new_col + w <= 64:
        region = grid[new_top:new_top+h, new_col:new_col+w]
        blocked = np.any((region == 4) | (region == 5))
    else:
        blocked = True

    if blocked:
        z2["agent_top_row"] = top
        z2["agent_col"] = col
    else:
        z2["agent_top_row"] = new_top
        z2["agent_col"] = new_col
    return z2

def wall_check(grid, new_top, new_col, w=5, h=2):
    """Check if agent at (new_top, new_col) overlaps any wall (4 or 5)."""
    if new_top < 0 or new_top + h > 64 or new_col < 0 or new_col + w > 64:
        return True
    region = grid[new_top:new_top+h, new_col:new_col+w]
    return bool(np.any((region == 4) | (region == 5)))

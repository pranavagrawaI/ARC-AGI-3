
import numpy as np

Z_SCHEMA = {
    "agent_top_row": "int, top row of agent (12-colored top block)",
    "agent_col": "int, left col of agent",
    "band_right": "int, rightmost col of 11-band (starts at 54, decreases by 2 per step)",
    "steps": "int, total steps taken this level",
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
        band_right = int(np.where(m11)[1].max())
    else:
        band_right = 12
    return {"agent_top_row": top, "agent_col": col, "band_right": band_right,
            "agent_w": 5, "agent_h": 5}

def predict(z, action):
    z2 = dict(z)
    step = 5
    if action == "UP":
        z2["agent_top_row"] = z["agent_top_row"] - step
    elif action == "DOWN":
        z2["agent_top_row"] = z["agent_top_row"] + step
    elif action == "LEFT":
        z2["agent_col"] = z["agent_col"] - step
    elif action == "RIGHT":
        z2["agent_col"] = z["agent_col"] + step
    z2["band_right"] = max(12, z["band_right"] - 2)
    return z2

def wall_check(grid, new_top, new_col, w=5, h=5):
    """Check if agent at (new_top, new_col) overlaps any 5-wall or 4-wall."""
    g = grid
    region = g[new_top:new_top+h, new_col:new_col+w]
    return np.any(region == 5) or np.any(region == 4)


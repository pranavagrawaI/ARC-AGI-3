
import numpy as np

Z_SCHEMA = {
    "agent_top_row": "int, top row of agent (12-colored top block)",
    "agent_col": "int, left col of agent",
    "band_left": "int, leftmost col of 11-band (starts at 13, increases by 1 per step)",
    "band_right": "int, rightmost col of 11-band (starts at 54, constant)",
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
        band_left = int(np.where(m11)[1].min())
        band_right = int(np.where(m11)[1].max())
    else:
        band_left = 13
        band_right = 54
    return {"agent_top_row": top, "agent_col": col,
            "band_left": band_left, "band_right": band_right,
            "agent_w": 5, "agent_h": 5}

def _wall_blocked(grid, new_top, new_col, w=5, h=5):
    """Check if agent at (new_top, new_col) overlaps any 4-wall or 5-wall."""
    if new_top < 0 or new_top + h > 64 or new_col < 0 or new_col + w > 64:
        return True
    region = grid[new_top:new_top+h, new_col:new_col+w]
    return bool(np.any(region == 5) or np.any(region == 4))

def predict(z, action, grid=None):
    """Predict next latent state.

    If grid is provided, wall checks are exact.
    If grid is None, movement is assumed unblocked (optimistic).
    """
    z2 = dict(z)
    step = 5
    new_top = z["agent_top_row"]
    new_col = z["agent_col"]
    if action == "UP":
        new_top -= step
    elif action == "DOWN":
        new_top += step
    elif action == "LEFT":
        new_col -= step
    elif action == "RIGHT":
        new_col += step
    elif action == "RESET":
        return {"agent_top_row": 45, "agent_col": 34,
                "band_left": 13, "band_right": 54,
                "agent_w": 5, "agent_h": 5}

    # Wall check
    if grid is not None:
        blocked = _wall_blocked(grid, new_top, new_col)
    else:
        blocked = False
    if not blocked:
        z2["agent_top_row"] = new_top
        z2["agent_col"] = new_col
    # else: agent stays in place (blocked)

    # Band: left edge advances by 1 col per step, right edge stays
    z2["band_left"] = min(z["band_left"] + 1, z["band_right"])
    z2["band_right"] = z["band_right"]
    return z2

def wall_check(grid, new_top, new_col, w=5, h=5):
    """Check if agent at (new_top, new_col) overlaps any 5-wall or 4-wall."""
    return _wall_blocked(grid, new_top, new_col, w, h)

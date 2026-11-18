import numpy as np

Z_SCHEMA = {
    "agent_top_row": "int, top row of agent (12-colored block)",
    "agent_col": "int, left col of agent",
    "band_right": "int, rightmost col of 11-band (fixed at 54; left edge sweeps right +1 per action)",
    "blocked_up": "bool, True if moving UP from current pos hits 4/5 wall",
    "blocked_down": "bool, True if moving DOWN from current pos hits 4/5 wall",
    "blocked_left": "bool, True if moving LEFT from current pos hits 4/5 wall",
    "blocked_right": "bool, True if moving RIGHT from current pos hits 4/5 wall",
}

def _wall_at(g, top, col, w, h):
    region = g[top:top+h, col:col+w]
    return bool(np.any(region == 5) or np.any(region == 4))

def encode(grid):
    g = grid
    m12 = (g == 12)
    if m12.sum() == 0:
        return None
    rs, cs = np.where(m12)
    top = int(rs.min())
    col = int(cs.min())
    m11 = (g == 11)
    band_right = int(np.where(m11)[1].max()) if m11.sum() > 0 else 12
    w, h = 5, 2
    return {
        "agent_top_row": top,
        "agent_col": col,
        "band_right": band_right,
        "blocked_up": _wall_at(g, top - 5, col, w, h),
        "blocked_down": _wall_at(g, top + 5, col, w, h),
        "blocked_left": _wall_at(g, top, col - 5, w, h),
        "blocked_right": _wall_at(g, top, col + 5, w, h),
    }

def predict(z, action):
    z2 = dict(z)
    step = 5
    if action == "UP":
        if z.get("blocked_up", False):
            pass  # stays
        else:
            z2["agent_top_row"] = z["agent_top_row"] - step
    elif action == "DOWN":
        if z.get("blocked_down", False):
            pass
        else:
            z2["agent_top_row"] = z["agent_top_row"] + step
    elif action == "LEFT":
        if z.get("blocked_left", False):
            pass
        else:
            z2["agent_col"] = z["agent_col"] - step
    elif action == "RIGHT":
        if z.get("blocked_right", False):
            pass
        else:
            z2["agent_col"] = z["agent_col"] + step
    # band right edge fixed at 54
    z2["band_right"] = z["band_right"]
    return z2

def wall_check(grid, new_top, new_col, w=5, h=2):
    """Check if agent at (new_top, new_col) overlaps any 5-wall or 4-wall."""
    g = grid
    region = g[new_top:new_top+h, new_col:new_col+w]
    return np.any(region == 5) or np.any(region == 4)

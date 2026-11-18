
import numpy as np

Z_SCHEMA = {
    "agent_top_row": "int, top row of agent (12-colored top block)",
    "agent_col": "int, left col of agent",
    "band_right": "int, rightmost col of 11-band (rows 61-62, starts 54)",
    "band_left": "int, leftmost col of 11-band (rows 61-62, starts 18)",
}

def encode(grid):
    """Measure the latent state from a 64x64 int grid.

    agent  = 12-colored block: top row / left col (width 5, height 5)
    band   = rows 61-62, color 11, cols band_left..band_right.
    """
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
        band_left = int(np.where(m11)[1].min())
    else:
        band_right = 12
        band_left = 18
    return {"agent_top_row": top, "agent_col": col,
            "band_right": band_right, "band_left": band_left,
            "agent_w": 5, "agent_h": 5}

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def predict(z, action):
    """Agent moves 5 cells in the commanded direction (clamped to bounds and
    walls). The band (rows 61-62) does NOT move on its own: it was observed
    static through 5 actions, and the single 2-cell change at the left edge
    (cols 18-19, 11->3) occurred on the one step the agent was blocked
    (a failed LEFT), consistent with a band-edge shift when movement is
    obstructed. Unresolved: does the shift also happen on successful
    moves? Test with a fresh action from this state."""
    z2 = dict(z)
    w, h = z.get("agent_w", 5), z.get("agent_h", 5)
    if action == "UP":
        z2["agent_top_row"] = _clamp(z["agent_top_row"] - 5, 0, 64 - h)
    elif action == "DOWN":
        z2["agent_top_row"] = _clamp(z["agent_top_row"] + 5, 0, 64 - h)
    elif action == "LEFT":
        z2["agent_col"] = _clamp(z["agent_col"] - 5, 0, 64 - w)
    elif action == "RIGHT":
        z2["agent_col"] = _clamp(z["agent_col"] + 5, 0, 64 - w)
    return z2

def wall_check(grid, new_top, new_col, w=5, h=5):
    """Check if agent at (new_top, new_col) overlaps any 5-wall or 4-wall."""
    g = grid
    region = g[new_top:new_top+h, new_col:new_col+w]
    return np.any(region == 5) or np.any(region == 4)

import numpy as np

Z_SCHEMA = {
    "agent_top_row": "int, top row of the 2-row x 5-col agent block (color 12); moves by 5 per successful step",
    "agent_col": "int, left col of the agent block",
    "band_right": "int, rightmost col of the 11-band (static, starts 54)",
    "band_left": "int, leftmost col of the 11-band (erodes +1 per step)",
    "n11": "int, cell count of the 11-band (erodes by 2 per step: 1 col of rows 61 and 62)",
    "agent_w": "int, constant 5",
    "agent_h": "int, constant 2",
}

def encode(grid):
    """grid: 64x64 int array (from self.grid_array). Returns latent dict."""
    g = np.asarray(grid)
    m12 = (g == 12)
    if m12.sum() == 0:
        return None
    rs, cs = np.where(m12)
    top = int(rs.min())
    col = int(cs.min())
    m11 = (g == 11)
    if m11.sum() > 0:
        bcols = np.where(m11)[1]
        band_right = int(bcols.max())
        band_left = int(bcols.min())
        n11 = int(m11.sum())
    else:
        band_right, band_left, n11 = 12, 12, 0
    return {"agent_top_row": top, "agent_col": col,
            "band_right": band_right, "band_left": band_left, "n11": n11,
            "agent_w": 5, "agent_h": 2}

def predict(z, action):
    """
    Latent-space dynamics (all 6 observed transitions consistent).
    Rules:
    - agent jumps 5 cells in the action direction; if the destination footprint
      (2x5) overlaps a 5, the agent does NOT move (caller signals via z['blocked']).
    - the 11-band erodes one column at its LEFT edge (2 cells, rows 61-62) on
      EVERY step, blocked or not. band_right stays static.
    """
    z2 = dict(z)
    blocked = bool(z.get("blocked", False))
    if not blocked:
        if action == "UP":
            z2["agent_top_row"] = z["agent_top_row"] - 5
        elif action == "DOWN":
            z2["agent_top_row"] = z["agent_top_row"] + 5
        elif action == "LEFT":
            z2["agent_col"] = z["agent_col"] - 5
        elif action == "RIGHT":
            z2["agent_col"] = z["agent_col"] + 5
    z2["n11"] = z["n11"] - 2
    if z["n11"] > 0:
        z2["band_left"] = z["band_left"] + 1
    z2["blocked"] = False
    return z2

def landing(grid, z, action, w=5, h=2):
    """Footprint top-left (row, col) after a successful move, or None if blocked by a 5."""
    r, c = z["agent_top_row"], z["agent_col"]
    if action == "UP":
        nr, nc = r - 5, c
    elif action == "DOWN":
        nr, nc = r + 5, c
    elif action == "LEFT":
        nr, nc = r, c - 5
    elif action == "RIGHT":
        nr, nc = r, c + 5
    else:
        return None
    if nr < 0 or nc < 0 or nr + h > 64 or nc + w > 64:
        return None
    if np.any(grid[nr:nr+h, nc:nc+w] == 5):
        return None
    return (nr, nc)

def plan(grid, z, actions):
    """Simulate a batch from the current grid; returns (final_z, per_step_labels)."""
    z2 = dict(z)
    labels = []
    for a in actions:
        lr = landing(grid, z2, a)
        if lr is None:
            labels.append("blocked")
        else:
            z2["agent_top_row"], z2["agent_col"] = lr
            labels.append("move")
        z2["n11"] -= 2
        if z2["n11"] > 0:
            z2["band_left"] = z2["band_left"] + 1
    return z2, labels

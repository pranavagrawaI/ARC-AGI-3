"""World model for the floor-counter game."""
import numpy as np

# The latent state is just: how many 4-cells are on floor row 63
# and which columns they occupy (they fill leftward from col 63)

Z_SCHEMA = {
    "counter": "int, 0-64, number of 4-cells on floor row 63",
    "leftmost_col": "int, 0-63, leftmost col with a 4 on floor (63-counter+1..63)",
    "structures_frozen": "bool, all non-floor structures are static",
}

def encode(grid_rows):
    """Encode 64x64 grid to latent z (dict)."""
    g = np.array([[int(c, 16) for c in row] for row in grid_rows])
    floor = g[63, :]
    counter = int(np.sum(floor == 4))
    if counter > 0:
        fours = np.where(floor == 4)[0]
        leftmost_col = int(fours.min())
        rightmost_col = int(fours.max())
    else:
        leftmost_col = 64  # sentinel
        rightmost_col = 63
    # Verify the 4s form a contiguous block from right
    if counter > 0:
        expected = list(range(64 - counter, 64))
        is_contiguous = all(floor[c] == 4 for c in expected) and all(floor[c] != 4 for c in range(0, 64 - counter))
    else:
        is_contiguous = True
    return {
        "counter": counter,
        "leftmost_col": leftmost_col,
        "rightmost_col": rightmost_col,
        "is_contiguous": is_contiguous,
        "structures_frozen": True,  # all structures unchanged so far
    }

def predict(z, action):
    """Predict next latent state after action."""
    z2 = dict(z)
    if action == "RESET":
        z2["counter"] = 0
        z2["leftmost_col"] = 64
        z2["rightmost_col"] = 63
    elif action.startswith("CLICK"):
        if z["counter"] < 64:
            z2["counter"] = z["counter"] + 1
            z2["leftmost_col"] = 64 - z2["counter"]
            z2["rightmost_col"] = 63
    return z2

def action_effect(z, action):
    """Return expected effect class: 'move', 'no_impact', 'level_up', 'terminal'."""
    z2 = predict(z, action)
    if z2["counter"] != z["counter"]:
        if z2["counter"] == 64:
            return "level_up"  # or terminal
        return "move"
    return "no_impact"

def plan_to_win(z, max_steps=10):
    """Return a list of CLICK actions to fill the counter."""
    n_needed = 64 - z["counter"]
    n = min(n_needed, max_steps)
    return [f"CLICK 32 32" for _ in range(n)]

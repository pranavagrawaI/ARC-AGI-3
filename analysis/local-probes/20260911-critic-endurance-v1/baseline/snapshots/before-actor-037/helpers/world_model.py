
import numpy as np

def encode(grid):
    """Encode grid into a JSON-safe dict."""
    g = grid.astype(int)
    char_cells = [(int(r), int(c), int(g[r, c])) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] in (0, 1)]
    nine_cells = [(int(r), int(c)) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == 9]
    c_cells = [(int(r), int(c)) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == 12]
    b_cells = [(int(r), int(c)) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == 11]
    eight_cells = [(int(r), int(c)) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == 8]
    return {
        "char": char_cells,
        "nines": nine_cells,
        "c_blocks": c_cells,
        "b_bar": b_cells,
        "eights": eight_cells,
    }

def predict(z, action):
    """Predict next state. The c-block and 9-block seem to move UP by 5 each step.
    Character movement is unclear."""
    import json
    z2 = json.loads(json.dumps(z))
    # Shift c-block and 9-block up by 5 (hypothesis from step 1)
    for key in ["c_blocks", "nines"]:
        z2[key] = [[r-5, c] for (r, c) in z[key] if r-5 >= 0]
    # Character: try shifting up by 1
    if z2["char"]:
        rows = [c[0] for c in z2["char"]]
        cols = [c[1] for c in z2["char"]]
        if action == "UP":
            z2["char"] = [[r-1, c, v] for (r, c, v) in z2["char"]]
        elif action == "DOWN":
            z2["char"] = [[r+1, c, v] for (r, c, v) in z2["char"]]
        elif action == "LEFT":
            z2["char"] = [[r, c-1, v] for (r, c, v) in z2["char"]]
        elif action == "RIGHT":
            z2["char"] = [[r, c+1, v] for (r, c, v) in z2["char"]]
    return z2

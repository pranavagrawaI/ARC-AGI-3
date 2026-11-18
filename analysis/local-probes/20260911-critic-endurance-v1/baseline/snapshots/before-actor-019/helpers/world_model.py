
import numpy as np

def encode(grid):
    """Encode grid into a JSON-safe dict."""
    g = grid.astype(int)
    # Find character (0 or 1 cells)
    char_cells = [(r, c, int(g[r, c])) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] in (0, 1)]
    # Find 9-cells
    nine_cells = [(int(r), int(c)) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == 9]
    # Find c-cells (12)
    c_cells = [(int(r), int(c)) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == 12]
    return {
        "char": char_cells,
        "nines": nine_cells,
        "c_blocks": c_cells,
    }

def predict(z, action):
    """Predict next state. Basic: shift character position."""
    z2 = dict(z)
    # Parse character center
    char = z["char"]
    if not char:
        return z2
    rows = [c[0] for c in char]
    cols = [c[1] for c in char]
    cr = sum(rows) // len(rows)
    cc = sum(cols) // len(cols)
    if action == "UP":
        cr -= 1
    elif action == "DOWN":
        cr += 1
    elif action == "LEFT":
        cc -= 1
    elif action == "RIGHT":
        cc += 1
    # Shift character cells
    dr = cr - sum(rows) // len(rows)
    dc = cc - sum(cols) // len(cols)
    z2["char"] = [(r + dr, c + dc, v) for (r, c, v) in char]
    return z2

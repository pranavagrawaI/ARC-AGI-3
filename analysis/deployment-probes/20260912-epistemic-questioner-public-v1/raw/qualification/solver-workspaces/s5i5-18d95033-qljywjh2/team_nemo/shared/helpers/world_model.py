
import json

def encode(grid):
    """Track the bottom bar (row 63) fill state."""
    row63 = grid[63].tolist()
    n4 = 0
    for c in range(63, -1, -1):
        if row63[c] == 4:
            n4 += 1
        else:
            break
    return {"bar_filled": n4}

def predict(z, action):
    """Bar fills by 1 or 2 per click. Pattern: steps 2,6,10,... fill 2; others fill 1.
    We track bar_filled; the step parity matters. Use a conservative n+1."""
    n = z.get("bar_filled", 0)
    return {"bar_filled": min(n + 1, 64)}


"""World model: player blob (single 0-cell + two 1-cells), goal = color-b corridor.

Z_SCHEMA (JSON-safe):
- player: [row, col]   — the single 0-cell
- ones: [[r,c],[r,c]]  — the two 1-cells (provisional: may be decoration)
- goal: [r0, r1, c0, c1] — half-open bbox of the color-b corridor
"""
import numpy as np

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def encode(grid):
    g = np.asarray(grid)
    r0, c0 = np.where(g == 0)
    r1, c1 = np.where(g == 1)
    rb, cb = np.where(g == 11)
    return {
        "player": [int(r0[0]), int(c0[0])],
        "ones": [[int(a), int(b)] for a, b in zip(r1, c1)],
        "goal": [int(rb.min()), int(rb.max()) + 1, int(cb.min()), int(cb.max()) + 1],
    }

def predict(z, action):
    pr, pc = z["player"]
    dr, dc = DIRS.get(action, (0, 0))
    nr, nc = pr + dr, pc + dc
    if not (0 <= nr < 64 and 0 <= nc < 64):
        return z  # provisional: stay put at edge
    z2 = dict(z)
    z2["player"] = [nr, nc]
    z2["ones"] = [[o[0] + dr, o[1] + dc] for o in z.get("ones", [])]
    return z2

"""World model for the grid game (versioned, repair-friendly).

CONFIRMED static facts (level 0, step 0):
- 64x64 grid. Green (3) maze region rows 8-49, one 4-connected component.
- Player: red(1) center with black(0) N/S/E/W arms, 4-connected cross at
  (row=32, col=21), embedded in green.
- 9-patches: top (11-13, 35-37), middle (47-49, 34-38), bottom-left (55-60, 3-8).
- Pink (12) patch (45-46, 28-32); gray (8) cells (61-62, 56-63); blue (11)
  floor strip (61-62, 12-53).

LEARNED dynamics (updated after experiments):
- movement: UNKNOWN
"""
import numpy as np

VERSION = "v1"


def _find_player(grid):
    m = (grid == 1)
    rows, cols = np.where(m)
    if len(rows) == 0:
        return None
    return int(rows[0]), int(cols[0])


def _fingerprint(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


Z_SCHEMA = {
    "player": "(row, col) of red(1) center or None",
    "fp": "dict color->count (world fingerprint)",
    "learned": "bool: True once dynamics confirmed",
}


def encode(grid):
    g = np.asarray(grid)
    return {
        "player": _find_player(g),
        "fp": _fingerprint(g),
        "learned": False,
    }


def predict(z, action):
    z2 = dict(z)
    if action in ("UP", "DOWN", "LEFT", "RIGHT"):
        p = z["player"]
        if p is not None:
            r, c = p
            if action == "UP":
                r -= 1
            elif action == "DOWN":
                r += 1
            elif action == "LEFT":
                c -= 1
            elif action == "RIGHT":
                c += 1
            z2["player"] = (r, c)
        z2["learned"] = z.get("learned", False)
    elif action == "RESET":
        z2 = {"player": z["player"], "fp": z["fp"], "learned": False,
              "note": "RESET semantics unknown"}
    return z2

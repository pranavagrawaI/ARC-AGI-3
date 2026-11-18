
import json
import numpy as np

def _bbox(mask):
    if not mask.any():
        return None
    r = np.where(mask.any(axis=1))[0]
    c = np.where(mask.any(axis=0))[0]
    return [int(r[0]), int(r[-1] + 1), int(c[0]), int(c[-1] + 1)]

def encode(grid):
    grid = np.asarray(grid, dtype=np.int64)
    target = _bbox(grid == 12)
    walk = _bbox(grid == 3)
    bar = _bbox(grid == 11)
    z = {
        "schema": "arc-grid-v3",
        "counts": {int(c): int((grid == c).sum()) for c in np.unique(grid)},
        "walk_bbox": walk,
        "target_bbox": target,
        "bar_bbox": bar,
        "target_cells": int((grid == 12).sum()),
        "bar_cells": int((grid == 11).sum()),
        "frozen_hash": int(np.frombuffer(
            np.asarray(grid, dtype=np.uint8).tobytes()[:16], dtype=np.uint64)[0],
        ) % 1000000007,
    }
    return z

def _check(z):
    for key in ("target_bbox", "bar_bbox", "frozen_hash"):
        if key not in z:
            raise ValueError("malformed latent: missing " + key)

def predict(z, action):
    z = dict(z)
    _check(z)
    action = str(action).upper()
    target = list(z["target_bbox"])
    bar = list(z["bar_bbox"])
    changed = False
    if action == "LEFT":
        target[2] -= 5
        target[3] -= 5
        if z["bar_cells"] >= 2:
            bar[2] += 1
        changed = True
    elif action == "RIGHT":
        target[2] += 5
        target[3] += 5
        if z["bar_cells"] >= 2:
            bar[2] += 1
        changed = True
    elif action in {"UP", "DOWN"}:
        # Measured: both drain the 11-bar left edge by 1 column (2 cells).
        if z["bar_cells"] >= 2:
            bar[2] += 1
        changed = True
    else:
        if action not in {"RESET", "UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError("unknown action " + action)
    z.update({
        "target_bbox": target,
        "bar_bbox": bar,
        "measured_action": action,
        "changed": changed,
    })
    return z

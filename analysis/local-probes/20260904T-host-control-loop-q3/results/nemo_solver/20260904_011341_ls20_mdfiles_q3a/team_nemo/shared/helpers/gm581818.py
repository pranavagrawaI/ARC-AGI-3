
"""World model for game-581818: maze navigation with a controllable 5x5 entity.

Z_SCHEMA:
  entity: int tuple (r, c)  — top-left cell of the 5x5 entity block
  level:  int 0..N
  static_patch: tuple (r, c)  — position of the static 3x3 9-patch (Room 1 marker)

Entity footprint = 5x5 (top-left = entity):
  [[12,12,12,12,12],
   [12,12,12,12,12],
   [ 9, 9, 9, 9, 9],
   [ 9, 9, 9, 9, 9],
   [ 9, 9, 9, 9, 9]]
Entity moves 5 cells per action in the specified direction.
Entity can pass through walls (3-color cells).
"""
import numpy as np

ENTITY_PAT = np.array([
    [12,12,12,12,12],
    [12,12,12,12,12],
    [ 9, 9, 9, 9, 9],
    [ 9, 9, 9, 9, 9],
    [ 9, 9, 9, 9, 9],
], dtype=np.uint8)

def entity_from_grid(g):
    """Locate the 5x5 entity block; return (r0, c0) top-left, or None."""
    # Find a 5x5 region with the entity pattern (2 rows of 12 + 3 rows of 9)
    H, W = g.shape
    for r0 in range(H - 4):
        for c0 in range(W - 4):
            blk = g[r0:r0+5, c0:c0+5]
            if (blk[:2] == 12).all() and (blk[2:] == 9).all():
                return (r0, c0)
    return None

def encode(g):
    return {"entity": entity_from_grid(g)}

def predict(g, action):
    """grid + action -> (grid', expect) where expect is
    move / no_impact / level_up / terminal."""
    if action in ("RESET", "UNDO"):
        return g.copy(), "no_impact"
    d = {"UP": (-5, 0), "DOWN": (5, 0), "LEFT": (0, -5), "RIGHT": (0, 5)}
    if action not in d:
        return g.copy(), "no_impact"
    e = entity_from_grid(g)
    if e is None:
        return g.copy(), "no_impact"
    r, c = e
    dr, dc = d[action]
    n0, n1 = r + dr, c + dc
    # Check bounds
    if n0 < 0 or n1 < 0 or n0 + 5 > 64 or n1 + 5 > 64:
        return g.copy(), "no_impact"
    ng = g.copy()
    ng[r:r+5, c:c+5] = 3  # restore walls
    ng[n0:n0+5, n1:n1+5] = ENTITY_PAT
    return ng, "move"

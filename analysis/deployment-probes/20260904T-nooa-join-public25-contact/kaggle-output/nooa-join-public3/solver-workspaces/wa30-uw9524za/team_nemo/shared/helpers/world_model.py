"""World model for the 64x64 grid game."""
import numpy as np

WALL = 4
BG = 1
PLAYER = 14
TARGET = 2
STEP = 4

def encode(grid):
    """grid: 64x64 numpy array -> latent z dict.
    z = {"walls": tuple((r0,c0,r1,c1)), "target": (r,c) center of 10x10 target,
         "player": (r,c) center of player token}"""
    z = {"walls": [], "target": None, "player": None}
    # walls: 4x4 blocks of color 4
    for r in range(0, 64, STEP):
        for c in range(0, 64, STEP):
            block = grid[r:r+4, c:c+4]
            if block.size and np.all(block == WALL):
                z["walls"].append((r, c, r + 3, c + 3))
    z["walls"] = tuple(z["walls"])
    # target: the big blob of color 2
    tr = np.argwhere(grid == TARGET)
    if len(tr):
        r0, r1 = int(tr[:, 0].min()), int(tr[:, 0].max())
        c0, c1 = int(tr[:, 1].min()), int(tr[:, 1].max())
        z["target"] = ((r0 + r1) // 2, (c0 + c1) // 2)
    # player: 4x4 blob of color 14
    pr = np.argwhere(grid == PLAYER)
    if len(pr):
        r0, r1 = int(pr[:, 0].min()), int(pr[:, 0].max())
        c0, c1 = int(pr[:, 1].min()), int(pr[:, 1].max())
        z["player"] = ((r0 + r1) // 2, (c0 + c1) // 2)
    return z

def _wallcells(z):
    out = set()
    for (wr0, wc0, wr1, wc1) in z["walls"]:
        for rr in range(wr0, wr1 + 1):
            for cc in range(wc0, wc1 + 1):
                out.add((rr, cc))
    return out

def predict(z, action):
    """Move player 4 cells (one square) if the strip between old and new
    center is free of walls. Goal when player 4x4 body lands inside the
    10x10 target zone."""
    if action in ("SPACEBAR", "RESET", "UNDO"):
        return dict(z)
    delta = {"UP": (-4, 0), "DOWN": (4, 0), "LEFT": (0, -4), "RIGHT": (0, 4)}
    if action not in delta:
        return dict(z)
    r, c = z["player"]
    dr, dc = delta[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr <= 60 and 0 <= nc <= 60):
        return dict(z)
    rr0, rr1 = sorted((r, nr))
    cc0, cc1 = sorted((c, nc))
    strip = {(rr, cc) for rr in range(rr0, rr1 + 1) for cc in range(cc0, cc1 + 1)}
    strip.discard((r, c))
    if strip & _wallcells(z):
        return dict(z)  # blocked
    nz = dict(z)
    nz["player"] = (nr, nc)
    tr, tc = z["target"]
    # player body rows nr-1..nr+2, cols nc-1..nc+2 inside target rows tr-4..tr+5
    nz["goal"] = any(tr - 4 <= rr <= tr + 5 and tc - 4 <= cc <= tc + 5
                     for rr in range(nr - 1, nr + 3) for cc in range(nc - 1, nc + 3))
    return nz

def is_goal(z):
    return bool(z.get("goal"))

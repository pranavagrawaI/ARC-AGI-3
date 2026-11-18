
"""
World model for the tile-board game.

Colors: 0=bg(holes), 1=tile, 5=frame, 9=HUD-row, 10=offboard, 14=gem/cross.
H4 hypothesis: the whole interior blob (cells in {0,1,14}) shifts rigidly in
the direction of a move, until blocked by the frame (5) or offboard (10).
"""
import numpy as np

OFFBOARD = 10
FRAME = 5
TILE = 1
GEM = 14
BG = 0

def encode(grid):
    blob = np.isin(grid, [BG, TILE, GEM])
    return {
        "grid": grid.copy(),
        "blob": blob,
        "n_crosses": int((grid == GEM).sum()),
        "hud": grid[0].copy(),
    }

def _rigid_max_shift(blob, immovable, dr, dc):
    h, w = blob.shape
    rows, cols = np.where(blob)
    k = 0
    while True:
        r0, c0 = (k+1)*dr, (k+1)*dc
        nr, nc = rows + r0, cols + c0
        if nr.min() < 0 or nr.max() >= h or nc.min() < 0 or nc.max() >= w:
            break
        if immovable[nr, nc].any():
            break
        k += 1
    return k

def predict(z, action):
    grid = z["grid"]
    blob = z["blob"]
    immovable = ~blob
    dr, dc = {"UP": (-1,0), "DOWN": (1,0), "LEFT": (0,-1), "RIGHT": (0,1)}[action]
    k = _rigid_max_shift(blob, immovable, dr, dc)
    new_grid = grid.copy()
    if k > 0:
        rows, cols = np.where(blob)
        shifted = np.zeros_like(blob)
        shifted[rows + k*dr, cols + k*dc] = True
        new_grid[blob] = OFFBOARD
        new_grid[shifted] = grid[blob]
    return {
        "grid": new_grid,
        "blob": np.isin(new_grid, [BG, TILE, GEM]),
        "n_crosses": int((new_grid == GEM).sum()),
        "hud": new_grid[0].copy(),
        "k": k,
    }

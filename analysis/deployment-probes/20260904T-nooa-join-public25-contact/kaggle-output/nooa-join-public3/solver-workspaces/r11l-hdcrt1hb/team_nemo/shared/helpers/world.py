
"""World model for the 64x64 grid game (turn 0).

Z_SCHEMA:
  level: int                 # current level index
  state: str                 # NOT_FINISHED / level_up / WIN / GAME_OVER
  step: int                  # env step count
  playfield_rows: (rmin, rmax)   # rows of the '5' playfield (col 0-1 boundary)
  objects: dict  color(1,3,6,f) -> sorted list of (r, c)
  bg_counts: dict color -> count
"""
import numpy as np
from collections import Counter

PLAYFIELD_BG = 5
CORNERS_BG = 2

def _bbox(cells):
    cells = sorted(cells)
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    return (min(rs), max(rs), min(cs), max(cs))

def label_components(mask):
    """4-connected components of a boolean mask -> list of (r,c) cell lists."""
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
                comps.append(cells)
    return comps

def encode(grid):
    """grid: 64x64 int array. Returns latent dict z (see Z_SCHEMA)."""
    H, W = grid.shape
    # playfield: the big 5 region. Its rows: rows where the majority is 5 or 2.
    five = (grid == PLAYFIELD_BG)
    two = (grid == CORNERS_BG)
    rows_in = np.where(five.any(axis=1) | two.any(axis=1))[0]
    objects = {}
    for color in (1, 3, 6, 0xF):
        cells = sorted(zip(*np.where(grid == color)))
        if cells:
            objects[int(color)] = {"cells": [list(m) for m in cells], "bbox": _bbox(cells)}
    zero_cells = sorted(zip(*np.where(grid == 0)))
    z = {
        "playfield_rows": (int(rows_in.min()), int(rows_in.max())),
        "playfield_cols": (int(np.where(five.any(axis=0))[0].min()), int(np.where(five.any(axis=0))[0].max())),
        "objects": objects,
        "holes": [list(m) for m in zero_cells],
        "counts": dict(Counter(grid.ravel().tolist())),
    }
    return z

def short(z):
    """One-line summary of a latent for logging."""
    objs = {c: (len(v["cells"]), v["bbox"]) for c, v in z["objects"].items()}
    return f"playfield rows {z['playfield_rows']} cols {z['playfield_cols']} objects={objs} holes={len(z['holes'])}"

def grid_from_rows(grid_rows):
    return np.array([[int(ch, 16) for ch in row] for row in grid_rows])

def grid_to_rows(g):
    return ["".join(f"{v:x}" for v in row) for row in g]

def diff_z(z0, z1):
    """Named-field diff between two latents."""
    out = {}
    if z0["objects"].keys() != z1["objects"].keys():
        out["object_colors"] = (sorted(z0["objects"]), sorted(z1["objects"]))
    for c in z0["objects"]:
        if c in z1["objects"] and z0["objects"][c]["cells"] != z1["objects"][c]["cells"]:
            a, b = z0["objects"][c]["cells"], z1["objects"][c]["cells"]
            added = set(b) - set(a); removed = set(a) - set(b)
            out[f"color{c}"] = (len(removed), len(added), [tuple(x) for x in (added | removed)])
    if z0["holes"] != z1["holes"]:
        out["holes"] = (len(z0["holes"]), len(z1["holes"]))
    return out

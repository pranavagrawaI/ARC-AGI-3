
import numpy as np

BG = 5
BG_ROW = 63  # bottom row is a status band (all 3 at start)

def components(mask):
    """8-connected component labeling; returns (labels, n)."""
    lab = np.zeros(mask.shape, dtype=int)
    H, W = mask.shape
    n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                n += 1
                lab[y, x] = n
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = n
                                stack.append((ny, nx))
    return lab, n

def extract(grid):
    """grid: 64x64 int array. Returns list of object dicts (bbox, colors, cores)."""
    mask = (grid != BG)
    mask[BG_ROW, :] = False  # ignore status band
    lab, n = components(mask)
    objs = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        cu, cc = np.unique(grid[ys, xs], return_counts=True)
        d = {"id": i, "r0": int(ys.min()), "r1": int(ys.max()),
             "c0": int(xs.min()), "c1": int(xs.max()), "size": int(len(ys)),
             "colors": {int(k): int(v) for k, v in zip(cu, cc)}}
        if 13 in d["colors"]:
            cores = list(zip(*np.where(grid == 13)))
            d["cores"] = [(int(r), int(c)) for r, c in cores
                          if d["r0"] <= r <= d["r1"] and d["c0"] <= c <= d["c1"]]
        else:
            d["cores"] = []
        objs.append(d)
    return objs

def status_row(grid):
    """Frozenset of (color,count) on the status band."""
    row = grid[BG_ROW]
    vals, cnts = np.unique(row, return_counts=True)
    return tuple(zip(vals.tolist(), cnts.tolist()))

def encode(grid):
    """Latent z = (tuple of sorted object signatures, status band)."""
    objs = extract(grid)
    sig = tuple(sorted(
        (o["r0"], o["r1"], o["c0"], o["c1"], tuple(sorted(o["colors"].items())),
         tuple(o["cores"])) for o in objs))
    return {"objs": sig, "status": status_row(grid)}

def predict(z, action):
    """Placeholder dynamics: unknown. Return z unchanged; refine after experiments."""
    return z

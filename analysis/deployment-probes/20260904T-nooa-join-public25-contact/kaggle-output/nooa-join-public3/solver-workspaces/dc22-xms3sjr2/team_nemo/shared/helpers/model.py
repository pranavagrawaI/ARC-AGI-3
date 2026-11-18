"""World model for the grid game — v0: extraction only."""
import numpy as np

Z_SCHEMA = {
    "blocks_2": "list[(r0,r1,c0,c1)]  # color-2 rectangles (player? target?)",
    "sq8":      "list[(r0,r1,c0,c1)]  # color-8 squares",
    "sq9":      "list[(r0,r1,c0,c1)]  # color-9 squares",
    "core11":   "list[(r0,r1,c0,c1)]  # color-11 small cores",
    "sq13":     "list[(r0,r1,c0,c1)]  # color-13 squares",
    "core14":   "list[(r0,r1,c0,c1)]  # color-14 small cores",
    "patches0": "list[(r0,r1,c0,c1)]  # 4x4 zero holes inside 4/5 wall zones",
    "n_colors": "int",
}

def _components(mask, min_size=1):
    """Connected components (4-conn); return [(r0,r1,c0,c1,count),...]."""
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                comp = []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                if len(comp) >= min_size:
                    ys = [p[0] for p in comp]; xs = [p[1] for p in comp]
                    out.append((min(ys), max(ys), min(xs), max(xs), len(comp)))
    return out

def encode(grid):
    """grid: 64x64 int array -> z dict with all extracted entities."""
    z = {"n_colors": int(len(np.unique(grid)))}
    for c, name in [(2, "blocks_2"), (8, "sq8"), (9, "sq9"), (11, "core11"),
                    (13, "sq13"), (14, "core14")]:
        z[name] = [(r0, r1, c0, c1) for (r0, r1, c0, c1, n) in _components(grid == c)]
    patches = []
    for (r0, r1, c0, c1, n) in _components(grid == 0):
        w, h = c1-c0+1, r1-r0+1
        if w == 64:                      # full row (status band)
            continue
        if c0 == 32 and c1 == 32:        # the wall column
            continue
        if w == 4 and h == 4:
            patches.append((r0, r1, c0, c1))
    z["patches0"] = sorted(patches)
    return z

def predict(z, action):
    """Placeholder dynamics — not learned yet. Returns a shallow copy."""
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in z.items()}


import numpy as np

Z_SCHEMA = """
z = {
  'level': int, 'step': int,
  'player': [r,c] or None,
  'screen_rows': [r0, r1],        # screen is at rows 40..49, cols 34..38
  'screen_top': int,               # row of topmost screen row (scrolls)
  'cells': { 'r,c': color }        # every non-{3,4,5} cell
  'grid_hash': int
}
"""

def encode(grid):
    g = np.asarray(grid)
    H, W = g.shape
    cells = {}
    for r in range(H):
        for c in range(W):
            v = int(g[r,c])
            if v not in (3,4,5):
                cells[f"{r},{c}"] = v
    return {
        'level': -1, 'step': -1,
        'player': None,
        'cells': cells,
        'grid_hash': int(np.packbits(g.astype(np.uint8)).tobytes()[:4]) if H*W<=65536 else 0,
    }

def _apply(z, action):
    cells = dict(z['cells'])
    dr, dc = {"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1)}[action]
    new = {}
    moved = False
    for key, v in cells.items():
        r,c = map(int, key.split(","))
        nr, nc = r+dr, c+dc
        key2 = f"{nr},{nc}"
        if key2 in cells and cells[key2] == v:
            # would overlap an existing same-color cell: keep at place
            new[key] = v
        else:
            new[key2] = v
            moved = True
    return {**z, 'cells': new, 'moved': moved}

def predict(z, action):
    z2 = _apply(z, action)
    z2.pop('moved', None)
    return z2

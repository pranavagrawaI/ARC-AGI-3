
import numpy as np

def _bbox(cells):
    if len(cells) == 0:
        return None
    a = np.asarray(cells)
    return [int(a[:,0].min()), int(a[:,0].max()), int(a[:,1].min()), int(a[:,1].max()), len(a)]

def encode(grid):
    """Compact latent: bbox + count per color of interest, plus mover/trail exact pixels
    (small), plus static object bboxes. Does NOT assume dynamics are correct."""
    g = np.asarray(grid)
    z = {}
    for col in (0, 1, 3, 4, 5, 8, 9, 11, 12):
        cells = np.argwhere(g == col)
        if len(cells):
            z[f'c{col}'] = _bbox(cells)
    # exact mover (5x3=15 pixels) + trail (2x5=10) + bar endpoints (compact)
    nine = np.argwhere(g == 9)
    if len(nine):
        z['mover_cells'] = [[int(r), int(c)] for r, c in nine]
    trail = np.argwhere(g == 12)
    if len(trail):
        z['trail_cells'] = [[int(r), int(c)] for r, c in trail]
    bar = np.argwhere(g == 11)
    if len(bar):
        # keep only endpoints: leftmost, rightmost cells in each row
        rows = {}
        for r, c in bar:
            rows.setdefault(int(r), []).append(int(c))
        z['bar_rows'] = {str(r): [min(v), max(v)] for r, v in sorted(rows.items())}
        z['bar_len'] = len(bar)
    bc = np.bincount(g.ravel().astype(int))
    z['counts'] = {int(k): int(v) for k, v in enumerate(bc) if v}
    return z

def predict(z, action):
    """Tentative model: 9-block (the 5x3 solid, rows 37-39 cols 39-43) moves 2 rows
    UP/DOWN or 5 cols LEFT/RIGHT; a 2x5 12-trail appears in the vacated band;
    the 11-bar loses 2 cells/move from the left end."""
    z2 = {k: (list(v) if isinstance(v, (list, tuple)) else dict(v) if isinstance(v, dict) else v)
          for k, v in z.items()}
    mover = np.array(z.get('mover_cells', []), dtype=int)
    if len(mover) == 0:
        return z2
    dr, dc = {'UP': (-2, 0), 'DOWN': (2, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}.get(action, (0, 0))
    new = mover + [dr, dc]
    z2['mover_cells'] = [[int(r), int(c)] for r, c in new]
    if action in ('UP', 'DOWN'):
        cols = sorted(set(int(c) for r, c in new))
        base = int(mover[0, 0]) - 1 if action == 'UP' else int(mover[-1, 0])
        z2['trail_cells'] = [[int(base + k), int(c)] for k in (0, 1) for c in cols]
    if action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
        bar_len = z.get('bar_len', 0)
        z2['bar_len'] = max(0, bar_len - 2)
    return z2

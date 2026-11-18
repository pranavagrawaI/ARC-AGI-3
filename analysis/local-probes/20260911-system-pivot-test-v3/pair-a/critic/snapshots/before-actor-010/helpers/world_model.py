
import numpy as np

def encode(grid):
    g = np.asarray(grid)
    z = {}
    nine = np.argwhere(g == 9)
    if len(nine):
        z['mover'] = [[int(r), int(c)] for r, c in nine]
    trail = np.argwhere(g == 12)
    if len(trail):
        z['trail'] = [[int(r), int(c)] for r, c in trail]
    bar = np.argwhere(g == 11)
    if len(bar):
        z['bar'] = [[int(r), int(c)] for r, c in bar]
    for col in (0, 1, 3, 4, 5, 8):
        cells = np.argwhere(g == col)
        if len(cells):
            z[f'c{col}'] = [[int(r), int(c)] for r, c in cells]
    bc = np.bincount(g.ravel().astype(int))
    z['counts'] = {int(k): int(v) for k, v in enumerate(bc)}
    return z

def predict(z, action):
    """Tentative model: 9-block moves 2 rows (UP/DOWN) or 5 cols (LEFT/RIGHT),
    leaves a 2x5 12-trail in the vacated band, and 11-bar loses 2 cells/move from left."""
    z2 = {k: list(v) for k, v in z.items()}
    mover = np.array(z.get('mover', []), dtype=int)
    if len(mover) == 0:
        return z2
    dr, dc = {'UP': (-2, 0), 'DOWN': (2, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}.get(action, (0, 0))
    new = mover + [dr, dc]
    z2['mover'] = [[int(r), int(c)] for r, c in new]
    if action in ('UP', 'DOWN'):
        cols = sorted(set(int(c) for r, c in new))
        base = int(mover[0, 0]) if action == 'UP' else int(mover[-1, 0]) - 1
        z2['trail'] = [[int(base + k), int(c)] for k in (0, 1) for c in cols]
    bar = [tuple(p) for p in z.get('bar', [])]
    if bar and action in ('UP','DOWN','LEFT','RIGHT'):
        bar = bar[2:]
        z2['bar'] = [[int(r), int(c)] for r, c in bar]
    return z2

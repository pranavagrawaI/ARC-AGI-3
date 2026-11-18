
import numpy as np

def encode(grid):
    """JSON-safe latent capturing the mover, trail, budget bar, and static object positions.
    Known unknowns recorded: whether 8-block/9-B/0-1 cross/3-box are goal objects."""
    g = np.asarray(grid)
    z = {}
    # 9-block (the solid 5x3 mover): 15 cells at rows 37-39, cols 39-43 currently.
    nine = np.argwhere(g == 9)
    if len(nine):
        z['mover'] = [[int(r), int(c)] for r, c in nine]
    # 12 trail: 2x5 at rows 35-36 cols 39-43
    trail = np.argwhere(g == 12)
    if len(trail):
        z['trail'] = [[int(r), int(c)] for r, c in trail]
    # 11 bar (budget?): cells in rows 61-62
    bar = np.argwhere(g == 11)
    if len(bar):
        z['bar'] = [[int(r), int(c)] for r, c in bar]
    # other colors, positions
    for col in (0, 1, 3, 4, 5, 8):
        cells = np.argwhere(g == col)
        if len(cells):
            z[f'c{col}'] = [[int(r), int(c)] for r, c in cells]
    # static features: 0/1 cross, 8-block, 3-box
    z['counts'] = {str(k): int(v) for k, v in np.bincount(g.ravel())}
    return z

def predict(z, action):
    """Tentative model: mover shifts 2 rows up/down, 5 cols left/right, leaves a 2x5 trail
    in the vacated row/col band, and the 11-bar loses 2 cells from the left end per move.
    This is a working hypothesis, not an established rule."""
    z2 = dict(z)
    z2['counts'] = dict(z['counts'])
    mover = np.array(z.get('mover', []), dtype=int)
    if len(mover) == 0:
        return z2
    dr, dc = {'UP': (-2, 0), 'DOWN': (2, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}.get(action, (0, 0))
    new = mover + [dr, dc]
    z2['mover'] = [[int(r), int(c)] for r, c in new]
    if action in ('UP', 'DOWN'):
        # trail occupies the vacated 2 rows at mover cols
        if action == 'UP':
            tr_rows = [int(mover[0, 0]), int(mover[0, 0]) + 1]  # rows 38,39 band? actually vacated 37-39 -> trail 2 rows
        cols = sorted(set(int(c) for r, c in new))
        z2['trail'] = [[int(r), int(c)] for r in (int(mover[0,0])-1, int(mover[0,0])) for c in cols]
    # bar decrement (tentative)
    bar = [tuple(p) for p in z.get('bar', [])]
    if bar and action in ('UP','DOWN','LEFT','RIGHT'):
        bar = bar[2:]
        z2['bar'] = [[int(r), int(c)] for r, c in bar]
    return z2

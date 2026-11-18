
import numpy as np

def _bbox(cells):
    if len(cells) == 0:
        return None
    a = np.asarray(cells)
    return [int(a[:,0].min()), int(a[:,0].max()), int(a[:,1].min()), int(a[:,1].max()), len(a)]

def encode(grid):
    """Compact latent: bbox + count per color, plus exact mover (5x3) and trail (2x5) pixels,
    plus bar endpoints. Does NOT assume dynamics are correct."""
    g = np.asarray(grid)
    z = {}
    for col in (0, 1, 3, 4, 5, 8, 9, 11, 12):
        cells = np.argwhere(g == col)
        if len(cells):
            z[f'c{col}'] = _bbox(cells)
    # exact 5x3 mover: find the dense 5x3 cluster of 9s (rows with 5+ 9s in a contiguous band)
    nine = np.argwhere(g == 9)
    if len(nine):
        from collections import Counter
        row_counts = Counter(r for r, c in nine)
        # The 5x3 block is the contiguous run of 3 rows each with exactly 5 9s in the same 5 cols
        dense = sorted([r for r, cnt in row_counts.items() if cnt == 5])
        # find contiguous run of 3
        block = None
        for i in range(len(dense) - 2):
            if dense[i+1] == dense[i]+1 and dense[i+2] == dense[i]+2:
                block = dense[i:i+3]
                break
        if block:
            r0 = block[0]
            cols = sorted(c for r, c in nine if r == r0)
            if len(cols) == 5:
                z['mover'] = [list(block[0]), list(cols)]  # [row0, [c0..c4]]
    trail = np.argwhere(g == 12)
    if len(trail):
        a = trail
        z['trail'] = _bbox(a)
        z['trail_cells'] = [[int(r), int(c)] for r, c in trail]
    bar = np.argwhere(g == 11)
    if len(bar):
        rows = {}
        for r, c in bar:
            rows.setdefault(int(r), []).append(int(c))
        z['bar_rows'] = {str(r): [min(v), max(v)] for r, v in sorted(rows.items())}
        z['bar_len'] = len(bar)
    bc = np.bincount(g.ravel().astype(int))
    z['counts'] = {int(k): int(v) for k, v in enumerate(bc) if v}
    return z

def predict(z, action):
    """Model (step 1 validated): 9-block (5x3) moves 5 rows UP/DOWN or 5 cols LEFT/RIGHT.
    12-band (2x5 'hat') tracks the 9-block: 2 rows above for vertical moves,
    5 cols to the left for LEFT (i.e., behind the direction of motion), 5 cols to the
    right for RIGHT. 11-bar loses 2 cells from the left end per move."""
    z2 = {k: (list(v) if isinstance(v, (list,)) else dict(v) if isinstance(v, dict) else v)
          for k, v in z.items()}
    if 'mover' not in z2:
        return z2
    r0, cols = z2['mover']
    r0 = int(r0); cols = [int(c) for c in cols]
    c0 = min(cols); c1 = max(cols)
    dr, dc = {'UP': (-5, 0), 'DOWN': (5, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}.get(action, (0, 0))
    new_r0 = r0 + dr; new_c0 = c0 + dc; new_c1 = c1 + dc
    z2['mover'] = [new_r0, [new_c0 + i for i in range(5)]]
    # 12-band: 2 rows above the block for vertical; for horizontal, 5 cols behind (opposite to motion)
    if action == 'UP':
        tr_rows = [new_r0 - 2, new_r0 - 1]
        tr_cols = [new_c0 + i for i in range(5)]
    elif action == 'DOWN':
        tr_rows = [new_r0 + 3, new_r0 + 4]
        tr_cols = [new_c0 + i for i in range(5)]
    elif action == 'LEFT':
        tr_rows = [new_r0, new_r0 + 1]
        tr_cols = [new_c0 - 5 + i for i in range(5)]
    elif action == 'RIGHT':
        tr_rows = [new_r0, new_r0 + 1]
        tr_cols = [new_c1 + 1 + i for i in range(5)]
    else:
        tr_rows, tr_cols = None, None
    if tr_rows is not None:
        z2['trail_cells'] = [[int(r), int(c)] for r in tr_rows for c in tr_cols]
        z2['trail'] = [min(tr_rows), max(tr_rows), min(tr_cols), max(tr_cols), len(tr_cols)*2]
    # 11-bar loses 2 from left
    if action in ('UP','DOWN','LEFT','RIGHT'):
        z2['bar_len'] = max(0, z2.get('bar_len', 0) - 2)
        br = z2.get('bar_rows')
        if br:
            for k in br:
                lo, hi = br[k]
                br[k] = [lo + 2, hi]
    return z2

# world model
import numpy as np

DIRS = {'UP': (-1,0),'DOWN': (1,0),'LEFT': (0,-1),'RIGHT': (0,1)}

def encode(grid):
    g = grid if isinstance(grid, np.ndarray) else np.asarray(grid)
    pm = np.isin(g, [0,1])
    rmin = int(np.where(pm.any(axis=1))[0][0])
    cmin = int(np.where(pm.any(axis=0))[0][0])
    gm = (g == 9)
    goal = None
    for r in range(g.shape[0]-4):
        row_hit = np.where(gm[r:r+5].all(axis=0))[0]
        if len(row_hit) == 5:
            goal = (r+2, int(row_hit[0])+2)
            break
    return {'player': (rmin, cmin), 'goal': goal, 'walls': (g == 3)}

def _blocked(z, r, c):
    w = z['walls']
    return not (0 <= r < 62 and 0 <= c < 62) or bool(w[r:r+2, c:c+2].any())

def predict(z, action):
    z2 = dict(z)
    r, c = z['player']
    if action in DIRS:
        dr, dc = DIRS[action]
        nr, nc = r+dr, c+dc
        if not _blocked(z, nr, nc):
            z2['player'] = (nr, nc)
    return z2

def bfs(z, max_steps=64):
    from collections import deque
    start = z['player']
    if z['goal'] is None:
        return None
    gr, gc = z['goal']
    target = {(r, c) for r in range(gr-2, gr+3) for c in range(gc-2, gc+3)}
    seen = {start}
    q = deque([(start, [])])
    while q:
        pos, path = q.popleft()
        if len(path) > max_steps:
            continue
        if pos in target:
            return path
        for a, (dr, dc) in DIRS.items():
            n = (pos[0]+dr, pos[1]+dc)
            if n in seen or _blocked(z, *n):
                continue
            seen.add(n)
            q.append((n, path+[a]))
    return None

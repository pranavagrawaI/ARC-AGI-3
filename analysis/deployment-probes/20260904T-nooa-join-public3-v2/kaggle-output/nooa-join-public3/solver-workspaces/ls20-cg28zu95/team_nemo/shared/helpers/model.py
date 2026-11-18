import numpy as np

Z_SCHEMA = {
  'player': '(row, col) of 0/1 cluster center or None',
  'target_bbox': '(r0, r1, c0, c1) of color-c block or None',
  'level': 'int',
}

DIRS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}

# Wall rectangles (r0, r1, c0, c1) inclusive — extracted from initial grid
WALLS = [
    (8, 16, 32, 38),    # top box outline + fill area
    (16, 24, 33, 37),    # vertical wall segment below top box
    (25, 29, 33, 37),    # wall separating corridor from upper room
    (40, 44, 19, 23),    # vertical wall left of lower area
    (45, 49, 33, 38),    # wall below/around target
    (25, 29, 14, 53),    # top wall of main room (with gap at cols 29-33)
    (25, 29, 38, 53),    # top wall right portion
    (40, 49, 14, 18),    # left wall of main room (partial)
    (40, 49, 41, 53),    # right wall of main room
    (49, 51, 14, 53),    # bottom wall of main room
    (25, 29, 14, 18),    # left wall top portion
    (16, 24, 19, 23),    # wall in upper area
    (40, 44, 33, 38),    # wall in lower-middle
    (8, 10, 32, 38),     # top of upper box
    (14, 15, 32, 38),    # bottom of upper box
]

def walls_grid():
    g = np.zeros((64, 64), dtype=bool)
    for (r0, r1, c0, c1) in WALLS:
        g[r0:r1+1, c0:c1+1] = True
    return g

WG = walls_grid()

def encode(grid, level=None):
    g = np.asarray(grid)
    z = {}
    pm = np.argwhere((g == 0) | (g == 1))
    z['player'] = (int(pm[0][0]), int(pm[0][1])) if len(pm) else None
    z['n_player'] = int(len(pm))
    cm = np.argwhere(g == 0x0c)
    z['target_bbox'] = (int(cm[:,0].min()), int(cm[:,0].max()), int(cm[:,1].min()), int(cm[:,1].max())) if len(cm) else None
    z['level'] = level
    return z

def is_free(r, c):
    if not (0 <= r < 64 and 0 <= c < 64):
        return False
    return not WG[r, c]

def step_dir(z, d):
    r, c = z['player']
    dr, dc = DIRS[d]
    if is_free(r + dr, c + dc):
        return (r + dr, c + dc)
    return (r, c)

def predict(z, action):
    z2 = dict(z)
    if action in DIRS and z.get('player'):
        z2['player'] = step_dir(z, action)
    return z2

def plan_to(z_from, goal_rc, max_steps=500):
    from collections import deque
    start = z_from['player']
    if start == goal_rc:
        return []
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal_rc:
            break
        r, c = cur
        for d, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if is_free(nr, nc) and (nr, nc) not in prev:
                prev[(nr, nc)] = (cur, d)
                q.append((nr, nc))
    if goal_rc not in prev:
        return None
    path = []
    cur = goal_rc
    while prev[cur] is not None:
        par, d = prev[cur]
        path.append(d)
        cur = par
    path.reverse()
    return path if len(path) <= max_steps else None

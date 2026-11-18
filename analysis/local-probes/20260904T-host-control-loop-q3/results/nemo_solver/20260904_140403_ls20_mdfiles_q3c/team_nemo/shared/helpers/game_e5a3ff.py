import numpy as np
from collections import deque

GOAL = [(r, c) for r in range(45, 50) for c in range(34, 39)]
ITEMS0 = [(31, 21), (32, 20), (32, 21), (32, 22), (33, 21)]
START0 = (11, 35)
DIR = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def encode(grid_rows):
    a = np.array([[int(ch, 16) for ch in row] for row in grid_rows], dtype=np.int64)
    z = {"grid": a}
    ys, xs = np.where(a == 9)
    if len(ys):
        z["mover"] = (int(ys.min()), int(xs.min()))
        z["mover_size"] = (int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
    else:
        z["mover"] = None
        z["mover_size"] = None
    z["items"] = sorted([(r, c) for r, c in zip(*np.where((a == 0) | (a == 1) | (a == 12)))])
    return z

def step_mover(a, r, c, action, size=5):
    dr, dc = DIR[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr and nr + size < 64 and 0 <= nc and nc + size < 64):
        return r, c, False, True
    for rr in range(nr, nr + size):
        for cc in range(nc, nc + size):
            if a[rr, cc] == 3:
                return r, c, False, True
    return nr, nc, True, False

def in_goal(mover, size=5):
    g = set(GOAL)
    r, c = mover
    return any((rr, cc) in g for rr in range(r, r + size) for cc in range(c, c + size))

def predict(z, action, size=5):
    z2 = dict(z)
    a = z["grid"]
    r, c = z["mover"]
    nr, nc, moved, _ = step_mover(a, r, c, action, size)
    z2["mover"] = (nr, nc) if moved else (r, c)
    z2["event"] = "move" if moved else "no_impact"
    z2["in_goal"] = in_goal(z2["mover"], size)
    return z2

def plan_to(z, target, size=5):
    a = z["grid"]
    start = z["mover"]
    q = deque([(start, [])])
    seen = {start}
    while q:
        pos, path = q.popleft()
        if pos == target:
            return path
        for act in ("LEFT", "RIGHT", "UP", "DOWN"):
            nr, nc, moved, _ = step_mover(a, pos[0], pos[1], act, size)
            if moved and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), path + [act]))
    return None


import numpy as np

# World model. Glyph: 5 cells. head = color-0 cell at min(row,col).
# Shape relative to head: (0,0) head; body (1,0),(1,1),(1,-1),(2,0).
SHAPE = [(0, 0), (1, 0), (1, 1), (1, -1), (2, 0)]
PICKUP = 9

def find_player(grid):
    ys, xs = np.where(grid == 0)
    if len(ys) == 0:
        return None
    r, c = int(ys.min()), int(xs[ys.argmin()])
    return r, c

def _cells(head):
    return [(head[0] + dr, head[1] + dc) for (dr, dc) in SHAPE]

def occupied(grid, r, c):
    if r < 0 or c < 0 or r >= grid.shape[0] or c >= grid.shape[1]:
        return True
    return int(grid[r, c]) not in (4, 0, 1, PICKUP)

def encode(grid):
    head = find_player(grid)
    goal_ys, goal_xs = np.where(grid == 12)
    goal = [int(goal_ys.min()), int(goal_xs.min()), int(goal_ys.max()), int(goal_xs.max())] if len(goal_ys) else None
    return {"head": [head[0], head[1]], "goal": goal, "n9": int((grid == 9).sum())}

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

def _move_cells(grid, head, dr, dc):
    """New glyph cells after head moves by (dr,dc); returns None if blocked, else new head."""
    cur = set(_cells(head))
    r, c = head
    nr, nc = r + dr, c + dc
    for (rr, cc) in _cells((nr, nc)):
        if (rr, cc) in cur:
            continue
        if occupied(grid, rr, cc):
            return None
    return (nr, nc)

def predict(z, action):
    """Pure latent step (movement blocked info unavailable without grid; pass through)."""
    if action not in DIRS:
        return dict(z)
    return dict(z)

def predict_grid(z, grid, action):
    """Returns (head_after, moved)."""
    if action not in DIRS:
        return z["head"], False
    dr, dc = DIRS[action]
    res = _move_cells(grid, tuple(z["head"]), dr, dc)
    if res is None:
        return z["head"], False
    return [res[0], res[1]], True

def on_goal(z):
    goal = z["goal"]
    if goal is None:
        return False
    r, c = z["head"]
    return goal[0] <= r <= goal[1] and goal[2] <= c <= goal[3]

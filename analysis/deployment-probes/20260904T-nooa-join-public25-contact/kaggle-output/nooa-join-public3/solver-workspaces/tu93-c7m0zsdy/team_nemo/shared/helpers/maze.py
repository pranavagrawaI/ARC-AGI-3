"""World model for the grid maze game.

Latent z (dict):
  player : (r, c) tuple — position of the color-4 cell
  goal   : (r0,c0,r1,c1) inclusive bbox of the 14-pad, or None
  walls  : frozenset of (r,c) where color == 2
  player : tuple

Movement rule (hypothesis H1, to confirm): single step in action direction;
blocked if destination cell is a wall (2). Stepping onto the 14-pad completes
the level.
"""
from collections import deque

WALL = 2
FLOOR = 0
PLAYER = 4
PAD_START = 9
PAD_GOAL = 14
BG = 5

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def encode(grid):
    """grid: 64x64 int ndarray -> latent dict z."""
    import numpy as np
    p = np.argwhere(grid == PLAYER)
    player = tuple(int(x) for x in p[0]) if len(p) else None
    gc = np.argwhere(grid == PAD_GOAL)
    if len(gc):
        goal = (int(gc[:, 0].min()), int(gc[:, 1].min()),
                int(gc[:, 0].max()), int(gc[:, 1].max()))
    else:
        goal = None
    walls = frozenset(map(tuple, np.argwhere(grid == WALL).tolist()))
    return {"player": player, "goal": goal, "walls": walls}


def on_goal(z, pos):
    r, c = pos
    if z["goal"] is None:
        return False
    r0, c0, r1, c1 = z["goal"]
    return r0 <= r <= r1 and c0 <= c <= c1


def predict(z, action):
    """Single-step. Returns (z2, expect)."""
    if action not in DIRS or z["player"] is None:
        return z, "no_impact"
    dr, dc = DIRS[action]
    n = (z["player"][0] + dr, z["player"][1] + dc)
    if n in z["walls"]:
        return z, "no_impact"
    z2 = dict(z); z2["player"] = n
    return z2, ("level_up" if on_goal(z2, n) else "move")


def bfs(z, max_steps=512):
    """Shortest action sequence from z['player'] to the goal pad.
    Returns list of action names including the final step onto the pad, or None."""
    start = z["player"]
    if start is None:
        return None
    if on_goal(z, start):
        return []
    parent = {start: (None, None)}
    q = deque([start])
    while q:
        cur = q.popleft()
        for action, (dr, dc) in DIRS.items():
            n = (cur[0] + dr, cur[1] + dc)
            if n in z["walls"] or n in parent:
                continue
            parent[n] = (cur, action)
            if on_goal(z, n):
                path = []
                node = n
                while parent[node][0] is not None:
                    path.append(parent[node][1])
                    node = parent[node][0]
                path.reverse()
                return path
            q.append(n)
    return None

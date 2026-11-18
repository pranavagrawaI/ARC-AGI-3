
import numpy as np
from collections import Counter, deque

# Z_SCHEMA:
#   level: int (current level index)
#   pr, pc: int (player row, col)
#   gr, gc: int (goal row, col)
#   walls: set of (r,c) tuples (immovable obstacle cells)
#
# Colors: 0=empty, 1=player, 5=structure/wall, 8=obstacle, 9=HUD/deco/lines

def _encode_walls(g):
    """Identify wall cells: color 8 (obstacles), and color 5 cells that are
    part of thick structures (not the thin goal beacon column).
    Also: color 9 cells that form horizontal/vertical lines (not HUD row 63)."""
    walls = set()
    # color 8 = solid obstacles
    ys, xs = np.argwhere(g == 8).T
    for r, c in zip(ys, xs):
        walls.add((int(r), int(c)))
    # color 5: identify thick blocks vs thin goal strip
    # The goal beacon is a single-column vertical strip of 5s
    col5 = np.argwhere(g == 5)
    if len(col5) > 0:
        # Count 5s per column, find columns that are "thin" (fewer 5s)
        cnt = Counter(c for r, c in col5)
        # The goal column has a vertical span but few total cells (thin strip)
        # Thick walls have many cells per column
        # Find the column with minimum 5-count among columns with >=5 cells
        thin_cols = {c: n for c, n in cnt.items() if n >= 3}
        if thin_cols:
            # Goal column = the one with smallest count
            goal_col = min(thin_cols, key=lambda c: thin_cols[c])
        else:
            goal_col = None
        # All 5-cells EXCEPT those in the goal column are walls
        for r, c in col5:
            if c != goal_col:
                walls.add((int(r), int(c)))
    # color 9: check for thick horizontal/vertical lines (walls)
    # Row 63 is always HUD (full row of 9s) — skip it
    ys9, xs9 = np.argwhere(g == 9).T
    for r, c in zip(ys9, xs9):
        r, c = int(r), int(c)
        if r == 63:  # HUD
            continue
        # Check if part of a thick horizontal run (>=3 consecutive)
        h_run = 0
        for dc in range(c, -1, -1):
            if g[r, dc] == 9:
                h_run += 1
            else:
                break
        for dc in range(c, 64):
            if g[r, dc] == 9:
                h_run += 1
            else:
                break
        h_run -= 1  # count self twice
        if h_run >= 3:
            walls.add((r, c))
            continue
        v_run = 0
        for dr in range(r, -1, -1):
            if g[dr, c] == 9:
                v_run += 1
            else:
                break
        for dr in range(r, 64):
            if g[dr, c] == 9:
                v_run += 1
            else:
                break
        v_run -= 1
        if v_run >= 3:
            walls.add((r, c))
    return walls

def encode(g):
    g = np.asarray(g)
    # Player = color 1
    ys, xs = np.argwhere(g == 1).T
    pr, pc = int(ys[0]), int(xs[0])
    # Goal = center of the thin 5-column strip
    col5 = np.argwhere(g == 5)
    cnt = Counter(c for r, c in col5)
    thin_cols = {c: n for c, n in cnt.items() if n >= 3}
    goal_col = min(thin_cols, key=lambda c: thin_cols[c]) if thin_cols else None
    gc = goal_col
    gcol_rows = col5[col5[:, 1] == gc][:, 0] if gc is not None else np.array([0])
    gr = int(np.median(gcol_rows))
    walls = _encode_walls(g)
    return {"pr": pr, "pc": pc, "gr": gr, "gc": gc, "walls": walls}

def predict(z, action, level_grid=None):
    """Predict next latent. If action leads to a wall, player doesn't move.
    If player reaches goal, we'd need to detect level_up (not modeled here)."""
    dz = dict(z)
    d = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
    if action in d:
        dr, dc = d[action]
        nr, nc = z["pr"] + dr, z["pc"] + dc
        if 0 <= nr < 64 and 0 <= nc < 64 and (nr, nc) not in z["walls"]:
            dz["pr"], dz["pc"] = nr, nc
        else:
            pass  # blocked, no movement
    return dz

def bfs_path(z):
    """BFS from player to goal avoiding walls. Returns list of actions."""
    start = (z["pr"], z["pc"])
    goal = (z["gr"], z["gc"])
    if start == goal:
        return []
    dirs = [("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1)]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for name, dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 64 and 0 <= nc < 64 and (nr, nc) not in prev and (nr, nc) not in z["walls"]:
                prev[(nr, nc)] = (r, c)
                if (nr, nc) == goal:
                    # Reconstruct
                    path = []
                    cur = (nr, nc)
                    while prev[cur] is not None:
                        pr2, pc2 = prev[cur]
                        for n2, dr2, dc2 in dirs:
                            if pr2 + dr2 == cur[0] and pc2 + dc2 == cur[1]:
                                path.append(n2)
                        cur = (pr2, pc2)
                    return path[::-1]
                q.append((nr, nc))
    return None  # no path found

def dist(z):
    return abs(z["pr"] - z["gr"]) + abs(z["pc"] - z["gc"])

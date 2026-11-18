
import numpy as np
from collections import deque

def _find_clusters(mask):
    visited = np.zeros_like(mask, dtype=bool)
    clusters = []
    rows, cols = np.where(mask)
    for r0, c0 in zip(rows, cols):
        if not visited[r0, c0]:
            queue = deque([(int(r0), int(c0))])
            comp = []
            while queue:
                r, c = queue.popleft()
                if visited[r, c] or not mask[r, c]:
                    continue
                visited[r, c] = True
                comp.append((r, c))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                        if not visited[nr, nc] and mask[nr, nc]:
                            queue.append((nr, nc))
            if comp:
                clusters.append(comp)
    return clusters

def encode(grid):
    g = grid
    f_mask = (g == 15)
    clusters = _find_clusters(f_mask)
    if not clusters:
        return {"player": None, "shape": [int(g.shape[0]), int(g.shape[1])]}
    player = max(clusters, key=len)
    rows = [r for r, c in player]
    cols = [c for r, c in player]
    z = {
        "player_rows": [int(min(rows)), int(max(rows))],
        "player_cols": [int(min(cols)), int(max(cols))],
        "player_center": [float(np.mean(rows)), float(np.mean(cols))],
        "player_count": int(len(player)),
        "shape": [int(g.shape[0]), int(g.shape[1])],
    }
    t_mask = (g == 2)
    t_rows, t_cols = np.where(t_mask)
    if len(t_rows) > 0:
        z["wall_rows"] = [int(t_rows.min()), int(t_rows.max())]
        z["wall_cols"] = [int(t_cols.min()), int(t_cols.max())]
        z["wall_count"] = int(len(t_rows))
    return z

def predict(z, action):
    z2 = dict(z)
    if z.get("player") is None:
        return z2
    r, c = z["player_center"]
    if action == "UP":
        z2["player_rows"] = [z["player_rows"][0] - 1, z["player_rows"][1] - 1]
        z2["player_center"] = [r - 1, c]
    elif action == "DOWN":
        z2["player_rows"] = [z["player_rows"][0] + 1, z["player_rows"][1] + 1]
        z2["player_center"] = [r + 1, c]
    elif action == "LEFT":
        z2["player_cols"] = [z["player_cols"][0] - 1, z["player_cols"][1] - 1]
        z2["player_center"] = [r, c - 1]
    elif action == "RIGHT":
        z2["player_cols"] = [z["player_cols"][0] + 1, z["player_cols"][1] + 1]
        z2["player_center"] = [r, c + 1]
    return z2

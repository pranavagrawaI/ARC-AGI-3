
import numpy as np

Z_SCHEMA = {
    "player_12_pos": "top-left (row, col) of the 2x5 color-12 block (the player)",
    "player_9_pos": "top-left (row, col) of the 3x5 color-9 block (rows player_12+2)",
    "glyph_01_pos": "static 0/1 cross glyph (anchor = topmost 0) — the GOAL",
}

def find_12_block(g):
    rs, cs = np.where(g == 12)
    if len(rs) == 0:
        return None
    return (int(rs.min()), int(cs.min()))

def find_9_block(g):
    rs, cs = np.where(g == 9)
    if len(rs) == 0:
        return None
    # The player 9 block is the 3x5 one near rows 44-55
    mask = (rs >= 44) & (rs <= 55) & (cs >= 30) & (cs <= 55)
    if mask.sum() == 0:
        return None
    return (int(rs[mask].min()), int(cs[mask].min()))

def find_glyph(g):
    rs, cs = np.where(g == 0)
    if len(rs) != 3:
        return None
    for r, c in zip(rs, cs):
        if g[r, c] == 0 and g[r+1, c] == 1 and g[r+1, c+1] == 0 and g[r+1, c-1] == 1 and g[r+2, c] == 1:
            return (int(r), int(c))
    return (int(rs.min()), int(cs.min()))

def encode(g):
    return {
        "player_12_pos": find_12_block(g),
        "player_9_pos": find_9_block(g),
        "glyph_01_pos": find_glyph(g),
    }

def is_walkable(g, r, c):
    """12-block at (r,c) + 9-block at (r+2,c) all on floor (3,0,1,9,12)."""
    if r < 0 or r+5 >= 64 or c < 0 or c+5 >= 64:
        return False
    for rr in range(r, r+5):
        for cc in range(c, c+5):
            if g[rr, cc] not in (3, 12, 9, 0, 1):
                return False
    return True

def predict(z, action, grid=None):
    """Step the player 5 cells. Needs grid to check walls."""
    r, c = z["player_12_pos"]
    dr, dc = {"UP": (-5, 0), "DOWN": (5, 0), "LEFT": (0, -5), "RIGHT": (0, 5)}[action]
    nr, nc = r + dr, c + dc
    if grid is not None and not is_walkable(grid, nr, nc):
        return dict(z)  # blocked, no move
    return {
        "player_12_pos": (nr, nc),
        "player_9_pos": (nr+2, nc),
        "glyph_01_pos": z["glyph_01_pos"],
    }

def bfs_path(g, start, goal):
    """BFS over 5-step grid from start 12-block pos to goal 12-block pos."""
    from collections import deque
    moves = {"UP": (-5,0), "DOWN": (5,0), "LEFT": (0,-5), "RIGHT": (0,5)}
    q = deque([(start, [])])
    visited = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == goal:
            return path
        for name, (dr, dc) in moves.items():
            nr, nc = r+dr, c+dc
            if (nr, nc) not in visited and is_walkable(g, nr, nc):
                visited.add((nr, nc))
                q.append(((nr, nc), path + [name]))
    return None


import numpy as np

# Colors
FLOOR = 3      # walkable floor
FLOOR2 = 5     # walkable (room interior)
WALL = 4       # not walkable (background)
PLAYER = 9     # 9-spiral player sprite
GOAL = 9       # same color as player at goal position
OBSTACLE = 0   # black (0) - possible obstacle
WHITE = 1      # white (1) - part of obstacle sprite
GREEN = 12     # green (12) - goal marker?
STRIPED = 11   # (11) - HUD bar
RED = 8        # (8) - HUD blocks

def _label_9(m9):
    """Connected components of a boolean mask, no scipy. Returns list of (rr, cc) arrays."""
    visited = np.zeros_like(m9, dtype=bool)
    comps = []
    for r in range(64):
        for c in range(64):
            if m9[r,c] and not visited[r,c]:
                # BFS
                stack = [(r,c)]
                comp = []
                visited[r,c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr,cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<64 and 0<=nc<64 and m9[nr,nc] and not visited[nr,nc]:
                            visited[nr,nc] = True
                            stack.append((nr,nc))
                comps.append(np.array(comp))
    return comps

def grid_to_array(grid_rows):
    """Parse 64 hex strings into 64x64 int array."""
    return np.array([[int(ch, 16) for ch in row] for row in grid_rows])

def find_all_9(A):
    """Find all 9-spiral clusters. Returns list of (center_row, center_col, size, floor_neighbors)."""
    m9 = (A == PLAYER)
    if not m9.any():
        return []
    comps = _label_9(m9)
    result = []
    for comp in comps:
        rr, cc = comp[:,0], comp[:,1]
        cr, ccen = int((rr.min()+rr.max())//2), int((cc.min()+cc.max())//2)
        # count floor neighbors
        fn = 0
        for r, c in zip(rr, cc):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<64 and 0<=nc<64 and A[nr,nc] == FLOOR:
                    fn += 1
        result.append((cr, ccen, len(rr), fn))
    return result

def find_player(A):
    """Find the player sprite (9) most adjacent to floor. Returns (row, col) or None."""
    all9 = find_all_9(A)
    if not all9:
        return None
    # Pick the one with most floor neighbors (player is inside the playfield)
    all9.sort(key=lambda x: -x[3])
    return (all9[0][0], all9[0][1])

def encode(A):
    """Encode grid into latent state z."""
    player = find_player(A)
    all9 = find_all_9(A)
    walkable = np.isin(A, [FLOOR, FLOOR2])
    return {
        "player": player,
        "all_9": all9,
        "walkable_rows": walkable.tolist(),
    }

def is_walkable(A, r, c):
    """Check if cell (r,c) is walkable (3 or 5)."""
    if r < 0 or r >= 64 or c < 0 or c >= 64:
        return False
    return A[r, c] in (FLOOR, FLOOR2)

DIRS = {"UP": (-1,0), "DOWN": (1,0), "LEFT": (0,-1), "RIGHT": (0,1)}

def predict_move(z, action, A):
    """Predict player position after one move. Returns (new_player, changed)."""
    if z["player"] is None:
        return z["player"], False
    r, c = z["player"]
    dr, dc = DIRS[action]
    nr, nc = r+dr, c+dc
    if is_walkable(A, nr, nc):
        return (nr, nc), True
    return (r, c), False

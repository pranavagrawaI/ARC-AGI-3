
"""World model for the grid game."""
import numpy as np
from collections import deque

def _parse(grid_rows):
    """Parse grid rows (each a 64-char hex string) into a 64x64 int array."""
    A = np.zeros((64, 64), dtype=np.int32)
    for i in range(64):
        for j in range(64):
            A[i, j] = int(grid_rows[i][j], 16)
    return A

def find_player(A):
    """Find the player's top-left position.
    The player is a 5x5 block with G (12) in top 2 rows and I (9) in bottom 3 rows.
    G cells only appear in the player, so find G first.
    """
    ys, xs = np.where(A == 12)
    if len(ys) == 0:
        return None
    top = int(ys.min())
    left = int(xs.min())
    return (top, left)

def encode(grid_rows):
    """Parse grid into latent state."""
    A = _parse(grid_rows)
    player = find_player(A)

    # Cross position (small cluster of 0 and 1 in the maze, not border)
    p0 = [(int(y), int(x)) for y, x in zip(*np.where(A == 0)) if 10 < y < 55 and 10 < x < 55]
    p1 = [(int(y), int(x)) for y, x in zip(*np.where(A == 1)) if 10 < y < 55 and 10 < x < 55]
    cross = p0 + p1

    # Bottom band
    ys_b, xs_b = np.where(A == 11)
    band_cols_min = int(xs_b.min()) if len(xs_b) > 0 else -1
    band_count = len(ys_b)

    # Walls
    walls = set(zip(*np.where(A == 4)))

    # Items (9 cells NOT part of player)
    if player:
        pr, pc = player
        player_cells = set((r, c) for r in range(pr, pr+5) for c in range(pc, pc+5))
    else:
        player_cells = set()
    item_cells = [(int(y), int(x)) for y, x in zip(*np.where(A == 9)) 
                  if (int(y), int(x)) not in player_cells]

    return {
        'player': player,
        'cross': cross,
        'band_cols_min': band_cols_min,
        'band_count': band_count,
        'walls': walls,
        'items': item_cells,
    }

def can_be_at(walls, top, left):
    """Check if the 5x5 block can be at position (top, left)."""
    if top < 0 or left < 0 or top + 5 > 64 or left + 5 > 64:
        return False
    for r in range(top, top+5):
        for c in range(left, left+5):
            if (r, c) in walls:
                return False
    return True

def predict(z, action):
    """Predict next state after action."""
    if z['player'] is None:
        return z
    r, c = z['player']
    deltas = {'UP': (-5, 0), 'DOWN': (5, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}
    if action not in deltas:
        return z
    dr, dc = deltas[action]
    new_r, new_c = r + dr, c + dc

    if not (0 <= new_r and 0 <= new_c and new_r + 5 <= 64 and new_c + 5 <= 64):
        return z  # can't move off grid

    if not can_be_at(z['walls'], new_r, new_c):
        return z  # blocked by wall

    new_z = dict(z)
    new_z['player'] = (new_r, new_c)
    new_z['band_cols_min'] = z['band_cols_min'] + 2  # band shrinks from left
    new_z['band_count'] = z['band_count'] - 2
    return new_z

def bfs(z, target_pos):
    """BFS to find path from current player position to target."""
    start = z['player']
    if start is None:
        return None
    if start == target_pos:
        return []

    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    deltas = {'UP': (-5, 0), 'DOWN': (5, 0), 'LEFT': (0, -5), 'RIGHT': (0, 5)}
    visited = {start}
    queue = deque([(start, [])])

    while queue:
        (r, c), path = queue.popleft()
        for a in actions:
            dr, dc = deltas[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr and 0 <= nc and nr + 5 <= 64 and nc + 5 <= 64 and (nr, nc) not in visited and can_be_at(z['walls'], nr, nc):
                if (nr, nc) == target_pos:
                    return path + [a]
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [a]))
    return None

def find_path(z, target_top_left):
    """Find path to position where block overlaps target."""
    return bfs(z, target_top_left)

def overlaps_cross(z, top, left):
    """Check if block at (top, left) overlaps with any cross cell."""
    for (cr, cc) in z['cross']:
        if top <= cr < top+5 and left <= cc < left+5:
            return True
    return False

def find_cross_position(z):
    """Find the best position where the block overlaps with the cross."""
    candidates = []
    for r in range(25, 40):
        for c in range(14, 30):
            if can_be_at(z['walls'], r, c):
                if overlaps_cross(z, r, c):
                    # Check if reachable from current position
                    if z['player']:
                        path = bfs(z, (r, c))
                        if path is not None:
                            candidates.append((len(path), r, c))
    if candidates:
        candidates.sort()
        return (candidates[0][1], candidates[0][2])
    return None

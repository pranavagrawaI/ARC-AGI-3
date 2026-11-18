
import numpy as np

# Z_SCHEMA:
# player_r: int 0-63   - player row (center of the diamond white block)
# player_c: int 0-63   - player col (center)
# goal_r: int 0-63     - goal row (center of black rect)
# goal_c: int 0-63     - goal col
# black_remaining: int - number of black cells remaining in the goal region
# level: int           - current level number

Z_SCHEMA = {
    "player_r": ("int", 0, 63),
    "player_c": ("int", 0, 63),
    "goal_r": ("int", 0, 63),
    "goal_c": ("int", 0, 63),
    "black_remaining": ("int", 0, 4096),
    "level": ("int", 0, 10),
}

def _find_components(mask):
    """Find connected components in a binary mask."""
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for r in range(64):
        for c in range(64):
            if mask[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < 64 and 0 <= nc < 64 and mask[nr,nc] and not visited[nr,nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                ys = [x[0] for x in cells]
                xs = [x[1] for x in cells]
                components.append({
                    "r": (min(ys)+max(ys))/2.0, "c": (min(xs)+max(xs))/2.0,
                    "size": len(cells),
                    "r0": min(ys), "r1": max(ys), "c0": min(xs), "c1": max(xs),
                    "h": max(ys)-min(ys)+1, "w": max(xs)-min(xs)+1
                })
    return components

def encode(grid):
    """Extract the latent state from a 64x64 grid array."""
    c15 = (grid == 15)
    white_comps = _find_components(c15)

    c0 = (grid == 0)
    black_comps = _find_components(c0)

    # The player is the medium white component (not the largest, not the smallest)
    # Sort white by size
    white_comps.sort(key=lambda x: x["size"], reverse=True)

    # The player is the one that's NOT at rows 8-12, cols 3-12 (stationary)
    # and NOT inside the goal region
    player = None
    for comp in white_comps:
        # Skip the stationary block at (8-12, 3-12)
        if comp["r0"] == 8 and comp["r1"] == 12 and comp["c0"] == 3 and comp["c1"] == 12:
            continue
        # Skip components that are inside the goal region (rows 34-43)
        if comp["r0"] >= 34 and comp["c0"] >= 27:
            continue
        player = comp
        break
    if player is None and white_comps:
        player = white_comps[0]

    # The goal is the largest black component
    goal = max(black_comps, key=lambda x: x["size"]) if black_comps else None

    # Count total black cells in the goal region
    black_remaining = sum(c["size"] for c in black_comps if c["size"] >= 50)

    z = {
        "player_r": int(round(player["r"])) if player else None,
        "player_c": int(round(player["c"])) if player else None,
        "goal_r": int(round(goal["r"])) if goal else None,
        "goal_c": int(round(goal["c"])) if goal else None,
        "black_remaining": black_remaining,
        "level": 0,
        "_white_comps": white_comps,
        "_black_comps": black_comps,
    }
    return z

def predict(z, action):
    """Predict the next latent state given current state and action."""
    z2 = dict(z)
    r = z["player_r"]
    c = z["player_c"]
    if r is None:
        return z2
    if action == "UP":
        z2["player_r"] = r - 1
    elif action == "DOWN":
        z2["player_r"] = r + 1
    elif action == "LEFT":
        z2["player_c"] = c - 1
    elif action == "RIGHT":
        z2["player_c"] = c + 1
    elif action == "SPACEBAR":
        # SPACEBAR fills 50 black cells with white
        z2["black_remaining"] = max(0, z["black_remaining"] - 50)
    return z2

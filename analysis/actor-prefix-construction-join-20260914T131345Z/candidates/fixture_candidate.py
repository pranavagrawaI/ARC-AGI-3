"""Motion-by-1 with occupancy: stay put when the destination is not floor 4."""

FLOOR = 4
BLOCK = 12


def initialize(observation):
    grid = [list(row) for row in observation["grid"]]
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == BLOCK]
    if not cells:
        raise ValueError("no block")
    block = [min(r for r, _c in cells), min(c for _r, c in cells)]
    return {"grid": grid, "block": block}


def predict(state, action):
    dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}.get(action, (0, 0))
    grid = state["grid"]
    r, c = state["block"]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == FLOOR):
        nr, nc = r, c
    nxt = {"grid": [list(row) for row in grid], "block": [nr, nc]}
    return [{"state": nxt, "commitments": {"block": [nr, nc]}, "declared_dependencies": ["block:state", "grid:observation", "action:driver"]}]


def assimilate(predicted_state, observation, verdict):
    grid = [list(row) for row in observation["grid"]]
    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == BLOCK]
    block = [min(r for r, _c in cells), min(c for _r, c in cells)] if cells else list(predicted_state["block"])
    return {"grid": grid, "block": block}

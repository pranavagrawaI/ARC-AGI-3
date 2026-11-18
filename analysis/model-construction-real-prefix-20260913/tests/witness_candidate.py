"""Host-only evaluator witness. Never include this source in constructor evidence."""


def _find(grid, value):
    return [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell == value]


def _observed(grid):
    block_cells = _find(grid, 12)
    block = [min(r for r, _ in block_cells), min(c for _, c in block_cells)]
    bar_left = next(c for c, cell in enumerate(grid[61]) if cell == 11)
    return block, bar_left


def initialize(observation):
    grid = observation["grid"]
    block, bar_left = _observed(grid)
    terrain = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell != 12 and cell != 9:
                terrain[f"{r},{c}"] = cell
    return {"block": block, "bar_left": bar_left, "terrain": terrain}


def predict(state, action):
    delta = {"UP": (-5, 0), "DOWN": (5, 0), "LEFT": (0, -5), "RIGHT": (0, 5)}.get(action, (0, 0))
    r, c = state["block"]
    nr, nc = r + delta[0], c + delta[1]
    footprint = [(rr, cc) for rr in range(nr, nr + 5) for cc in range(nc, nc + 5)]
    known = [state["terrain"].get(f"{rr},{cc}") for rr, cc in footprint]
    applicable = all(value in (0, 1, 3, None) for value in known)
    successor = dict(state)
    successor["block"] = [nr, nc] if applicable else [r, c]
    successor["bar_left"] = state["bar_left"] + 1
    return [{
        "state": successor,
        "commitments": {"block": successor["block"], "bar_left": successor["bar_left"]},
        "declared_dependencies": ["block", "bar_left", "terrain", "footprint"],
    }]


def assimilate(predicted_state, observation, verdict):
    grid = observation["grid"]
    block, bar_left = _observed(grid)
    revised = dict(predicted_state)
    revised["block"] = block
    revised["bar_left"] = bar_left
    terrain = dict(predicted_state["terrain"])
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell != 12 and cell != 9:
                terrain[f"{r},{c}"] = cell
    revised["terrain"] = terrain
    return revised

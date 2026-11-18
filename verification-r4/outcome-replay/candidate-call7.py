import json, copy

TARGETS = [(61, 13)]

def initialize(observation):
    grid = observation.get("grid", [])
    return {"grid": copy.deepcopy(grid), "targets": list(TARGETS), "step": 0}

def predict(state, action):
    if not action:
        return []
    branch = copy.deepcopy(state)
    grid = branch["grid"]
    commitments = []
    for r, c in branch.get("targets", []):
        try:
            old = grid[r][c]
        except Exception:
            old = None
        # Only a ground commitment: the previously observed transition changed 11 to 3.
        new = 3 if old == 11 else old
        grid[r][c] = new
        commitments.append({"cell": [r, c], "expected": new, "action": str(action)})
    branch["step"] = branch.get("step", 0) + 1
    branch["last_action"] = str(action)
    return [{
        "state": branch,
        "commitments": commitments,
        "declared_dependencies": ["visible grid from observation", "visible action string"]
    }]

def assimilate(predicted_state, observation, verdict):
    state = copy.deepcopy(predicted_state)
    grid = observation.get("grid", [])
    if grid:
        state["grid"] = copy.deepcopy(grid)
    state["last_verdict"] = verdict
    state["assimilated"] = True
    return state
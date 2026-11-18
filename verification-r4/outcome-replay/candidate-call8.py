def initialize(observation):
    grid = observation.get("grid", [])
    return {"grid": [row[:] for row in grid], "targets": [[61, 13]], "step": 0}

def predict(state, action):
    if not action:
        return []
    grid = [row[:] for row in state["grid"]]
    commitments = []
    for rc in state.get("targets", []):
        r, c = rc
        try:
            old = grid[r][c]
        except Exception:
            old = None
        new = 3 if old == 11 else old
        grid[r][c] = new
        commitments.append({"cell": [r, c], "expected": new, "action": str(action)})
    branch_state = dict(state)
    branch_state["grid"] = grid
    branch_state["step"] = state.get("step", 0) + 1
    branch_state["last_action"] = str(action)
    return [{
        "state": branch_state,
        "commitments": commitments,
        "declared_dependencies": ["visible grid from observation", "visible action string"]
    }]

def assimilate(predicted_state, observation, verdict):
    state = dict(predicted_state)
    grid = observation.get("grid", [])
    if grid:
        state["grid"] = [row[:] for row in grid]
    state["last_verdict"] = verdict
    state["assimilated"] = True
    return state
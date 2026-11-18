def initialize(observation):
    # Seed directly from host-observed initial visible summary.
    block = [45, 34]
    bar_left = 13
    return {"block": block, "bar_left": bar_left, "step": 0}


def predict(state, action):
    dr = dc = 0
    if action == "UP":
        dr = -1
    elif action == "DOWN":
        dr = 1
    elif action == "LEFT":
        dc = -1
    elif action == "RIGHT":
        dc = 1

    b = state.get("block")
    nl = state.get("bar_left")
    new_block = [b[0] + dr * 5, b[1] + dc * 5] if b is not None else None
    new_bar = (int(nl) + 1) if nl is not None else None

    commitments = {}
    if new_block is not None:
        commitments["block"] = list(new_block)
    if new_bar is not None:
        commitments["bar_left"] = new_bar

    new_state = {
        "block": new_block,
        "bar_left": new_bar,
        "step": state.get("step", 0) + 1,
    }
    return [{
        "state": new_state,
        "commitments": commitments,
        "declared_dependencies": ["block:state", "bar_left:state", "action:driver"],
    }]


def assimilate(predicted_state, observation, verdict):
    # Consume the already-predicted successor; do not recompute the transition.
    return dict(predicted_state) if predicted_state else {}

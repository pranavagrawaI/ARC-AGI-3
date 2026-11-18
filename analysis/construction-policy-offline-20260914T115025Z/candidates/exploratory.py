"""Playable fallback: store the grid; do not invent dynamics."""


def initialize(observation):
    return {"grid": [list(row) for row in observation["grid"]]}


def predict(state, action):
    nxt = {"grid": [list(row) for row in state["grid"]]}
    return [{"state": nxt, "commitments": {}, "declared_dependencies": ["unresolved_dynamics"]}]


def assimilate(predicted_state, observation, verdict):
    return {"grid": [list(row) for row in observation["grid"]]}

"""Capability and instruction assignments for the factorial continuation test."""
VARIANTS = ('control', 'planner', 'probing', 'both')
SEARCH = "\nUse bounded_search(initial_state, successors, state_key=..., is_goal=..., heuristic=None, max_depth=None, max_nodes=100000, max_expanded=50000, max_seconds=10.0) for search over your own learned model. successors(state) yields (action, next_state); callbacks must be pure. FOUND is only model-valid. HORIZON_EXHAUSTED, GRAPH_EXHAUSTED, RESOURCE_LIMIT and CALLBACK_ERROR are distinct; a depth/time/state cutoff does not refute the goal or rules. Inspect and test returned actions against the environment. Optional nonnegative finite heuristics guide search but may not produce shortest routes. Search consumes tool time; reserve time for checking and execution.\n"
PROBE = "\nBefore relying on an untested relevant action family, consider one bounded prospective test with probe_transition(action, observe=..., predict=..., state_key=..., name='fresh_name'). observe(frame) extracts JSON-compatible state; predict(before, action) predicts its successor; state_key(state) defines the comparison. Pure callbacks cannot act. The helper saves the prediction before one real action, then saves MATCH/MISMATCH or refusal/terminal/unknown status. Timeouts can leave PREDICTED or ACTED: inspect saved data before proceeding. A match concerns only declared state variables. Probe relevant uncovered controls/interactions when their expected information warrants spending an action and time; do not repeatedly retest supported rules.\n"


def guidance(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant: ' + variant)
    return (SEARCH if variant in ('planner', 'both') else '') + (PROBE if variant in ('probing', 'both') else '')


def wire_messages(messages):
    """Project only known harness/response metadata out of request messages."""
    import copy
    output=copy.deepcopy(messages)
    for message in output:
        message.pop('_arc3_control',None)
        for tool_call in message.get('tool_calls') or []:
            tool_call.pop('index',None)
    return output

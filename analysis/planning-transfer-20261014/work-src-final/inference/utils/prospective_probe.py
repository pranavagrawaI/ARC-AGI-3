"""Predict, persist, execute one action, then compare declared state.

The host-owned record callback must acknowledge storage before execution.
A match concerns only the supplied observation/key, not complete game dynamics.
Interrupted operations retain the latest acknowledged stage; never retry them
automatically. Observation and prediction must be pure JSON-compatible states.
"""
import copy as _probe_copy


def prospective_probe(action, *, observe, predict, state_key, execute, record):
    """Callbacks: observe(), predict(before, action), state_key(state).

execute(action) returns the authoritative action receipt. record(dict) persists
an independent snapshot. Exceptions propagate; the persisted stage distinguishes
an unexecuted prediction from an acted-but-unobserved transition.
"""
    for callback in (observe, predict, state_key, execute, record):
        if not callable(callback):
            raise TypeError('Probe callbacks must be callable')
    action = _probe_copy.deepcopy(action)
    before = _probe_copy.deepcopy(observe())
    predicted = _probe_copy.deepcopy(predict(_probe_copy.deepcopy(before), _probe_copy.deepcopy(action)))
    if predicted is None:
        raise ValueError('Prediction is required before executing a probe')
    predicted_key = _probe_copy.deepcopy(state_key(_probe_copy.deepcopy(predicted)))
    hash(predicted_key)
    row = {'status': 'PREDICTED', 'action': action, 'before': before, 'predicted': predicted,
           'actual': None, 'matched': None, 'receipt': None}
    record(_probe_copy.deepcopy(row))
    receipt = _probe_copy.deepcopy(execute(_probe_copy.deepcopy(action)))
    row.update(status='ACTED', receipt=receipt)
    record(_probe_copy.deepcopy(row))
    if not isinstance(receipt, dict) or type(receipt.get('executed_count')) is not int:
        row['status'] = 'ACTION_UNCONFIRMED'
    elif receipt['executed_count'] != 1:
        row['status'] = 'REFUSED' if receipt['executed_count'] == 0 else 'ACTION_UNCONFIRMED'
    elif any(receipt.get(flag) for flag in ('game_over', 'level_completed', 'run_complete', 'done')):
        # A reset/new-level board cannot validate the old model's successor.
        row['status'] = 'TERMINAL'
    else:
        row['actual'] = _probe_copy.deepcopy(observe())
        actual_key = state_key(_probe_copy.deepcopy(row['actual']))
        hash(actual_key)
        row['matched'] = bool(predicted_key == actual_key)
        row['status'] = 'MATCH' if row['matched'] else 'MISMATCH'
    record(_probe_copy.deepcopy(row))
    return row

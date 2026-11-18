"""Budget stops cannot forgive missing, late, failed or unfinished requests."""
import importlib.util
import json
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parent.parent / 'submission-runtime-20260912-v7'


def module(name):
    spec = importlib.util.spec_from_file_location(name, RUNTIME / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('defect', ['unfinished', 'failed', 'late', 'missing_call_record'])
def test_budget_stop_does_not_excuse_bad_lifecycle(tmp_path, defect):
    ipc = tmp_path/'ipc'
    ipc.mkdir()
    (ipc/'budget_exhausted.json').write_text(json.dumps(dict(
        reason='model_admission_budget', at_monotonic=3, admission_deadline=2, active_requests=0)))
    events = [dict(request_id=1, event='admitted', at_monotonic=1, admission_deadline=2)]
    if defect != 'unfinished':
        events.append(dict(request_id=1, event='failed' if defect == 'failed' else 'completed'))
    if defect == 'late':
        events[0]['at_monotonic'] = 3
    (ipc/'request_lifecycle.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    if defect != 'missing_call_record':
        (ipc/'model_calls.jsonl').write_text(json.dumps(dict(turn=0))+'\n')
    with pytest.raises(RuntimeError):
        module('nooa_public_probe')._model_evidence(tmp_path, 'fixture')


@pytest.mark.parametrize('acknowledged', [False, True])
def test_outer_zero_calls_require_explicit_budget_stop(acknowledged):
    result = dict(games=[dict(game_id='fixture', stop_reason='model_admission_budget' if acknowledged else None)],
        solver_sessions=dict(schema='nooa-join-sessions-v4', game_count=1, total_actor_calls=0,
            games=dict(fixture=dict(model_evidence=dict(actor_call_count=0, actor_calls_by_sequence={})))))
    outer = module('exact_request_outer_runtime')
    if acknowledged:
        assert outer.expected_launcher_solver_requests(result) == 0
    else:
        with pytest.raises(RuntimeError):
            outer.expected_launcher_solver_requests(result)

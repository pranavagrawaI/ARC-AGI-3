"""Recheck decision-analysis claims using preserved public observations only.

No game launch, model call or environment-source inspection. The one imported
generated helper was inspected before execution and is hash-bound to live checks.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent/'raw/qualification'
rows = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
allocations = rows(ROOT/'budget-allocations.jsonl')
facts = {}
for session in sorted((ROOT/'solver-workspaces').iterdir()):
    if not session.is_dir():
        continue
    ipc = session/'ipc'
    calls = rows(ipc/'model_calls.jsonl')
    transitions = rows(ipc/'transitions.jsonl')
    timing = rows(ipc/'execution_timing.jsonl')
    checks = rows(ipc/'step_checks.jsonl')
    feedback = rows(ipc/'model_feedback.jsonl')
    allocation = next(a for a in allocations if session.name.startswith(a['game_id']))
    start = allocation['admission_deadline']-600
    game = session.name.split('-')[0]
    data = dict(session=session.name, calls=len(calls), actions=len(transitions),
        calls_at_step_zero=sum(c['step'] == 0 for c in calls),
        first_action_seconds=timing[0]['observed_at_monotonic']-start,
        last_action_seconds=timing[-1]['observed_at_monotonic']-start,
        calls_by_observed_step=dict(Counter(c['step'] for c in calls)),
        action_counts=dict(Counter(t['action'] for t in transitions)),
        step_check_statuses=dict(Counter(c['status'] for c in checks)),
        changes=[])
    assert all(not c['reasoning_present'] and not c['reasoning_tokens'] for c in calls)
    for t in transitions:
        before, after = np.array(t['before']['grid']), np.array(t['after']['grid'])
        indices = np.argwhere(before != after)
        data['changes'].append(dict(step=t['step'], action=t['action'],
            changed_cells=len(indices), outside_bottom_row=int((before[:63] != after[:63]).sum()),
            cells=[[int(r), int(c), int(before[r,c]), int(after[r,c])] for r,c in indices] if len(indices) <= 2 else None))
    if game == 's5i5':
        assert len(transitions) == 47
        assert all(c['outside_bottom_row'] == 0 for c in data['changes'])
        assert data['action_counts']['CLICK 24 41'] == 45
        data['final_bottom_row_color4_count'] = int((np.array(transitions[-1]['after']['grid'])[63] == 4).sum())
        assert data['final_bottom_row_color4_count'] == 60
        data['checked_helper_hashes'] = sorted({f['model_sha256'] for f in feedback if f.get('model_sha256')})
        assert len(data['checked_helper_hashes']) == 1
    if game == 'ls20':
        later = [c for c in calls if c['step'] == 6]
        data['calls_after_final_action'] = len(later)
        data['call_seconds_after_final_action'] = sum(c['wall_seconds'] for c in later)
        assert len(later) == 8
    if game == 'cd82':
        helper = session/'team_nemo/shared/helpers/world_model.py'
        digest = hashlib.sha256(helper.read_bytes()).hexdigest()
        assert digest == '196758f8c4d07d1708daeb9fc653c7b3dad91609f158878b524abd3f0626ee39'
        spec = importlib.util.spec_from_file_location('saved_helper', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for t in transitions:
            encoded = module.encode(np.array(t['before']['grid']))
            assert 'player' not in encoded
            for action in ('UP', 'DOWN', 'LEFT', 'RIGHT'):
                assert module.predict(encoded, action) == encoded
        reproduced = 0
        for check in checks:
            if check['model_sha256'] != digest:
                continue
            t = next(t for t in transitions if t['step'] == check['step'])
            assert module.predict(module.encode(np.array(t['before']['grid'])), t['action']) == check['expected']
            assert module.encode(np.array(t['after']['grid'])) == check['observed']
            reproduced += 1
        assert reproduced == 8
        for step, col in [(4,61), (9,58), (11,57), (12,56)]:
            change = next(c for c in data['changes'] if c['step'] == step)
            assert change['cells'] == [[63,col,4,5]]
            assert next(c for c in checks if c['step'] == step)['status'] == 'matched'
        data['helper_replay'] = dict(sha256=digest, saved_pre_states=15,
            directional_predictions_tested=60, all_predict_unchanged=True,
            live_checks_reproduced=reproduced)
    facts[game] = data
print(json.dumps(dict(scope='saved-trace measurements and helper replay, not a new game experiment',
                      status='passed', games=facts), indent=2))

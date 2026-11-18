"""CPU-only audit of the reviewed candidate and its parent's public trace."""
import collections
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / 'deployment/kaggle/q6-summary-boundary-20260906'
RAW = ROOT / ('analysis/deployment-probes/20260906-q6-prediction-feedback/'
              'long-run/raw/nooa-join-public3/solver-workspaces/ls20-w_gx27z4/ipc')


def read(name):
    return [json.loads(line) for line in (RAW / name).read_text().splitlines() if line]


def module(name):
    spec = importlib.util.spec_from_file_location(name, CANDIDATE / 'chassis' / f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    prediction = module('prediction_feedback')
    host = module('host_contract')
    calls, transitions = read('model_calls.jsonl'), read('transitions.jsonl')
    predictions, feedback = read('model_predictions.jsonl'), read('model_feedback.jsonl')
    batch = next(row for row in predictions if row['turn'] == 5)
    verdict = next(row for row in feedback if row['turn'] == 5)
    mismatch = next(row['step'] for row in verdict['differences'] if row['fields'])
    batch_transitions = [row for row in transitions if row['turn'] == 5]
    observed = []
    for row in batch_transitions:
        before, after = row['before'], row['after']
        record = host.ObservationRecord()
        label = record.classify(before['grid'], after['grid'])
        outcome = record.outcome(label, after['state'], after['levels_completed'] > before['levels_completed'])
        observed.append({'step': row['step'], 'action': row['action'], 'host_outcome': outcome,
                         'changed_pixels': sum(a != b for ra, rb in zip(before['grid'], after['grid']) for a, b in zip(ra, rb))})
    assert batch['actions'] == ['UP', 'LEFT', 'LEFT', 'LEFT']
    assert mismatch == 7
    assert [row['step'] for row in batch_transitions if row['step'] > mismatch] == [8, 9]
    assert all(row['host_outcome'] == 'move' for row in observed)

    # Synthetic counterexample: this is a limit of the consistency gate, not
    # evidence that the live actor deliberately supplied a constant encoder.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        helpers = directory / 'helpers'
        helpers.mkdir()
        (helpers / 'world_model.py').write_text('def encode(grid):\n    return {"x": 0}\ndef predict(z, action):\n    return {"x": 0}\n')
        checker = prediction.PredictionFeedback(directory, helpers, lambda source: None)
        before = [[0] * 64 for _ in range(64)]
        after = [[1] * 64 for _ in range(64)]
        state = {'turn': 0, 'step': 0, 'levels_completed': 0, 'grid': before}
        checker.commit(checker.prepare(state, ['UP']))
        prediction.write_row(directory / 'transitions.jsonl', {'turn': 0, 'step': 1, 'action': 'UP', 'after': {'grid': after, 'levels_completed': 0}})
        result = checker.observe()
        proposal = checker.prepare({**state, 'turn': 1, 'step': 1, 'grid': after}, ['LEFT'] * 20)
        error = checker.batch_error(proposal, result)
        assert result['status'] == 'matched' and error is None
        synthetic = {'all_4096_pixels_changed': True, 'feedback_status': result['status'],
                     'new_action_batch_length': 20, 'model_gate_error': error,
                     'scope': 'model consistency gate only; other submit gates not invoked; synthetic helper uses no source validation'}

    output = {'candidate': str(CANDIDATE.relative_to(ROOT)),
              'candidate_contract': json.loads((CANDIDATE / 'CHASSIS.json').read_text())['contract_sha256'],
              'calls': len(calls), 'executed_actions': len(transitions),
              'model_wall_seconds': sum(row['wall_seconds'] for row in calls),
              'calls_per_turn': dict(collections.Counter(row['turn'] for row in calls)),
              'batch': {'turn': 5, 'first_prediction_mismatch_step': mismatch, 'transitions': observed,
                        'executed_after_first_mismatch': 2},
              'synthetic_encoder_counterexample': synthetic,
              'raw_sha256': {name: hashlib.sha256((RAW / name).read_bytes()).hexdigest() for name in
                             ['model_calls.jsonl', 'transitions.jsonl', 'model_predictions.jsonl', 'model_feedback.jsonl']},
              'limits': 'Parent run aborted at summarization; descriptive trace and CPU counterexample, no gameplay effect estimate.'}
    target = Path(__file__).with_name('EVIDENCE.json')
    target.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

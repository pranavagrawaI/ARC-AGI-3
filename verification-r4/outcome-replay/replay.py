"""Read-only R4 diagnostic. Run from repo with mapped-root command in REPORT.md.
Untrusted source is never executed outside the frozen SealedBoundary.
"""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PKG = REPO / 'deployment/kaggle/actor-critic-construction-kaggle-r4'
RAW = REPO / 'analysis/actor-critic-construction-kaggle-20260914T195518Z/runtime-evidence-exact-request/model-runtime/solver-workspaces/raw'
sys.path.insert(0, str(PKG))
from evaluate_visible import evaluate_visible_abi
from integration_boundary import SealedBoundary
from prefix import ActorPrefixCase
from constructor import _validate_work_item

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

requests = [json.loads(line) for line in (RAW / 'logical-requests.jsonl').read_text().splitlines()]
r7, r8 = [next(r for r in requests if r['call'] == n) for n in (7, 8)]
packet = next(json.loads(m['content'])['visible_case'] for m in r7['request']['messages'] if m['role'] == 'user' and 'visible_case' in json.loads(m['content']))
assert packet['frontier'] == 1
assert packet['visible_transition_summaries'] == [{'step': 1, 'action': 'RIGHT'}]
row = dict(packet['visible_transition_summaries'][0], before=packet['initial_observation'], after=packet['current_observation'])
case = ActorPrefixCase(visible=(row,), frontier=1)
save('reconstructed-one-visible-step.json', {'provenance': 'DIAGNOSTIC reconstruction from call7 compact visible_case; NOT downloaded /evidence/case.json', 'visible': [row]})
save('captured-constructor-messages.json', {'call7': r7['request']['messages'], 'call8': r8['request']['messages']})
remote = [json.loads(m['content']) for m in r8['request']['messages'] if m['role'] == 'tool']
save('remote-first-feedback.json', remote)
results = {'scope': 'LOCAL SEALED DIAGNOSTIC, not remote second verdict', 'python': sys.version, 'submissions': []}
for n in (7, 8):
    response = json.loads((RAW / 'raw-http' / f'{n:02}.body').read_text())
    tool = response['choices'][0]['message']['tool_calls'][0]
    assert tool['function']['name'] == 'submit_candidate'
    args = json.loads(tool['function']['arguments'])
    source = args['candidate_source']
    (OUT / f'candidate-call{n}.py').write_text(source)
    save(f'submission-call{n}.json', args)
    item = {'call': n, 'source_sha256': hashlib.sha256(source.encode()).hexdigest()}
    try:
        _validate_work_item(args['work_item'], 1)
        item['work_item_validation'] = 'passed'
    except Exception as exc:
        item['work_item_validation'] = f'{type(exc).__name__}: {exc}'
    item['evaluation'] = evaluate_visible_abi(source, case)
    # Observe exact RPC outputs independently; no bypass of evaluator rejection.
    diagnostic = {}
    try:
        state = SealedBoundary().invoke(source, 'initialize', [{'grid': row['before']['grid']}])
        diagnostic['initialize'] = state
        prediction = SealedBoundary().invoke(source, 'predict', [state, row['action']])
        diagnostic['predict'] = prediction
        diagnostic['assimilate_off_path_only'] = SealedBoundary().invoke(source, 'assimilate', [prediction[0]['state'], {'grid': row['after']['grid']}, {'status': 'unscored'}])
    except Exception as exc:
        diagnostic['error'] = f'{type(exc).__name__}: {exc}'
    save(f'rpc-diagnostic-call{n}.json', diagnostic)
    item['rpc_diagnostic_file'] = f'rpc-diagnostic-call{n}.json'
    results['submissions'].append(item)
# Separate synthetic import-availability probe, not a repair or candidate.
probe = '''def initialize(observation):
    result = {}
    for name in ['json', 'copy', 'math', 'random', 'collections', 'os']:
        try:
            __import__(name)
            result[name] = 'available'
        except Exception as error:
            result[name] = type(error).__name__ + ': ' + str(error)
    return result
def predict(state, action):
    return []
def assimilate(state, observation, verdict):
    return state
'''
(OUT / 'import-probe.py').write_text(probe)
results['separate_synthetic_import_probe'] = SealedBoundary().invoke(probe, 'initialize', [{}])
deltas = [{'cell': [r, c], 'before': a, 'after': b} for r, (aa, bb) in enumerate(zip(row['before']['grid'], row['after']['grid'])) for c, (a, b) in enumerate(zip(aa, bb)) if a != b]
results['observed_grid_delta_count'] = len(deltas)
save('observed-grid-deltas.json', deltas)
save('results.json', results)
paths = [RAW / 'logical-requests.jsonl', RAW / 'raw-http/07.body', RAW / 'raw-http/08.body'] + list(PKG.glob('*.py'))
save('input-hashes.json', {str(p.relative_to(REPO)): sha(p) for p in paths})
assert results['submissions'][0]['source_sha256'] == remote[0]['visible_evaluation']['model_sha256']
assert "No module named 'copy'" in results['submissions'][0]['evaluation']['visible']['error']
assert results['submissions'][1]['evaluation']['visible']['error'] == 'ValueError: commitments and declared_dependencies have bad types'
assert [r['method'] for r in results['submissions'][1]['evaluation']['rpc_methods']] == ['initialize', 'predict']
print(json.dumps(results, indent=2))

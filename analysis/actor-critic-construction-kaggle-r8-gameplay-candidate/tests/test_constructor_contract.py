"""R5 outbound disclosure and real sealed ABI regression tests; no model calls."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys

import pytest
from test_runtime import PACKAGE, ROOT, WORK

VALID = '''
def initialize(observation):
    return {"grid": [row[:] for row in observation["grid"]]}
def predict(state, action):
    return [{"state": {"grid": state["grid"], "action": action},
             "commitments": {}, "declared_dependencies": []}]
def assimilate(predicted_state, observation, verdict):
    assert verdict == {"status": "unscored"}
    assert predicted_state["action"] == "TEST"
    return {"grid": observation["grid"]}
'''
CASE = {"step": 1, "action": "TEST", "before": {"grid": [[0, 1], [2, 3]]}, "after": {"grid": [[9, 8], [7, 6]]}}


def isolated(code, payload=None):
    proc = subprocess.run(
        [sys.executable, '-I', '-S', '-B', '-c',
         f'import sys,json;sys.path.insert(0,{str(PACKAGE)!r});\n' + code],
        input=json.dumps(payload), capture_output=True, text=True, timeout=40,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def evaluate(source):
    return isolated('''from evaluate_visible import evaluate_visible_abi
from prefix import ActorPrefixCase
p=json.load(sys.stdin)
print(json.dumps(evaluate_visible_abi(p['source'],ActorPrefixCase((p['row'],),1))))
''', {"source": source, "row": CASE})


def assert_outbound_contract(body):
    """Also called on actual local HTTP server bytes, after relay projection."""
    system = next(m['content'] for m in body['messages'] if m['role'] == 'system')
    for text in (
        'import copy is unavailable', 'NOT THE FULL STANDARD LIBRARY',
        'initialize(observation) -> dict', 'exactly ONE dict branch',
        'commitments must be a dict, NOT a list', 'defaults to []',
        'does not validate its element types', '{"status":"unscored"}',
        'NOT established', '200,000', '32,000', '268435456',
        '15 seconds', '3 seconds', '8-second', '16,000', '4,000',
        '1,000,000', '20,000', 'RLIMIT_CPU=2', 'RLIMIT_FSIZE=65536',
        'RLIMIT_CORE=0', 'RLIMIT_NOFILE=32', 'RLIMIT_NPROC=0',
        'fresh sealed process', 'stdin\nis EOF', 'starts in COMMIT phase',
        'no inspection is available', 'only initial/current grids',
        'NOT billed cost', '2048', 'No new request at <=20 seconds',
        'only successful ABI replay', 'No suffix is assessed',
    ):
        assert text in system, text
    for stale in ('singleton commitments', 'block=[row,column]', 'bar_left',
                  'joint commitments are host-enforced', 'all-unknown predictions receive no credit'):
        assert stale not in system
    packets = []
    for m in body['messages']:
        if m['role'] == 'user':
            try:
                packets.append(json.loads(m['content']))
            except json.JSONDecodeError:
                pass
    contract = next(p['runtime_contract'] for p in packets if 'runtime_contract' in p)
    assert contract['constructor_limits'] == {
        'max_model_requests': 2, 'max_inspections': 10, 'max_submissions': 4,
        'max_completion_tokens': 16384, 'max_prompt_tokens': 300000,
        'wall_seconds': 45, 'reserve_requests': 2,
        'reserve_completion_tokens': 4096, 'max_dollars': .5,
    }
    assert contract['effective_candidate_source_max_bytes'] + contract['wrapped_source_overhead_bytes'] == 200000
    budget = [p['budget'] for p in packets if 'budget' in p][-1]
    assert budget['phase'] == 'commit' and budget['inspections_remaining'] == 0
    assert budget['requests_remaining'] in (1, 2)
    assert 'execute_python' not in {t['function']['name'] for t in body['tools']}


def test_actual_bound_constructor_outbound_and_repair():
    result = isolated('''from host import bound_constructor, ConstructionLimits
from prefix import ActorPrefixCase
p=json.load(sys.stdin); bodies=[]
def transport(body):
    bodies.append(body)
    source=p['valid'] if len(bodies)==2 else 'import copy\\n'+p['valid']
    return {'usage':{'prompt_tokens':100,'completion_tokens':30},'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':str(len(bodies)),'type':'function','function':{'name':'submit_candidate','arguments':json.dumps({'work_item':p['work'],'candidate_source':source})}}]}}]}
import constructor
old=constructor.SYSTEM; old_evaluate=constructor.evaluate_candidate
with bound_constructor() as bound:
    result=bound.run_construction(ActorPrefixCase((p['row'],),1),transport=transport,limits=ConstructionLimits(wall_seconds=45,max_model_requests=2))
assert constructor.SYSTEM == old and constructor.evaluate_candidate is old_evaluate
print(json.dumps({'bodies':bodies,'status':result.status,'evaluation':result.evaluation}))
''', {'row': CASE, 'valid': VALID, 'work': WORK})
    assert len(result['bodies']) == 2
    for body in result['bodies']:
        assert_outbound_contract(body)
    feedback = [json.loads(m['content']) for m in result['bodies'][1]['messages'] if m['role'] == 'tool']
    assert "No module named 'copy'" in feedback[0]['visible_evaluation']['visible']['error']
    assert result['status'] == 'CANDIDATE_READY'
    assert result['evaluation']['evidence_claim'] == 'ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL'


@pytest.mark.parametrize('old,new,error', [
    ('return {"grid": [row[:] for row in observation["grid"]]}', 'return []', 'initialize must return an object'),
    ('"commitments": {}', '"commitments": []', 'bad types'),
    ('"declared_dependencies": []', '"declared_dependencies": {}', 'bad types'),
    ('return [{"state":', 'return [{"wrong":', 'must contain state and commitments'),
    ('return [{"state": {"grid": state["grid"], "action": action},\n             "commitments": {}, "declared_dependencies": []}]', 'return []', 'nonempty list'),
    ('return [{"state": {"grid": state["grid"], "action": action},\n             "commitments": {}, "declared_dependencies": []}]', 'return [{"state": {}, "commitments": {}}] * 2', 'multiple branches'),
    ('return {"grid": observation["grid"]}', 'return []', 'assimilate must return an object'),
    ('return {"grid": observation["grid"]}', 'return {"x": float("nan")}', 'failed closed'),
    ('return {"grid": observation["grid"]}', 'return {"x": {1, 2}}', 'failed closed'),
    ('return {"grid": observation["grid"]}', 'return {"x": "a" * 200001}', 'JSON size limit'),
    ('return {"grid": observation["grid"]}', 'print("extra"); return {}', 'JSONDecodeError'),
])
def test_rejected_shapes_through_actual_sealed_evaluator(old, new, error):
    assert old in VALID
    result = evaluate(VALID.replace(old, new))
    assert result['visible']['status'] == 'failed', result
    assert error in result['visible']['error'], result


@pytest.mark.parametrize('change', ['valid', 'omitted_dependencies', 'unchecked_dependency_elements', 'scalar_branch_state'])
def test_valid_generic_shapes_do_not_establish_semantic_support(change):
    source = VALID
    if change == 'omitted_dependencies':
        source = source.replace(', "declared_dependencies": []', '')
    if change == 'unchecked_dependency_elements':
        source = source.replace('"declared_dependencies": []', '"declared_dependencies": [1, null]'.replace('null', 'None'))
    if change == 'scalar_branch_state':
        source = source.replace('{"grid": state["grid"], "action": action}', '7').replace('assert predicted_state["action"] == "TEST"', 'assert predicted_state == 7')
    result = evaluate(source)
    assert result['visible']['status'] == 'passed', result
    assert result['coverage'] == {'required': [], 'missing_required': []}
    assert result['suffix']['status'] == 'not_run'
    assert result['dependencies']['static_completeness_established'] is False
    assert result['evidence_claim'] == 'ABI_VISIBLE_REPLAY_ONLY_NOT_SUPPORTED_MODEL'


def test_import_support_and_isolation_in_both_sealed_processes():
    probe = '''import os
result = {}
for name in ('json', 'collections', 're', 'copy', 'numpy'):
    try:
        __import__(name); result[name] = 'available'
    except ImportError:
        result[name] = 'unavailable'
result['stdin'] = os.read(0,10).decode()
result['environment'] = dict(os.environ)
try:
    open('/evidence/case.json').read(); result['prefix'] = True
except OSError:
    result['prefix'] = False
'''
    result = isolated('''from integration_boundary import SealedBoundary
p=json.load(sys.stdin)
src=p['probe']+'\\ndef initialize(observation): return result\\ndef predict(state,action): return []\\ndef assimilate(state,observation,verdict): return state\\n'
b=SealedBoundary()
a=b.invoke(src,'initialize',[{}])
i=b.inspect(p['probe']+'\\nimport json;print(json.dumps(result))',{'visible':['TEST']})
print(json.dumps({'candidate':a,'inspection':json.loads(i['stdout'])}))
''', {'probe': probe})
    for mode, data in result.items():
        assert {k: data[k] for k in ('json', 'collections', 're', 'copy', 'numpy')} == {
            'json': 'available', 'collections': 'available', 're': 'available',
            'copy': 'unavailable', 'numpy': 'unavailable',
        }
        assert data['prefix'] == (mode == 'inspection')
        assert data['stdin'] == ''
        assert data['environment'] == {'LANG': 'C', 'PYTHONDONTWRITEBYTECODE': '1'}


@pytest.mark.parametrize('call,error', [(7, "No module named 'copy'"), (8, 'commitments and declared_dependencies have bad types')])
def test_exact_r4_captured_sources_still_rejected(call, error):
    raw = ROOT / 'analysis/actor-critic-construction-kaggle-20260914T195518Z/runtime-evidence-exact-request/model-runtime/solver-workspaces/raw'
    response = json.loads((raw / 'raw-http' / f'{call:02}.body').read_text())
    source = json.loads(response['choices'][0]['message']['tool_calls'][0]['function']['arguments'])['candidate_source']
    requests = [json.loads(line) for line in (raw / 'logical-requests.jsonl').read_text().splitlines()]
    packet = next(json.loads(m['content'])['visible_case'] for r in requests if r['call'] == 7 for m in r['request']['messages'] if m['role'] == 'user' and 'visible_case' in json.loads(m['content']))
    row = dict(packet['visible_transition_summaries'][0], before=packet['initial_observation'], after=packet['current_observation'])
    result = isolated('''from evaluate_visible import evaluate_visible_abi
from prefix import ActorPrefixCase
p=json.load(sys.stdin)
print(json.dumps(evaluate_visible_abi(p['source'],ActorPrefixCase((p['row'],),1))))
''', {'source': source, 'row': row})
    assert result['model_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    assert result['visible']['status'] == 'failed'
    assert error in result['visible']['error']


def test_runtime_limits_match_disclosed_sources_and_json_admission():
    result = isolated('''from constructor import ConstructionLimits
from constructor_contract import SYSTEM, runtime_contract
from integration_boundary import SealedBoundary, _wrapped
from containment_base import RUNTIME_FILES, RUNTIME_DIRS
p=json.load(sys.stdin)
contract=runtime_contract(ConstructionLimits(wall_seconds=45,max_model_requests=2))
for name in (*RUNTIME_FILES,*RUNTIME_DIRS):
    assert name in SYSTEM, name
assert contract['wrapped_source_overhead_bytes'] == len(_wrapped('').encode())
source=p['source']; errors={}
for label,src,args in [('source',source+'#' * 200000,[{}]),('args',source,[{'large':'x' * 200000}])]:
    try:
        SealedBoundary().invoke(src,'initialize',args)
    except ValueError as error:
        errors[label]=str(error)
assert errors == {'source':'bad source','args':'bad args'},errors
print(json.dumps(contract))
''', {'source': VALID})
    assert result['package_envelope']['application_requests'] == 23
    assert result['package_envelope']['constructor_hard_seconds'] == 50


def test_globals_restart_and_candidate_json_roundtrip():
    source = '''counter=0
def initialize(observation):
    global counter
    counter+=1
    return {'counter':counter,'tuple':(1,2)}
def predict(state,action): return []
def assimilate(state,observation,verdict): return state
'''
    result = isolated('''from integration_boundary import SealedBoundary
p=json.load(sys.stdin); b=SealedBoundary()
print(json.dumps([b.invoke(p['source'],'initialize',[{}]) for _ in range(2)]))
''', {'source': source})
    assert result == [{'counter': 1, 'tuple': [1, 2]}] * 2


def test_r6_only_host_owned_freeze_use_and_inherited_envelope():
    prior = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r5'
    frozen = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r6'
    old = json.loads((prior / 'ASSET-HASHES.json').read_text())
    new = json.loads((frozen / 'ASSET-HASHES.json').read_text())
    assert new.keys() == old.keys()
    assert {n for n in old if old[n] != new[n]} == {'live_agent.py', 'EXPERIMENT-CARD.md', 'nooa-join/CHASSIS.json'}
    a, b = [json.loads((p / 'MANIFEST.json').read_text()) for p in (prior, frozen)]
    assert a['caps'] == b['caps'] and a['driver'] == b['driver']
    assert ast.dump(ast.parse((prior / 'constructor.py').read_text())) == ast.dump(ast.parse((frozen / 'constructor.py').read_text()))
    source = (frozen / 'live_agent.py').read_text()
    assert 'HOST-OWNED FREEZE USE' in source
    assert 'FROZEN_PREDICTION_PREVIEW' not in source
    assert source.count('return super().submit_actions(actions, *args, **kwargs)') >= 2


def test_r7_only_teardown_reap_and_inherited_envelope():
    prior = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r6'
    frozen = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r7'
    old = json.loads((prior / 'ASSET-HASHES.json').read_text())
    new = json.loads((frozen / 'ASSET-HASHES.json').read_text())
    assert new.keys() == old.keys()
    assert {n for n in old if old[n] != new[n]} == {'donor_bootstrap.py', 'EXPERIMENT-CARD.md', 'nooa-join/CHASSIS.json'}
    a, b = [json.loads((p / 'MANIFEST.json').read_text()) for p in (prior, frozen)]
    assert a['caps'] == b['caps'] and a['driver'] == b['driver']
    assert (prior / 'process_guard.py').read_bytes() == (frozen / 'process_guard.py').read_bytes()


def test_r8_only_official_loop_driver_and_frozen_predecessor_files():
    """r8 swaps the probe driver for the official-loop driver; R4/R5/R6/R7 files stay byte-identical."""
    prior = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r7'
    old = json.loads((prior / 'ASSET-HASHES.json').read_text())
    new = json.loads((PACKAGE / 'ASSET-HASHES.json').read_text())
    assert new.keys() - old.keys() == {'gameplay_play.py'}
    assert old.keys() - new.keys() == {'kaggle_probe.py', 'convert_frame.py', 'driver_patch.py', 'test_join_qualification.py', 'test_my_agent.py', 'test_nooa_join.py'}
    changed = {n for n in old if n in new and old[n] != new[n]}
    assert changed == {'donor_bootstrap.py', 'local_accounting.py', 'official_launcher_runtime.py', 'EXPERIMENT-CARD.md', 'nooa-join/CHASSIS.json'}
    for name in ('game_preflight.py', 'constructor_contract.py', 'live_agent.py', 'process_guard.py', 'headless.py', 'constructor.py', 'containment_preflight.py', 'runtime_install.py'):
        assert (prior / name).read_bytes() == (PACKAGE / name).read_bytes(), name
    # R7 reap: every bootstrap function except main (argv only) is AST-identical.
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    a, b = functions(prior / 'donor_bootstrap.py'), functions(PACKAGE / 'donor_bootstrap.py')
    assert {k for k in a if a[k] != b[k]} == {'main'}
    assert "'--launcher-timeout-s', '1800'" in (PACKAGE / 'donor_bootstrap.py').read_text()
    # Supervisor: only the probe timeout cap and the claim ceiling text differ.
    old_sup = (prior / 'official_launcher_runtime.py').read_text().splitlines()
    new_sup = (PACKAGE / 'official_launcher_runtime.py').read_text().splitlines()
    assert len(old_sup) == len(new_sup)
    diff = [(o, n) for o, n in zip(old_sup, new_sup) if o != n]
    assert any('min(900, launcher_timeout - 60)' in o and 'launcher_timeout - 60' in n for o, n in diff)
    assert all('min(900' in o or 'claim' in o.lower() or o.strip().startswith('"') for o, n in diff), diff
    manifest = json.loads((PACKAGE / 'MANIFEST.json').read_text())
    assert manifest['caps']['admission_seconds_per_arm'] == 600 and manifest['caps']['actions_per_arm'] == 200
    assert manifest['caps']['calls_total_including_warmup'] == 259 and manifest['caps']['games'] == ['ls20']
    assert manifest['caps']['arm_call_cap'] == {'A': 129, 'B': 129}
    assert 'Agent.main' in manifest['driver'] and 'test_join_qualification' not in manifest['driver']

"""Bounded CPU qualification of frozen generated search, with native validation."""
import ast
import contextlib
import copy
from collections import deque
import hashlib
import heapq
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT/'analysis/counterexample-continuations-v3-20261013'
EPISODE = PRIOR/'output/continuations/episodes/b1-2903-reminder'
MODEL_NAMES = ['gdir_of', 'cellsS', 'growth_region', 'static_set', 'keyS',
               'dotS', 'adj', 'removed_cells', 'sim_step']
ARMS = ['original24', 'original40', 'incremental', 'astar']
SECONDS, EXPANSIONS, STATES = 90, 300000, 500000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definitions(path, namespace, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), (path, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def load_model():
    model = {'copy': copy}
    for name, code in json.loads((HERE/'inputs/model.json').read_text()).items():
        tree = ast.parse(code)
        assert sum(isinstance(n, ast.FunctionDef) for n in tree.body) == 1
        assert all(isinstance(n, ast.FunctionDef) or (isinstance(n, ast.Import)
                   and all(alias.name == 'copy' for alias in n.names)) for n in tree.body)
        exec(compile(tree, 'frozen:'+name, 'exec'), model)
    return model


class Limit(Exception):
    pass


def worker(arm):
    model = load_model()
    original = (HERE/'inputs/original.py').read_text()
    setup = original[:original.index('while q:')]
    ns = dict(model)
    exec(compile(setup, 'original-setup', 'exec'), ns)
    initial = copy.deepcopy(ns['cur'])
    started = time.monotonic()
    stats = {'simulator_calls': 0, 'rejected': 0, 'expanded': 0, 'discovered': 1}
    raw_sim = model['sim_step']

    def check_limits():
        if time.monotonic()-started >= SECONDS:
            raise Limit('TIME_LIMIT')
        if stats['expanded'] >= EXPANSIONS:
            raise Limit('EXPANSION_LIMIT')
        if stats['discovered'] >= STATES:
            raise Limit('STATE_LIMIT')

    def step(state, op, key, static):
        stats['simulator_calls'] += 1
        check_limits()
        result = raw_sim(state, op, key, static)
        stats['rejected'] += result is None
        return result

    result = {'arm': arm, 'status': 'RUNNING', 'path': None,
              'start': initial, 'model_sha256': sha(HERE/'inputs/model.json')}
    try:
        if arm.startswith('original'):
            horizon = int(arm.removeprefix('original'))
            code = original if horizon == 24 else original.replace('len(p)>=24', 'len(p)>=40')
            assert code.count('len(p)>=') == 1
            def original_step(state, op, key, static):
                stats['expanded'] = ns['exp']
                stats['discovered'] = len(ns['seen'])
                return step(state, op, key, static)
            ns['sim_step'] = original_step
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    exec(compile(code, 'frozen-original-search', 'exec'), ns)
            finally:
                result['stdout'] = output.getvalue()
                stats['expanded'], stats['discovered'] = ns['exp'], len(ns['seen'])
                result['best'] = ns['best']
                result['horizon'] = horizon
            found = ns['found']
            result.update(status='FOUND' if found else 'NO_PATH_WITHIN_HORIZON', path=found)
        else:
            static, actions = ns['st'], ns['acts']
            key_of, dot = model['keyS'], model['dotS']
            start_key = key_of(initial)
            parents, edges, depths = {start_key: None}, {}, {start_key: 0}
            serial = 0
            def heuristic(state):
                r, c = dot(state)
                return math.ceil((abs(r-31)+abs(c-52))/3)
            horizon, boundaries = 24 if arm == 'incremental' else 40, []
            queue = deque([(initial, 0)])
            heap = [(heuristic(initial), heuristic(initial), serial, initial, 0)]
            result['horizons'] = [horizon]
            cut_states = 0
            while queue if arm == 'incremental' else heap:
                check_limits()
                if arm == 'incremental':
                    state, depth = queue.popleft()
                else:
                    _, _, _, state, depth = heapq.heappop(heap)
                    if depth != depths[key_of(state)]:
                        continue
                state_key = key_of(state)
                if dot(state) == (31, 52):
                    path = []
                    while parents[state_key] is not None:
                        path.append(edges[state_key]); state_key = parents[state_key]
                    result.update(status='FOUND', path=list(reversed(path)))
                    break
                if depth >= horizon:
                    boundaries.append((state, depth)); cut_states += 1
                else:
                    stats['expanded'] += 1
                    for op, key in actions:
                        successor = step(state, op, key, static)
                        if successor is None:
                            continue
                        successor_key = key_of(successor)
                        next_depth = depth+1
                        if successor_key in depths and (arm == 'incremental' or depths[successor_key] <= next_depth):
                            continue
                        parents[successor_key], edges[successor_key] = state_key, (op, key)
                        depths[successor_key] = next_depth
                        stats['discovered'] = len(depths)
                        if arm == 'incremental':
                            queue.append((successor, next_depth))
                        else:
                            serial += 1
                            h = heuristic(successor)
                            heapq.heappush(heap, (next_depth+h, h, serial, successor, next_depth))
                if arm == 'incremental' and not queue and boundaries and horizon < 40:
                    horizon += 4
                    result['horizons'].append(horizon)
                    queue, boundaries = deque(boundaries), []
            else:
                result['status'] = 'NO_PATH_WITHIN_HORIZON' if cut_states else 'EXHAUSTED_MODEL_GRAPH'
            result['horizon'] = horizon
            result['frontier_cut_states'] = cut_states
    except Limit as exc:
        result.update(status='RESOURCE_LIMIT', reason=str(exc))
    except Exception as exc:
        result.update(status='EXECUTION_ERROR', error=repr(exc))
    result.update(stats, seconds=time.monotonic()-started)
    if result['path'] is not None:
        result['route_length'] = len(result['path'])
    (HERE/'results'/f'{arm}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('start','best','path','stdout')}), flush=True)


def native_context():
    import numpy as np
    from arcengine import ActionInput, ARCBaseGame, GameAction
    receipt = json.loads((ROOT/'analysis/s5i5-inspection-20261012/engine-replay.json').read_text())
    source = Path(receipt['engine_source'])
    assert sha(source) == receipt['engine_sha256']
    spec = importlib.util.spec_from_file_location('planner_qualification_native', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cls = next(v for v in vars(module).values() if isinstance(v,type)
               and issubclass(v,ARCBaseGame) and v is not ARCBaseGame)
    helpers = {'np':np,'ActionInput':ActionInput,'GameAction':GameAction,'cls':cls}
    definitions(ROOT/'analysis/s5i5-inspection-20261012/replay.py', helpers,
                {'make','perform','state_from_board'})
    return helpers, receipt


def prefix(helpers):
    game = helpers['make']()
    for action in json.loads((HERE/'inputs/actions.json').read_text()):
        frame = helpers['perform'](game, action)
    setup = (HERE/'inputs/original.py').read_text().split('while q:')[0]
    ns = load_model(); exec(compile(setup, 'original-setup', 'exec'), ns)
    assert frame.levels_completed == 1
    assert helpers['state_from_board'](frame.frame[-1]) == ns['cur']
    return game, frame, ns


def native_route(arm, helpers):
    result = json.loads((HERE/'results'/f'{arm}.json').read_text())
    if result['path'] is None:
        return {'arm':arm,'status':'NO_RETURNED_ROUTE'}
    game, frame, model = prefix(helpers)
    static = model['static_set']()
    click = {'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
    rows = []
    for op, key in result['path']:
        before = helpers['state_from_board'](frame.frame[-1])
        predicted = model['sim_step'](copy.deepcopy(before), op, key, static)
        frame = helpers['perform'](game, {'id':'ACTION6','data':{'x':click[key+op],'y':57}})
        board = game.inspection_wins[-1] if frame.levels_completed > 1 else frame.frame[-1]
        actual = helpers['state_from_board'](board)
        rows.append({'step':len(rows)+1,'action':key+op,'input':before,
                     'predicted':predicted,'actual':actual,'matched':predicted==actual,
                     'board_sha256':hashlib.sha256(json.dumps(board).encode()).hexdigest(),
                     'levels_completed':frame.levels_completed})
        if predicted != actual or frame.levels_completed > 1:
            break
    solved = frame.levels_completed == 2 and all(r['matched'] for r in rows)
    return {'arm':arm,'status':'NATIVE_COMPLETION' if solved else 'NATIVE_MISMATCH',
            'native_solved':solved,'actions_executed':len(rows),'rows':rows}


def main():
    (HERE/'inputs').mkdir(exist_ok=True); (HERE/'results').mkdir(exist_ok=True)
    inputs = [PRIOR/'transcript-audit/b1-2903-reminder.json', EPISODE/'restoration.json',
              EPISODE/'actions.json', ROOT/'analysis/s5i5-inspection-20261012/engine-replay.json',
              ROOT/'analysis/s5i5-inspection-20261012/replay.py',
              ROOT/'solvers/m2_b1_fixed/evidence/verification.json']
    hashes = {str(p):sha(p) for p in inputs}
    calls = json.loads(inputs[0].read_text())
    call = next(c for c in calls if c['call']==8)
    code = call['tools'][0]['display_code']
    (HERE/'inputs/original.py').write_text(code)
    (HERE/'inputs/original-receipt.json').write_text(json.dumps(call['tools'][0]['receipt'],indent=2)+'\n')
    retained = json.loads(inputs[1].read_text())['retained_functions']
    (HERE/'inputs/model.json').write_text(json.dumps({name:retained[name] for name in MODEL_NAMES},indent=2)+'\n')
    (HERE/'inputs/actions.json').write_bytes(inputs[2].read_bytes())
    helpers, receipt = native_context()
    game, frame, ns = prefix(helpers)
    manifest = {'source_sha256':hashes,'engine_sha256':receipt['engine_sha256'],
                'runner_sha256':sha(Path(__file__)), 'card_sha256':sha(HERE/'EXPERIMENT-CARD.md'),
                'frozen_model_sha256':sha(HERE/'inputs/model.json'),
                'prefix_verified':True,'start':ns['cur'], 'caps':{'seconds':SECONDS,'expanded':EXPANSIONS,'states':STATES}}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for arm in ARMS:
        print('Starting',arm,flush=True)
        with (HERE/'results'/f'{arm}.log').open('w') as log:
            try:
                run = subprocess.run([sys.executable,'-B',__file__,arm],stdout=log,stderr=subprocess.STDOUT,timeout=110)
                if run.returncode:
                    raise RuntimeError(f'Worker {arm} failed: {run.returncode}')
            except subprocess.TimeoutExpired:
                (HERE/'results'/f'{arm}.json').write_text(json.dumps({'arm':arm,'status':'RESOURCE_LIMIT','reason':'SUPERVISOR_TIMEOUT','path':None})+'\n')
        result = json.loads((HERE/'results'/f'{arm}.json').read_text())
        print(arm,result['status'],result.get('reason'),result.get('route_length'),result.get('seconds'),flush=True)
    native = [native_route(arm,helpers) for arm in ARMS]
    (HERE/'native-receipts.json').write_text(json.dumps(native,indent=2)+'\n')
    outcomes = [json.loads((HERE/'results'/f'{arm}.json').read_text()) for arm in ARMS]
    baseline = outcomes[0]
    reproduced = (baseline.get('discovered')==211935 and baseline.get('expanded')==158413
                  and baseline['status']=='NO_PATH_WITHIN_HORIZON')
    validity = 'VALID' if reproduced or baseline['status']=='RESOURCE_LIMIT' else 'INVALID_BASELINE_RECONSTRUCTION'
    for p, digest in hashes.items():
        assert sha(Path(p))==digest, 'Protected source changed: '+p
    summary = {'scope':'CPU frozen-code diagnostic; no model/GPU calls or solver deployment',
               'validity':validity, 'original_counts_reproduced':reproduced,
               'results':outcomes,'native_validation':[{k:v for k,v in row.items() if k!='rows'} for row in native],
               'protected_sources_unchanged':True}
    (HERE/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2),flush=True)


if __name__ == '__main__':
    worker(sys.argv[1]) if len(sys.argv)>1 else main()

"""Frozen six-model census; no reference dynamics/route enter the planner."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / 'analysis/counterexample-continuations-v3-20261013/output/continuations/episodes'
EPISODES = {'b1-1701-control':'growS', 'b1-2903-control':'growS',
            'b1-1701-reminder':'growC', 'b1-2903-reminder':'sim_step',
            'b1-1701-counterexample':'growS4', 'b1-2903-counterexample':'growS7'}
CLICK = {'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_prefix(episode):
    qualification = load_module('previous_qualification', ROOT / 'analysis/planner-qualification-v2-20261013/run_probe.py')
    helpers, receipt = qualification.native_context()
    game = helpers['make']()
    for action in json.loads((ARCHIVE / episode / 'actions.json').read_text()):
        frame = helpers['perform'](game, action)
    if frame.levels_completed != 1:
        raise RuntimeError('Unexpected archived level')
    return helpers, game, frame


def model(episode):
    namespace = {'copy':copy}
    retained = json.loads((ARCHIVE / episode / 'restoration.json').read_text())['retained_functions']
    for name, source in retained.items():
        tree = ast.parse(source)
        if not all(isinstance(n, ast.FunctionDef) or (isinstance(n, ast.Import)
                   and all(a.name == 'copy' for a in n.names)) for n in tree.body):
            raise RuntimeError('Non-definition retained source: ' + name)
        exec(compile(tree, 'archived:'+episode+':'+name, 'exec'), namespace)
    return namespace


def worker(episode, arm):
    planner = load_module('qualified_search', ROOT / 'solvers/m2_planning_tools/overlay/inference/utils/bounded_search.py')
    namespace = model(episode)
    helpers, game, frame = native_prefix(episode)
    initial = helpers['state_from_board'](frame.frame[-1])
    static = namespace['static_set']()
    grow = EPISODES[episode]
    def successors(state):
        for operation in ('g', 's'):
            for key in 'YONS':
                if grow == 'sim_step':
                    successor = namespace[grow](copy.deepcopy(state), operation, key, static)
                elif operation == 'g':
                    successor = namespace[grow](copy.deepcopy(state), static, key)
                else:
                    successor = namespace['shrinkS'](copy.deepcopy(state), key)
                yield [operation,key], successor
    def heuristic(state):
        row, column = namespace['dotS'](state)
        return math.ceil((abs(row-31)+abs(column-52))/3)
    result = planner.bounded_search(initial, successors, state_key=namespace['keyS'],
        is_goal=lambda state:namespace['dotS'](state)==(31,52),
        heuristic=heuristic if arm=='guided' else None,
        max_depth=40, max_nodes=500000, max_expanded=300000, max_seconds=20)
    rows = []
    predicted_state = copy.deepcopy(initial)
    for operation, key in result.get('path') or []:
        predicted = (namespace[grow](copy.deepcopy(predicted_state), operation, key, static)
                     if grow=='sim_step' else namespace[grow](copy.deepcopy(predicted_state),static,key)
                     if operation=='g' else namespace['shrinkS'](copy.deepcopy(predicted_state),key))
        frame = helpers['perform'](game,{'id':'ACTION6','data':{'x':CLICK[key+operation],'y':57}})
        board = game.inspection_wins[-1] if frame.levels_completed>1 else frame.frame[-1]
        actual = helpers['state_from_board'](board)
        rows.append({'action':[operation,key],'predicted':predicted,'actual':actual,'matched':predicted==actual})
        if predicted!=actual or frame.levels_completed>1:
            break
        predicted_state = predicted
    native_valid = bool(rows) and all(r['matched'] for r in rows) and frame.levels_completed==2
    result.update(episode=episode, arm=arm, start=initial, native_rows=rows,
                  native_valid_completion=native_valid,
                  native_status='VALID_COMPLETION' if native_valid else 'FIRST_MISMATCH' if rows and not rows[-1]['matched']
                  else 'NO_COMPLETION' if rows else 'NO_RETURNED_ROUTE')
    (HERE / 'evidence' / (episode+'-'+arm+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['episode','arm','status','reason','expanded','seconds','depth','native_status']}),flush=True)


def main():
    manifest_path = HERE / 'model-manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Do not overwrite a completed or partial census')
    inputs = [path for episode in EPISODES for path in [ARCHIVE/episode/'restoration.json', ARCHIVE/episode/'actions.json']]
    inputs += [ROOT/'solvers/m2_planning_tools/overlay/inference/utils/bounded_search.py', Path(__file__)]
    manifest = {'episodes':EPISODES,'arms':['unguided','guided'], 'limits':{'seconds':20,'depth':40,'stored_nodes':500000,'expanded':300000,'process_seconds':30},
                'scope':'Retrospective six-model diagnostic on one game; adapters and goal are analyst supplied from archived agent code',
                'inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs}}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    results = []
    for episode in EPISODES:
        for arm in manifest['arms']:
            with (HERE/'evidence'/(episode+'-'+arm+'.log')).open('x') as log:
                try:
                    run = subprocess.run([sys.executable,'-B',str(Path(__file__)),episode,arm],stdout=log,stderr=subprocess.STDOUT,timeout=30)
                    if run.returncode:
                        row = {'episode':episode,'arm':arm,'status':'WORKER_ERROR'}
                    else:
                        row = json.loads((HERE/'evidence'/(episode+'-'+arm+'.json')).read_text())
                except subprocess.TimeoutExpired:
                    row = {'episode':episode,'arm':arm,'status':'PROCESS_LIMIT'}
            results.append(row)
            print(episode,arm,row['status'],row.get('native_status'),flush=True)
    for relative, expected in manifest['inputs'].items():
        if sha(ROOT/relative)!=expected:
            raise RuntimeError('Frozen input changed: '+relative)
    summary = {'scope':manifest['scope'],'selected':len(results),'executed':sum(r['status'] not in ('WORKER_ERROR','PROCESS_LIMIT') for r in results),
               'native_completions':sum(r.get('native_valid_completion',False) for r in results),
               'rows':[{k:v for k,v in r.items() if k not in ('start','state','path','native_rows')} for r in results],
               'model_calls':0,'gpu_launches':0,'input_hashes_verified':True}
    (HERE/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    worker(*sys.argv[1:]) if len(sys.argv)>1 else main()

"""Frozen paired CPU screen of the shipped bounded-search utility."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'analysis/counterexample-continuations-v3-20261013'
SOLVER=ROOT/'solvers/m2_search'
MODELS=[('b1-1701-control','growS'),('b1-2903-control','growS'),
        ('b1-1701-reminder','growC'),('b1-2903-reminder','sim_step'),
        ('b1-1701-counterexample','growS4'),('b1-2903-counterexample','growS7')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def model(case):
    ns={'copy':copy}
    for name,code in case['functions'].items():
        tree=ast.parse(code)
        assert all(isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Import)
                   and all(a.name=='copy' for a in n.names)) for n in tree.body),(case['episode'],name)
        exec(compile(tree,case['episode']+':'+name,'exec'),ns)
    static=ns['static_set']()
    def predict(state,action):
        op,key=action
        if case['grow']=='sim_step':
            return ns['sim_step'](copy.deepcopy(state),op,key,static)
        return (ns[case['grow']](copy.deepcopy(state),static,key) if op=='g'
                else ns['shrinkS'](copy.deepcopy(state),key))
    def successors(state):
        for op in ('g','s'):
            for key in 'YONS':
                action=(op,key)
                yield action,predict(state,action)
    def goal(state):
        return ns['dotS'](state)==(31,52)
    def heuristic(state):
        r,c=ns['dotS'](state)
        return math.ceil((abs(r-31)+abs(c-52))/3)
    return ns,predict,successors,goal,heuristic


def worker(index,arm):
    cases=json.loads((HERE/'cases.json').read_text());case=cases[index]
    helper=load_module('frozen_search',SOLVER/'src/inference/utils/bounded_search.py')
    _,_,successors,goal,heuristic=model(case)
    result=helper.bounded_search(case['start'],successors,state_key=model(case)[0]['keyS'],is_goal=goal,
         heuristic=heuristic if arm=='guided' else None,max_depth=60,max_nodes=100000,
         max_expanded=100000,max_seconds=20)
    result.update(episode=case['episode'],arm=arm,
                  helper_sha256=digest(SOLVER/'src/inference/utils/bounded_search.py'))
    (HERE/'results'/f'{case["episode"]}-{arm}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('path','state')}),flush=True)


def replay_prefix(case,helpers):
    game=helpers['make']()
    for action in case['actions']:
        frame=helpers['perform'](game,action)
    assert frame.levels_completed==1
    assert helpers['state_from_board'](frame.frame[-1])==case['start']
    return game,frame


def verify_route(case,result,helpers):
    if result['status']!='FOUND':
        return {'episode':case['episode'],'arm':result['arm'],'status':'NO_RETURNED_ROUTE'}
    game,frame=replay_prefix(case,helpers)
    _,predict,_,_,_=model(case)
    click={'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
    rows=[]
    for action in result['path']:
        before=helpers['state_from_board'](frame.frame[-1])
        expected=predict(before,action)
        op,key=action
        frame=helpers['perform'](game,{'id':'ACTION6','data':{'x':click[key+op],'y':57}})
        board=game.inspection_wins[-1] if frame.levels_completed>1 else frame.frame[-1]
        actual=helpers['state_from_board'](board)
        rows.append({'step':len(rows)+1,'action':action,'before':before,'predicted':expected,
                     'actual':actual,'matched':expected==actual,'levels_completed':frame.levels_completed,
                     'board_sha256':hashlib.sha256(json.dumps(board).encode()).hexdigest()})
        if expected!=actual or frame.levels_completed>1:
            break
    solved=frame.levels_completed==2 and all(r['matched'] for r in rows)
    status='NATIVE_COMPLETION' if solved else ('NATIVE_MISMATCH' if any(not r['matched'] for r in rows) else 'NATIVE_NO_COMPLETION')
    return {'episode':case['episode'],'arm':result['arm'],'status':status,
            'native_solved':solved,'actions_executed':len(rows),'rows':rows}


def main():
    assert not (HERE/'RESULT.json').exists(),'Existing frozen result; choose a new run for another attempt'
    (HERE/'results').mkdir(exist_ok=True)
    prior_runner=load_module('prior_native_helpers',ROOT/'analysis/planner-qualification-v2-20261013/run_probe.py')
    helpers,engine=prior_runner.native_context()
    hashes={str(SOLVER/'src/inference/utils/bounded_search.py'):digest(SOLVER/'src/inference/utils/bounded_search.py')}
    cases=[]
    for episode,grow in MODELS:
        directory=PRIOR/'output/continuations/episodes'/episode
        restoration=directory/'restoration.json';actions=directory/'actions.json'
        hashes.update({str(p):digest(p) for p in (restoration,actions)})
        case={'episode':episode,'grow':grow,'functions':json.loads(restoration.read_text())['retained_functions'],
              'actions':json.loads(actions.read_text())}
        model(case)
        game=helpers['make']()
        for action in case['actions']:
            frame=helpers['perform'](game,action)
        assert frame.levels_completed==1
        case['start']=helpers['state_from_board'](frame.frame[-1])
        cases.append(case)
    assert len(cases)==6
    (HERE/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    hashes[str(HERE/'cases.json')]=digest(HERE/'cases.json')
    manifest={'sources':hashes,'engine_sha256':engine['engine_sha256'],
              'runner_sha256':digest(Path(__file__)),'card_sha256':digest(HERE/'EXPERIMENT-CARD.md'),
              'protected_parent_hashes':json.loads((SOLVER/'evidence/parent-hashes.json').read_text())}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    outcomes=[];native=[]
    for index,case in enumerate(cases):
        for arm in ('unguided','guided'):
            name=case['episode']+'-'+arm
            print('Starting',name,flush=True)
            with (HERE/'results'/f'{name}.log').open('w') as log:
                run=subprocess.run([sys.executable,'-B',__file__,str(index),arm],stdout=log,stderr=subprocess.STDOUT,timeout=40)
            assert run.returncode==0,(name,'worker failed; matrix stopped')
            result=json.loads((HERE/'results'/f'{name}.json').read_text())
            assert result['helper_sha256']==hashes[str(SOLVER/'src/inference/utils/bounded_search.py')]
            outcome=verify_route(case,result,helpers)
            outcomes.append(result);native.append(outcome)
            print(name,result['status'],result['reason'],round(result['seconds'],2),outcome['status'],flush=True)
            (HERE/'native-receipts.json').write_text(json.dumps(native,indent=2)+'\n')
    for path,d in hashes.items():
        assert digest(Path(path))==d,path
    parent=SOLVER.parent/'m2_b1_fixed'
    for path,d in manifest['protected_parent_hashes'].items():
        assert digest(parent/path)==d,path
    summary={'validity':'VALID','scope':'Frozen models and analyst-written adapters; not live model uptake or score efficacy',
             'models':6,'arms':len(outcomes),'outcomes':outcomes,
             'native':[{k:v for k,v in n.items() if k!='rows'} for n in native],
             'protected_sources_unchanged':True}
    (HERE/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Screen complete:',len(outcomes),'arms',flush=True)


if __name__=='__main__':
    worker(int(sys.argv[1]),sys.argv[2]) if len(sys.argv)>1 else main()

"""CPU-only component tests: causal past, retrospective transfer, search and native score.

Run with solvers/tycho_nooa/.venv/bin/python -B .../run_probe.py.
No imports of previous analysis scripts: some have write-on-import side effects.
"""
from pathlib import Path
from collections import Counter, deque
import ast
import copy
import hashlib
import importlib.util
import json
import re
import signal
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'analysis/s5i5-inspection-20261012'
COLORS = 'OSYN'
CLICK = {'Og':12, 'Os':6, 'Sg':27, 'Ss':21, 'Yg':42, 'Ys':36, 'Ng':57, 'Ns':51}
started = time.monotonic()
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('180s cap')))
signal.alarm(180)
report = {'scope':'Retrospective component diagnosis; no new LLM treatment or benchmark run',
          'sources':{}, 'coverage':{}, 'search':[], 'score':[], 'status':'RUNNING'}


def read(path):
    report['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def extract(code, names=None, ns=None):
    ns = {} if ns is None else ns
    tree = ast.parse(code)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<archived-functions>', 'exec'), ns)
    return ns


def functions(label, line, ns):
    fragment = next(f for b in records[label]['blocks'] for f in b['fragments'] if f['source_line'] == line)
    code = re.search(r'<parameter=code>\s*(.*?)\s*</parameter>', fragment['text'], re.S)[1]
    return extract(code, ns=ns)


def original(state, key, delta):
    return bad['growS'](copy.deepcopy(state), bad['static_set'](), key) if delta > 0 else bad['shrinkS'](copy.deepcopy(state), key)


def one_rule(state, key, delta):
    """Repair only the witnessed S+ coupling; do not supply other chain mechanics."""
    pred = original(state, key, delta)
    if pred is not None and key == 'S' and delta > 0:
        pred['N']['r'] = state['N']['r'] + pred['Y']['r'] - state['Y']['r']
    return pred


def reference(state, key, delta):
    lengths = [state[k]['L'] for k in COLORS]
    lengths[COLORS.index(key)] += delta
    if not good['ok3'](tuple(lengths)):
        return None
    Lo,Ls,Ly,Lg = lengths
    return {'O':{'axis':'H','r':39,'c':10,'L':Lo},
            'S':{'axis':'V','dir':-1,'r':41-Ls,'c':10+Lo,'L':Ls},
            'Y':{'axis':'H','r':38-Ls,'c':11+Lo,'L':Ly},
            'N':{'axis':'V','dir':1,'r':39-Ls,'c':11+Lo+Ly,'L':Lg}}


def check(predict, rows):
    counts = Counter(selected=len(rows), called=0, compared=0, matched=0, mismatched=0,
                     rejected_change=0, abstained_noop=0, errors=0)
    details = []
    for row in rows:
        key,delta = row['interpreted_panel_action']
        counts['called'] += 1
        try:
            pred = predict(copy.deepcopy(row['input_state']), key, delta)
            if pred is None:
                status = 'abstained_noop' if row['input_state'] == row['actual_successor'] else 'rejected_change'
            else:
                counts['compared'] += 1
                status = 'matched' if pred == row['actual_successor'] else 'mismatched'
        except Exception as exc:
            pred = {'error':repr(exc)}
            status = 'errors'
        counts[status] += 1
        details.append({'run':row['run'], 'local_action':row['local_action'], 'input':row['input_state'],
                        'action':[key,delta], 'predicted':pred, 'actual':row['actual_successor'], 'status':status})
    assert counts['called'] == sum(counts[k] for k in ['matched','mismatched','rejected_change','abstained_noop','errors'])
    status = 'INVALID_EMPTY' if not counts['compared'] else ('CONTRADICTED' if counts['mismatched'] or counts['rejected_change'] or counts['errors'] else 'CONSISTENT_ON_COMPARED')
    return {'counts':dict(counts), 'status':status, 'rows':details}


def search(start, horizon):
    start = tuple(start[k]['L'] for k in COLORS)
    queue = deque([(start,0)])
    parents = {start:None}
    edge = {}
    expanded = 0
    frontier_cut = 0
    while queue:
        state, depth = queue.popleft()
        if good['marker'](state) == (31,52):
            path=[]
            while parents[state] is not None:
                path.append(edge[state]);state=parents[state]
            return {'status':'FOUND', 'path':list(reversed(path)), 'expanded':expanded,
                    'discovered':len(parents), 'horizon':horizon}
        if depth == horizon:
            frontier_cut += 1
            continue
        expanded += 1
        for i,key in enumerate(COLORS):
            for delta in (3,-3):
                candidate=list(state);candidate[i]+=delta;candidate=tuple(candidate)
                if candidate in parents or not good['ok3'](candidate):continue
                parents[candidate]=state;edge[candidate]=key+('g' if delta>0 else 's')
                queue.append((candidate,depth+1))
                if len(parents)>10000:raise RuntimeError('search state cap')
    return {'status':'NO_PATH_WITHIN_HORIZON' if frontier_cut else 'EXHAUSTED_MODEL_GRAPH',
            'path':None, 'horizon':horizon, 'expanded':expanded, 'discovered':len(parents),
            'depth_frontier_states':frontier_cut}


try:
    records=read(PRIOR/'records.json')
    observed=read(PRIOR/'engine-replay.json')
    previous=read(PRIOR/'model-check.json')
    bad={};functions('b1-r1',4731,bad);functions('b1-r1',7835,bad)
    good={'OBS':{(r,c) for r,row in enumerate(records['control-pc']['entry_board']) for c,col in enumerate(row)
                 if col==15 or (r>=52 and col!=5)}}
    functions('control-pc',4862,good);functions('control-pc',5635,good)
    rows=[{**row,'run':label} for label,entry in previous['transition_tests'].items() for row in entry['rows']]
    subsets={'past_at_checkpoint':[r for r in rows if r['run']=='b1-r1' and r['local_action']<=22],
             'later_same_run':[r for r in rows if r['run']=='b1-r1' and r['local_action']>22],
             'other_selected_runs':[r for r in rows if r['run']!='b1-r1']}
    for subset,rr in subsets.items():
        report['coverage'][subset]={name:check(fn,rr) for name,fn in [('original',original),('one_rule',one_rule),('reference',reference)]}
    report['validator_sanity']={
        'empty_selection':check(original,[])['status'],
        'always_abstain':check(lambda *_:None,subsets['past_at_checkpoint'])['status'],
        'always_copy_predecessor':check(lambda s,*_:s,subsets['past_at_checkpoint'])['status']}
    assert report['validator_sanity']=={'empty_selection':'INVALID_EMPTY','always_abstain':'INVALID_EMPTY','always_copy_predecessor':'CONTRADICTED'}
    known=next(r for r in subsets['past_at_checkpoint'] if r['local_action']==16)
    assert original(known['input_state'], 'S',3)!=known['actual_successor']
    assert one_rule(known['input_state'], 'S',3)==known['actual_successor']

    for label,stop in [('b1-r1',22),('b1-r2',46)]:
        for horizon in [27,31,44]:
            receipt=search(observed['runs'][label]['states'][stop]['state'],horizon)
            report['search'].append({'run':label,'after_local_action':stop,**receipt})

    # Load only helper definitions from the previous replay; do not replay/write it on import.
    import numpy as np
    from arcengine import ActionInput, ARCBaseGame, GameAction
    src=Path(observed['engine_source'])
    assert hashlib.sha256(src.read_bytes()).hexdigest()==observed['engine_sha256']
    spec=importlib.util.spec_from_file_location('s5_native_probe',src)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cls=next(v for v in vars(module).values() if isinstance(v,type) and issubclass(v,ARCBaseGame) and v is not ARCBaseGame)
    helpers=extract((PRIOR/'replay.py').read_text(),ns={'np':np,'ActionInput':ActionInput,'GameAction':GameAction,'cls':cls})
    scorefile=ROOT/'solvers/m2_b1/src/inference/tools/traces.py'
    report['sources'][str(scorefile)]=hashlib.sha256(scorefile.read_bytes()).hexdigest()
    scoring=extract(scorefile.read_text(),['_weighted_mean','compute_level_score','compute_game_score'],
                    {'COMPLETED_LEVEL_STATUS':'COMPLETED','PER_LEVEL_SCORE_CAP':115.,'GAME_SCORE_CAP':100.})
    def score(run,completed,actions):
        return scoring['compute_game_score']([(i+1,scoring['compute_level_score'](baseline_actions=baseline,
                       agent_steps=actions[i],agent_status='COMPLETED' if i<completed else 'MISSING'))
                       for i,baseline in enumerate(run['base_actions_per_level'])])
    # Qualify the local metric against every archived game in both repetitions.
    validated_scores=0
    native=[]
    for label,stop in [('b1-r1',22),('b1-r2',46)]:
        output=ROOT/Path(records[label]['source']).parent.parent
        benchmark=read(output/'benchmark.json')
        for run in benchmark['game_runs']:
            assert abs(score(run,run['levels_completed'],run['actions_per_level'])-run['final_score'])<1e-8,run['game_id']
            validated_scores+=1
        run=next(r for r in benchmark['game_runs'] if r['game_id']=='s5i5-18d95033')
        chosen=next(r for r in report['search'] if r['run']==label and r['status']=='FOUND')
        game=helpers['make']()
        for action in run['history'][:run['actions_per_level'][0]+stop]:
            frame=helpers['perform'](game,action['action'])
        assert frame.frame[-1]==observed['runs'][label]['states'][stop]['board']
        remaining=game.gwiuiwqizb.current_steps
        receipt={'run':label,'prefix_verified':True,'remaining_steps':remaining,'actions':[]}
        for token in chosen['path']:
            pre=helpers['state_from_board'](frame.frame[-1])
            predicted=reference(pre,token[0],3 if token[1]=='g' else -3)
            frame=helpers['perform'](game,{'id':'ACTION6','data':{'x':CLICK[token],'y':57}})
            board=game.inspection_wins[-1] if frame.levels_completed>1 else frame.frame[-1]
            actual=helpers['state_from_board'](board)
            assert predicted==actual,(label,token,'reference plan transition mismatch')
            receipt['actions'].append({'token':token,'actual_state':actual,'frame':board,'levels_completed':frame.levels_completed})
        assert frame.levels_completed==2
        receipt['solved']=True;native.append(receipt)
        counts=run['actions_per_level'].copy();counts[1]=stop+len(chosen['path'])
        before=score(run,1,run['actions_per_level']);after=score(run,2,counts)
        report['score'].append({'run':label,'original_game_score':before,'recovery_game_score':after,
                                'game_score_delta':after-before,'public25_mean_delta_if_others_unchanged':(after-before)/25,
                                'level2_prefix_actions':stop,'recovery_actions':len(chosen['path']),
                                'total_level2_actions':counts[1],'baseline_level2_actions':run['base_actions_per_level'][1],
                                'scope':'Offline privileged-reference counterfactual; excludes later-level benefit, inference/queue time and opportunity cost.'})
    (OUT/'native-receipts.json').write_text(json.dumps(native,indent=2)+'\n')
    report['archived_scores_validated']=validated_scores
    report['native_recovery_steps_verified']=sum(len(r['actions']) for r in native)
    report['status']='COMPLETE'
except Exception as exc:
    report['status']='FAILED';report['error']=repr(exc)
    raise
finally:
    signal.alarm(0)
    report['wall_seconds']=time.monotonic()-started
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources','coverage']},indent=2))
    for subset,models in report['coverage'].items():
        print(subset,{name:result['counts'] for name,result in models.items()})

"""CPU check of the actual retained B1 repairs against native-verified transitions."""
import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
prior_path = ROOT / 'analysis/score-conversion-20261013/results.json'
prior = json.loads(prior_path.read_text())
subsets = {key: value['original']['rows'] for key, value in prior['coverage'].items()}
native_path = ROOT / 'analysis/score-conversion-20261013/native-receipts.json'
engine_path = ROOT / 'analysis/s5i5-inspection-20261012/engine-replay.json'
native = next(r for r in json.loads(native_path.read_text()) if r['run']=='b1-r1')
state = json.loads(engine_path.read_text())['runs']['b1-r1']['states'][22]['state']
route = []
for i, step in enumerate(native['actions']):
    route.append({'run':'native_verified_31_click_recovery','local_action':i+1,
                  'input':state,'action':[step['token'][0],3 if step['token'][1]=='g' else -3],
                  'actual':step['actual_state']})
    state = step['actual_state']
subsets['native_verified_recovery_route'] = route
allowed = {'gdir_of','growth_region','cellsS','static_set','shrinkS','cellsFull','tipface','neigh','growS4','growS7',
           'growS','growC','adj','removed_cells','sim_step'}
report = {'scope': 'Retrospective CPU diagnosis of frozen generated code. No new model calls.',
          'source_sha256': {str(prior_path): hashlib.sha256(prior_path.read_bytes()).hexdigest()},
          'historical_action_coverage': dict(Counter(r['action'][0]+('+' if r['action'][1]>0 else '-') for r in subsets['past_at_checkpoint'])),
          'models': {}}
for path in [native_path, engine_path]:
    report['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
for episode, grow in [('b1-1701-control','growS'), ('b1-2903-control','growS'),
                      ('b1-1701-reminder','growC'), ('b1-2903-reminder','sim_step'),
                      ('b1-1701-counterexample','growS4'), ('b1-2903-counterexample','growS7')]:
    path = HERE / 'output/continuations/episodes' / episode / 'restoration.json'
    report['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = json.loads(path.read_text())['retained_functions']
    namespace = {}
    for name, code in sources.items():
        if name not in allowed:
            continue
        tree = ast.parse(code)
        assert all(isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Import) and all(a.name=='copy' for a in n.names)) for n in tree.body)
        exec(compile(tree, str(path)+':'+name, 'exec'), namespace)
    outcome = {}
    for subset, rows in subsets.items():
        counts = Counter(selected=len(rows), called=0, matched=0, mismatched=0,
                         abstained_noop=0, rejected_change=0, errors=0)
        details=[]
        for row in rows:
            key, delta = row['action']; counts['called'] += 1
            try:
                state=copy.deepcopy(row['input'])
                if grow=='sim_step':
                    predicted=namespace[grow](state,'g' if delta>0 else 's',key,namespace['static_set']())
                else:
                    predicted=namespace[grow](state,namespace['static_set'](),key) if delta>0 else namespace['shrinkS'](state,key)
                if predicted is None:
                    status='abstained_noop' if row['input']==row['actual'] else 'rejected_change'
                else:
                    status='matched' if predicted==row['actual'] else 'mismatched'
            except Exception as exc:
                predicted={'error':repr(exc)};status='errors'
            counts[status]+=1
            details.append({**row,'predicted':predicted,'status':status})
        outcome[subset]={'counts':dict(counts),'rows':details}
    report['models'][episode]=outcome
out=HERE/'transcript-audit/repair-probe.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print('Historical actions:',report['historical_action_coverage'])
for episode,subsets in report['models'].items():
    print(episode)
    for name,result in subsets.items():print(name,result['counts'])

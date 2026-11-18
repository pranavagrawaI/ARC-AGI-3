"""Independent known-graph checks of the frozen runner's search/limit contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_runner_test', HERE/'run_probe.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
saved = runner.HERE, runner.SECONDS, runner.EXPANSIONS, runner.STATES
checks = []
original = '''from collections import deque
cur={'pos': -26}
st=static_set()
acts=[('g','N')]
seen={keyS(cur):None}; q=deque([(cur,())]); best=(99,None,None); found=None; exp=0
while q:
    S,p=q.popleft()
    if len(p)>=24: continue
    exp+=1
    dt=dotS(S); dd=abs(dt[0]-31)+abs(dt[1]-52)
    if dd<best[0]: best=(dd,dt,p)
    for op,k in acts:
        n2=sim_step(S,op,k,st)
        if n2 is None: continue
        kk=keyS(n2)
        if kk in seen: continue
        seen[kk]=1; p2=p+((op,k),)
        if dotS(n2)==(31,52): found=p2; break
        q.append((n2,p2))
    if found: break
print('states',len(seen),'exp',exp)
print('found',found)
print('best',best)
'''
model = {'static_set': 'def static_set(): return set()',
         'keyS': 'def keyS(s): return s["pos"]',
         'dotS': 'def dotS(s): return (31,s["pos"])',
         'sim_step': 'def sim_step(s,op,key,static):\n    return None if s["pos"] >= 52 else {"pos":s["pos"]+3}'}
try:
    with tempfile.TemporaryDirectory() as directory:
        runner.HERE = Path(directory)
        (runner.HERE/'inputs').mkdir(); (runner.HERE/'results').mkdir()
        (runner.HERE/'inputs/original.py').write_text(original)
        (runner.HERE/'inputs/model.json').write_text(json.dumps(model))
        for arm in runner.ARMS:
            runner.worker(arm)
            result = json.loads((runner.HERE/'results'/f'{arm}.json').read_text())
            if arm == 'original24':
                assert result['status']=='NO_PATH_WITHIN_HORIZON' and result['expanded']==24
            else:
                assert result['status']=='FOUND' and result['route_length']==26
            checks.append({'arm':arm,'independent_26_edge_chain':True})
        runner.STATES = 2
        for arm in runner.ARMS:
            runner.worker(arm)
            result = json.loads((runner.HERE/'results'/f'{arm}.json').read_text())
            assert result['status']=='RESOURCE_LIMIT' and result['reason']=='STATE_LIMIT'
            assert result['discovered'] <= 2
            checks.append({'arm':arm,'state_limit_not_impossibility':True})
        runner.STATES = saved[3]; runner.SECONDS = 0
        runner.worker('astar')
        result = json.loads((runner.HERE/'results/astar.json').read_text())
        assert result['status']=='RESOURCE_LIMIT' and result['reason']=='TIME_LIMIT'
        checks.append({'absolute_search_deadline':True})
finally:
    runner.HERE, runner.SECONDS, runner.EXPANSIONS, runner.STATES = saved
(HERE/'runner-validation.json').write_text(json.dumps({'checks':checks,'passed':len(checks)},indent=2)+'\n')
print('Runner contracts PASS:',len(checks))

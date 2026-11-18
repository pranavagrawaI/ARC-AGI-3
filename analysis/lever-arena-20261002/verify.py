"""Verify consequential arena facts locally. No solver edits or model/network calls."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
ARCH=REPO/'analysis/m2-winners-20261001'
scorer=ARCH/'audit/dfranzen-repo/ARC3-Inference/inference/tools/traces.py'
tree=ast.parse(scorer.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_weighted_mean','compute_game_score')]
assert len(nodes)==2
ns={'GAME_SCORE_CAP':100.0}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(scorer),'exec'),ns)
results={}
inputs=[scorer,ROOT/'sources/topic-717133.md',ROOT/'sources/leaderboard.json']
for arm in ['control','control-r2','s1']:
    path=ARCH/f'runs/out-{arm}/summary.txt';inputs.append(path)
    rows=re.findall(r'  (\S+): score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)',path.read_text())
    assert len(rows)==25
    ceilings=[];scores=[]
    for game,score,level,n,*rest in rows:
        level=int(float(level));n=int(n)
        # Execute archived scorer with ideal completed-level credit rather than
        # simply repeating the candidate's closed-form ceiling expression.
        ceiling=ns['compute_game_score']([(i,100.0 if i<=level else 0.0) for i in range(1,n+1)])
        assert ceiling+0.005>=float(score),(game,ceiling,score)
        ceilings.append(ceiling);scores.append(float(score))
    results[arm]={'observed_mean_rounded':sum(scores)/25,'same_completions_ceiling':sum(ceilings)/25,
                 'headroom':(sum(ceilings)-sum(scores))/25,'rounding_tolerance':0.005}
    claimed=json.loads((ROOT/'c/score-decomposition.json').read_text())[arm]
    assert abs(results[arm]['headroom']-claimed['mean_efficiency_headroom'])<1e-10
fades={}
for arm,expected in [('control',98.7),('control-r2',82.4)]:
    path=ARCH/f'runs/out-{arm}/dfranzen-m2-{arm}.log';inputs.append(path)
    times=[float(v) for v in re.findall(r'tail fade phase after ([\d.]+) minutes',path.read_text())]
    assert times and all(v==expected for v in times)
    fades[arm]=times[0]
leader=json.loads((ROOT/'sources/leaderboard.json').read_text())
assert leader['rows'][0]['teamName']=='Tufa Labs' and leader['rows'][0]['score']=='52.51'
assert leader['rows'][1]['score']=='48.07' and leader['rows'][2]['score']=='34.77'
# Check formula illustration, not an empirical confidence claim.
stat={'assumed_single_run_sd':3.93,'assumed_independent_difference_sd':(2**0.5)*3.93,
      'limit':'Screenshot cohort identity/selection unknown; no empirical inference from this formula'}
inputs += sorted(ROOT.glob('[abc]/*.md'))
inputs += [ROOT/'LEVER-MAP.md', ROOT/'SYNTHESIS.md', ROOT/'PLAN.md']
protected=REPO/'solvers/m2_memory'
for rel,expected in json.loads((protected/'evidence/verification.json').read_text())['files'].items():
    assert hashlib.sha256((protected/rel).read_bytes()).hexdigest()==expected,rel
output={'checks':'PASS','score_headroom':results,'observed_fade_minutes':fades,'leaderboard_snapshot':leader['retrieved_utc'],
        'statistics':stat,'solver_candidate_unchanged':True,
        'inputs':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
(ROOT/'verification.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k!='inputs'},indent=2))

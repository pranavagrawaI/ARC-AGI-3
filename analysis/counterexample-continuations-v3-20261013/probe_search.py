"""Diagnose the reminder's search bound using its actual state and frozen model."""
import ast
import copy
import hashlib
import importlib.util
import json
import re
from collections import deque
from pathlib import Path
import numpy as np
from arcengine import ActionInput, ARCBaseGame, GameAction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / 'analysis/s5i5-inspection-20261012'
ns = {'ast':ast, 'copy':copy, 're':re, 'deque':deque, 'COLORS':'OSYN'}


def definitions(path, namespace, names=None):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


definitions(ROOT/'analysis/score-conversion-20261013/run_probe.py', ns, {'extract','functions','search','reference'})
ns['records'] = json.loads((PRIOR/'records.json').read_text())
good = {'OBS': {(r,c) for r,row in enumerate(ns['records']['control-pc']['entry_board']) for c,col in enumerate(row)
                if col==15 or (r>=52 and col!=5)}}
ns['functions']('control-pc',4862,good)
ns['functions']('control-pc',5635,good)
ns['good']=good
observed=json.loads((PRIOR/'engine-replay.json').read_text())
source=Path(observed['engine_source'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==observed['engine_sha256']
spec=importlib.util.spec_from_file_location('search_diagnostic_native',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cls=next(v for v in vars(module).values() if isinstance(v,type) and issubclass(v,ARCBaseGame) and v is not ARCBaseGame)
helpers={'np':np,'ActionInput':ActionInput,'GameAction':GameAction,'cls':cls}
definitions(PRIOR/'replay.py',helpers,{'make','perform','state_from_board'})
episode='b1-2903-reminder'
directory=HERE/'output/continuations/episodes'/episode
game=helpers['make']()
for action in json.loads((directory/'actions.json').read_text()):
    frame=helpers['perform'](game,action)
start=helpers['state_from_board'](frame.frame[-1])
assert frame.levels_completed==1
searches=[ns['search'](start,horizon) for horizon in [24,40]]
chosen=searches[-1]
assert chosen['status']=='FOUND'
retained=json.loads((directory/'restoration.json').read_text())['retained_functions']
model={'copy':copy}
for name in ['gdir_of','cellsS','growth_region','static_set','adj','removed_cells','sim_step']:
    exec(compile(retained[name],f'{episode}:{name}','exec'),model)
click={'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
rows=[]
for i,token in enumerate(chosen['path']):
    before=helpers['state_from_board'](frame.frame[-1])
    predicted=model['sim_step'](copy.deepcopy(before),token[1],token[0],model['static_set']())
    frame=helpers['perform'](game,{'id':'ACTION6','data':{'x':click[token],'y':57}})
    board=game.inspection_wins[-1] if frame.levels_completed>1 else frame.frame[-1]
    actual=helpers['state_from_board'](board)
    rows.append({'step':i+1,'action':token,'input':before,'predicted':predicted,'actual':actual,'matched':predicted==actual})
assert frame.levels_completed==2
result={'episode':episode,'scope':'Offline native replay and diagnostic reference search; no model calls.',
        'start':start,'searches':searches,'native_solved':True,'route_length':len(rows),
        'frozen_model_matches':sum(r['matched'] for r in rows),'transitions':rows,
        'engine_sha256':observed['engine_sha256']}
(HERE/'transcript-audit/search-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='transitions'},indent=2))

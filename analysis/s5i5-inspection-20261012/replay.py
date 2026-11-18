"""Verify all four s5i5 action traces through L2 on the local engine."""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
from arcengine import ActionInput, ARCBaseGame, GameAction

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = Path('/home/pranav/projects/arc-agi-3-reference/competition-env/environment_files/s5i5/18d95033/s5i5.py')
R = json.loads((OUT/'records.json').read_text())
spec = importlib.util.spec_from_file_location('s5i5_replay', SRC)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cls = next(v for v in vars(module).values() if isinstance(v,type) and issubclass(v,ARCBaseGame) and v is not ARCBaseGame)


def state_from_board(board):
    a = np.array(board)
    result = {}
    for key,color in [('O',12),('S',10),('Y',11),('N',14)]:
        pts = np.argwhere(a[:52,:]==color)
        assert len(pts)>0, key
        r0,c0 = pts.min(axis=0);r1,c1 = pts.max(axis=0)
        horizontal = key in ['O','Y']
        s = {'axis':'H' if horizontal else 'V','r':int(r0),'c':int(c0),
             'L':int((c1-c0 if horizontal else r1-r0)+1)}
        if not horizontal:s['dir']=1 if key=='N' else -1
        result[key]=s
    return result


def tip(board):
    s = state_from_board(board)['N']
    return [s['r']+s['L']-2,s['c']+1]


def make():
    g=cls();g.perform_action(ActionInput(id=GameAction.RESET))
    g.inspection_wins=[]
    original=g.neurwiqfry
    def checked():
        result=original()
        if result:g.inspection_wins.append(g.camera.render(g.current_level.get_sprites()).tolist())
        return result
    g.neurwiqfry=checked
    return g


def perform(g,a):
    return g.perform_action(ActionInput(id=GameAction[a['id']],data=a.get('data') or {}))


audit={'engine_source':str(SRC),'engine_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'runs':{},'counterfactuals':[]}
prefixes,segments={},{}
for label,rec in R.items():
    p=ROOT/Path(rec['source']).parent.parent
    b=next(r for r in json.loads((p/'benchmark.json').read_text())['game_runs'] if r['game_id']=='s5i5-18d95033')
    events=[json.loads(l) for l in (p/'artifacts/s5i5-18d95033_p0_events.jsonl').open()]
    acts=[e for e in events if e['type']=='action']
    k,end=b['actions_per_level'][0],sum(b['actions_per_level'][:2])
    prefixes[label]=[h['action'] for h in b['history'][:k]]
    segments[label]=[h['action'] for h in b['history'][k:end]]
    g=make();states=[]
    for i,h in enumerate(b['history'][:end]):
        fr=perform(g,h['action'])
        assert fr.frame[-1]==acts[i]['board'],(label,i,'frame mismatch')
        assert fr.levels_completed==acts[i]['score'],(label,i,'score mismatch')
        if i>=k-1:
            board=g.inspection_wins[-1] if fr.levels_completed>1 else fr.frame[-1]
            states.append({'local_action':i-k+1,'levels_completed':fr.levels_completed,
                           'state':state_from_board(board),'tip_row_col':tip(board),
                           'board':board})
    assert states[-1]['tip_row_col']==[31,52] if rec['solved'] else states[-1]['tip_row_col']!=[31,52]
    audit['runs'][label]={'verified_actions':end,'all_frames_match':True,'states':states,'winning_boards':g.inspection_wins}
    print(label,'verified',end,'solves L2',fr.levels_completed>1,'end tip',states[-1]['tip_row_col'])

# Both successful action segments solve after either B1 run's L1 prefix.
for failed in ['b1-r1','b1-r2']:
    for solved in ['control-pc','evidence-pt']:
        g=make()
        for a in prefixes[failed]+segments[solved]:fr=perform(g,a)
        assert fr.levels_completed==2,(failed,solved)
        audit['counterfactuals'].append({'prefix':failed,'segment':solved,'actions':len(segments[solved]),'solves':True})
(OUT/'engine-replay.json').write_text(json.dumps(audit,indent=2)+'\n')
print('route transfer checks:',len(audit['counterfactuals']))

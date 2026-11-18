"""Verify tn36 traces and log transient opcode effects before reset."""
from pathlib import Path
import hashlib
import importlib.util
import json

from arcengine import ActionInput,ARCBaseGame,GameAction

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SRC=Path('/home/pranav/projects/arc-agi-3-reference/competition-env/environment_files/tn36/ef4dde99/tn36.py')
R=json.loads((OUT/'records.json').read_text())
s=importlib.util.spec_from_file_location('tn36_audit',SRC);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
cls=next(v for v in vars(m).values() if isinstance(v,type) and issubclass(v,ARCBaseGame) and v is not ARCBaseGame)


def key_state(machine):
    key=machine.htntnzkbzu
    return {'x':int(key.x),'y':int(key.y),'rotation':int(key.rotation),'scale':int(key.scale)}


def make():
    g=cls();g.audit_commands=[]
    original=g.on_set_level
    def set_level(level):
        original(level)
        right=g.fdksqlmpki.bzirenxmrg
        for opcode,fn in list(right.okllwtboml.items()):
            def wrapped(fn=fn,opcode=opcode,right=right):
                before=key_state(right);before_shape=right.htntnzkbzu.axbjgpzkyi.render().tolist()
                fn();after=key_state(right);after_shape=right.htntnzkbzu.axbjgpzkyi.render().tolist()
                g.audit_commands.append({'opcode':opcode,'before':before,'after':after,
                                         'before_shape':before_shape,'after_shape':after_shape})
            right.okllwtboml[opcode]=wrapped
    g.on_set_level=set_level
    g.perform_action(ActionInput(id=GameAction.RESET))
    return g


audit={'engine_source':str(SRC),'engine_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'runs':{}}
for label,rec in R.items():
    p=ROOT/Path(rec['source']).parent.parent
    row=next(r for r in json.loads((p/'benchmark.json').read_text())['game_runs'] if r['game_id']=='tn36-ef4dde99')
    es=[json.loads(l) for l in (p/'artifacts/tn36-ef4dde99_p0_events.jsonl').open()]
    acts=[e for e in es if e['type']=='action']
    k,end=row['actions_per_level'][0],sum(row['actions_per_level'][:2])
    g=make();steps=[]
    for i,h in enumerate(row['history'][:end]):
        before=key_state(g.fdksqlmpki.bzirenxmrg)
        program=g.fdksqlmpki.bzirenxmrg.ukwrvhanub.vkuvtkaerv
        start=len(g.audit_commands)
        fr=g.perform_action(ActionInput(id=GameAction[h['action']['id']],data=h['action'].get('data') or {}))
        assert fr.frame[-1]==acts[i]['board'],(label,i,'frame mismatch')
        assert fr.levels_completed==acts[i]['score'],(label,i,'score mismatch')
        if i>=k:
            steps.append({'local_action':i-k+1,'global_action':i+1,'action':h['action'],
                          'program_before':program,'transient_commands':g.audit_commands[start:],
                          'before_key':before,'settled_key':key_state(g.fdksqlmpki.bzirenxmrg),
                          'level_after':fr.levels_completed+1})
            if label=='b1-r1' and i-k+1==41:
                steps[-1]['animation_frames']=fr.frame
    audit['runs'][label]={'verified_actions':end,'commands':steps,'solves_l2':fr.levels_completed>=2}
    print(label,'verified',end,'solves',fr.levels_completed>=2)

probe=next(a for a in audit['runs']['b1-r1']['commands'] if a['local_action']==41)
assert probe['program_before']==[16,0,0,0],probe['program_before']
operation=next(c for c in probe['transient_commands'] if c['opcode']==16)
assert operation['before']['x']==operation['after']['x'] and operation['before']['y']==operation['after']['y']
assert operation['before']['rotation']!=operation['after']['rotation']
shape=operation['after_shape'];x,y=operation['after']['x'],operation['after']['y']
visible=[i for i,frame in enumerate(probe['animation_frames'])
         if all(frame[y+r][x+c]==v for r,row in enumerate(shape) for c,v in enumerate(row) if v>=0)]
assert visible,'Rotated shape not visible in captured animation'
audit['missed_rotation_probe']={'run':'b1-r1','local_action':41,'program':probe['program_before'],'operation':operation,
                               'rotated_shape_visible_in_animation_frames':visible,
                               'model_claim_source_line':10788,
                               'conclusion':'No translation was generalized to no effect, although the command rotates the key.'}
(OUT/'engine-replay.json').write_text(json.dumps(audit,indent=2)+'\n')
print('missed rotation:',json.dumps(operation))

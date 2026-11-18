"""Test whether independent switch effects compose additively."""
from pathlib import Path
import json
import replay
from arcengine import ActionInput,GameAction

OUT=Path(__file__).resolve().parent
row=next(r for r in json.loads((replay.ROOT/'solvers/m2_b1/experiment/pair-1/b1-r1/output/benchmark.json').read_text())['game_runs']
         if r['game_id']=='tn36-ef4dde99')
prefix=row['history'][:row['actions_per_level'][0]]
probes=[]
for opcode,clicks in [(1,[(33,39)]),(32,[(47,39)]),(33,[(33,39),(47,39)])]:
    game=replay.make()
    for h in prefix:
        game.perform_action(ActionInput(id=GameAction[h['action']['id']],data=h['action'].get('data') or {}))
    assert game.fdksqlmpki.bzirenxmrg.ukwrvhanub.vkuvtkaerv==[0,0,0,0]
    for y,x in clicks:game.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':x,'y':y}))
    assert game.fdksqlmpki.bzirenxmrg.ukwrvhanub.vkuvtkaerv==[opcode,0,0,0]
    before=replay.key_state(game.fdksqlmpki.bzirenxmrg);offset=len(game.audit_commands)
    game.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':47,'y':58}))
    ops=game.audit_commands[offset:]
    operation=next((op for op in ops if op['opcode']==opcode),None)
    delta=[0,0] if operation is None else [operation['after']['y']-operation['before']['y'],operation['after']['x']-operation['before']['x']]
    probes.append({'opcode':opcode,'clicks_row_col':clicks,'delta_row_col':delta,'operation':operation})
assert [p['delta_row_col'] for p in probes]==[[0,-4],[0,0],[-4,0]]
report={'probes':probes,'independent_bit_additivity_falsified':True,
        'conclusion':'The column is a compound instruction: bit 32 alone is a no-op; combining it with bit 1 changes LEFT into UP.',
        'scope':'Offline probes through ordinary mouse actions, from a recorded L2 entry. No model saw the probes.'}
(OUT/'encoding-check.json').write_text(json.dumps(report,indent=2)+'\n')
print('observed effects:',[p['delta_row_col'] for p in probes])
